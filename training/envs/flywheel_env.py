"""Vectorized MuJoCo flywheel environment for RSL-RL."""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from flywheel_rewards import (
    DENSE_REWARD_SCALE,
    compute_closeness,
    compute_hit_score,
    compute_miss_score,
    compute_target_reward,
    impact_distance_on_face,
)
from rsl_rl.env import VecEnv

HIDDEN_BALL_POS = np.array([0.0, 0.0, -10.0], dtype=np.float64)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "flywheel_test.xml")


@dataclass
class EnvHandles:
    model: mujoco.MjModel
    data: mujoco.MjData
    flywheel_actuator: int
    yaw_actuator: int
    hood_id: int
    body_id: int
    ball_id: int
    ball_geom_id: int
    target_geom_id: int
    target_half_thickness: float
    ball_radius: float
    floor_z: float
    ball_qpos_adr: int
    ball_dof_adr: int
    spawn_local: np.ndarray


class FlywheelVecEnv(VecEnv):
    """Parallel flywheel shooter environments."""

    cfg = {
        "model_path": MODEL_PATH,
        "spawn_delay_steps": 20,
        "flywheel_spawn_ctrl": -1.0,
    }

    def __init__(
        self,
        num_envs: int = 16,
        device: str = "cpu",
        max_episode_length: int = 600,
        seed: int | None = 0,
    ) -> None:
        self.num_envs = num_envs
        self.num_actions = 2
        self.device = device
        self.max_episode_length = max_episode_length

        self.rng = np.random.default_rng(seed)
        self.handles = [self._make_handles() for _ in range(num_envs)]

        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ball_spawned = np.zeros(num_envs, dtype=bool)
        self.best_lateral = np.full(num_envs, np.inf, dtype=np.float64)
        self.best_miss = np.full(num_envs, np.inf, dtype=np.float64)
        self.prev_accuracy = np.zeros(num_envs, dtype=np.float64)
        self.pre_step_ball_pos = np.zeros((num_envs, 3), dtype=np.float64)

        self._reset_all()

    def _make_handles(self) -> EnvHandles:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)

        hood_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "launcher_middle_hood")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "platform_body")
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "random_platform")
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ball_geom_id = model.body_geomadr[ball_id]
        ball_joint_id = model.body_jntadr[ball_id]

        spawn_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "shooter_spawn")
        spawn_local = model.site_pos[spawn_site_id].copy()

        return EnvHandles(
            model=model,
            data=data,
            flywheel_actuator=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "flywheel_motor"),
            yaw_actuator=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor"),
            hood_id=hood_id,
            body_id=body_id,
            ball_id=ball_id,
            ball_geom_id=ball_geom_id,
            target_geom_id=target_geom_id,
            target_half_thickness=float(model.geom_size[target_geom_id][2]),
            ball_radius=float(model.geom_size[ball_geom_id][0]),
            floor_z=float(model.geom_pos[floor_geom_id][2]),
            ball_qpos_adr=int(model.jnt_qposadr[ball_joint_id]),
            ball_dof_adr=int(model.jnt_dofadr[ball_joint_id]),
            spawn_local=spawn_local,
        )

    def get_observations(self) -> TensorDict:
        return self._build_obs()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        extras = {"log": {}}

        for env_id, handles in enumerate(self.handles):
            self._apply_action(handles, actions_np[env_id])
            self.pre_step_ball_pos[env_id] = handles.data.xpos[handles.ball_id].copy()

            if not self.ball_spawned[env_id]:
                self._hide_ball(handles)
            elif self.episode_length_buf[env_id].item() >= self.cfg["spawn_delay_steps"]:
                pass

            if (
                not self.ball_spawned[env_id]
                and self.episode_length_buf[env_id].item() >= self.cfg["spawn_delay_steps"]
            ):
                self._spawn_ball(handles)
                self.ball_spawned[env_id] = True
                self.prev_accuracy[env_id] = 0.0

            if self.ball_spawned[env_id]:
                rewards[env_id] += self._dense_reward(env_id, handles)

            mujoco.mj_step(handles.model, handles.data)
            self.episode_length_buf[env_id] += 1

            if self.ball_spawned[env_id]:
                terminal_reward, done, outcome, impact_lateral = self._terminal_step(env_id, handles)
                rewards[env_id] += terminal_reward
                if done:
                    dones[env_id] = True
                    extras.setdefault("episode", {})
                    extras["log"][f"/episode/outcome_{outcome}"] = 1.0
                    self._record_episode_done(
                        extras,
                        env_id,
                        outcome,
                        impact_lateral=impact_lateral,
                        best_lateral=self._finite_best_lateral(env_id),
                        episode_length=int(self.episode_length_buf[env_id].item()),
                    )

            if self.episode_length_buf[env_id].item() >= self.max_episode_length and not dones[env_id]:
                rewards[env_id] += compute_miss_score(self.best_lateral[env_id])
                dones[env_id] = True
                extras["log"]["/episode/timeout"] = 1.0
                self._record_episode_done(
                    extras,
                    env_id,
                    "timeout",
                    best_lateral=self._finite_best_lateral(env_id),
                    episode_length=int(self.episode_length_buf[env_id].item()),
                )

            if dones[env_id]:
                self._reset_env(env_id)

        obs = self._build_obs()
        reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        return obs, reward_t, done_t, extras

    def _reset_all(self) -> None:
        for env_id in range(self.num_envs):
            self._reset_env(env_id)

    def _reset_env(self, env_id: int) -> None:
        handles = self.handles[env_id]
        random_x = self.rng.uniform(-10.0, 10.0)
        random_y = self.rng.uniform(-10.0, 10.0)
        random_z = self.rng.uniform(2.0, 10.0)
        target_pos = np.array([random_x, random_y, random_z], dtype=np.float64)

        handles.model.body_pos[handles.body_id] = target_pos
        mujoco.mj_forward(handles.model, handles.data)
        shooter_pos = self._hood_local_to_world(handles, handles.spawn_local)
        self._orient_target_toward_shooter(handles, target_pos, shooter_pos)

        self._hide_ball(handles)
        handles.data.ctrl[handles.flywheel_actuator] = self.cfg["flywheel_spawn_ctrl"]
        handles.data.ctrl[handles.yaw_actuator] = 0.0
        mujoco.mj_forward(handles.model, handles.data)

        self.episode_length_buf[env_id] = 0
        self.ball_spawned[env_id] = False
        self.best_lateral[env_id] = np.inf
        self.best_miss[env_id] = np.inf
        self.prev_accuracy[env_id] = 0.0

    def _apply_action(self, handles: EnvHandles, action: np.ndarray) -> None:
        handles.data.ctrl[handles.flywheel_actuator] = np.clip(action[0], -1.0, 1.0)
        handles.data.ctrl[handles.yaw_actuator] = np.clip(action[1], -1.0, 1.0)

    def _hide_ball(self, handles: EnvHandles) -> None:
        adr = handles.ball_qpos_adr
        handles.data.qpos[adr : adr + 3] = HIDDEN_BALL_POS
        handles.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        handles.data.qvel[handles.ball_dof_adr : handles.ball_dof_adr + 6] = 0.0
        handles.model.geom_contype[handles.ball_geom_id] = 0
        handles.model.geom_conaffinity[handles.ball_geom_id] = 0

    def _spawn_ball(self, handles: EnvHandles) -> None:
        spawn_pos = self._hood_local_to_world(handles, handles.spawn_local)
        adr = handles.ball_qpos_adr
        handles.data.qpos[adr : adr + 3] = spawn_pos
        handles.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        handles.data.qvel[handles.ball_dof_adr : handles.ball_dof_adr + 6] = 0.0
        handles.model.geom_contype[handles.ball_geom_id] = 1
        handles.model.geom_conaffinity[handles.ball_geom_id] = 1
        mujoco.mj_forward(handles.model, handles.data)

    def _dense_reward(self, env_id: int, handles: EnvHandles) -> float:
        ball_pos = handles.data.xpos[handles.ball_id].copy()
        slab_pos = handles.data.xpos[handles.body_id].copy()
        plane_normal = handles.data.xmat[handles.body_id].reshape(3, 3)[:, 2]
        _, lateral_distance, _ = compute_closeness(
            ball_pos, slab_pos, plane_normal, handles.target_half_thickness
        )

        self.best_lateral[env_id] = min(self.best_lateral[env_id], lateral_distance)
        accuracy = compute_target_reward(lateral_distance)
        progress = max(0.0, accuracy - self.prev_accuracy[env_id])
        self.prev_accuracy[env_id] = accuracy
        return DENSE_REWARD_SCALE * progress

    def _terminal_step(
        self, env_id: int, handles: EnvHandles
    ) -> tuple[float, bool, str, float | None]:
        ball_pos = handles.data.xpos[handles.ball_id].copy()
        slab_pos = handles.data.xpos[handles.body_id].copy()
        plane_normal = handles.data.xmat[handles.body_id].reshape(3, 3)[:, 2]
        signed_distance = np.dot(ball_pos - slab_pos, plane_normal)
        hit_front_face = signed_distance <= handles.target_half_thickness + handles.ball_radius

        if self._ball_contacts_target(handles) or hit_front_face:
            impact_distance = impact_distance_on_face(
                self.pre_step_ball_pos[env_id],
                slab_pos,
                plane_normal,
                handles.target_half_thickness,
            )
            return compute_hit_score(impact_distance), True, "hit", impact_distance

        if ball_pos[2] <= handles.floor_z + handles.ball_radius:
            return compute_miss_score(self.best_lateral[env_id]), True, "floor", None

        return 0.0, False, "running", None

    def _finite_best_lateral(self, env_id: int) -> float | None:
        best_lateral = self.best_lateral[env_id]
        if np.isinf(best_lateral):
            return None
        return float(best_lateral)

    @staticmethod
    def _record_episode_done(
        extras: dict,
        env_id: int,
        outcome: str,
        *,
        impact_lateral: float | None = None,
        best_lateral: float | None = None,
        episode_length: int | None = None,
    ) -> None:
        extras.setdefault("episode_done", []).append(
            {
                "env_id": env_id,
                "outcome": outcome,
                "episode_length": episode_length,
                "impact_lateral": impact_lateral,
                "best_lateral": best_lateral,
            }
        )

    def _ball_contacts_target(self, handles: EnvHandles) -> bool:
        for i in range(handles.data.ncon):
            contact = handles.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if geoms == {handles.ball_geom_id, handles.target_geom_id}:
                return True
        return False

    def _build_obs(self) -> TensorDict:
        policy = np.zeros((self.num_envs, 10), dtype=np.float32)
        privileged = np.zeros((self.num_envs, 4), dtype=np.float32)

        for env_id, handles in enumerate(self.handles):
            hood_rot = handles.data.xmat[handles.hood_id].reshape(3, 3)
            hood_pos = handles.data.xpos[handles.hood_id]
            ball_pos = handles.data.xpos[handles.ball_id]
            ball_vel = handles.data.qvel[handles.ball_dof_adr : handles.ball_dof_adr + 3]
            ball_local = hood_rot.T @ (ball_pos - hood_pos)
            ball_vel_local = hood_rot.T @ ball_vel

            flywheel_joint = mujoco.mj_name2id(handles.model, mujoco.mjtObj.mjOBJ_JOINT, "flywheel_joint")
            yaw_joint = mujoco.mj_name2id(handles.model, mujoco.mjtObj.mjOBJ_JOINT, "yaw joint")
            flywheel_vel = handles.data.qvel[handles.model.jnt_dofadr[flywheel_joint]]
            yaw_angle = handles.data.qpos[handles.model.jnt_qposadr[yaw_joint]]

            target_pos = handles.data.xpos[handles.body_id]
            target_local = hood_rot.T @ (target_pos - hood_pos)
            plane_normal = handles.data.xmat[handles.body_id].reshape(3, 3)[:, 2]
            _, lateral_distance, face_distance = compute_closeness(
                ball_pos, target_pos, plane_normal, handles.target_half_thickness
            )

            best_lateral = self.best_lateral[env_id]
            if np.isinf(best_lateral):
                best_lateral = lateral_distance

            policy[env_id] = np.array(
                [
                    flywheel_vel / 100.0,
                    yaw_angle / np.pi,
                    ball_local[0] / 10.0,
                    ball_local[1] / 10.0,
                    ball_local[2] / 10.0,
                    ball_vel_local[0] / 20.0,
                    ball_vel_local[1] / 20.0,
                    ball_vel_local[2] / 20.0,
                    best_lateral / 6.0,
                    float(self.ball_spawned[env_id]),
                ],
                dtype=np.float32,
            )
            privileged[env_id] = np.array(
                [
                    target_local[0] / 15.0,
                    target_local[1] / 15.0,
                    target_local[2] / 15.0,
                    face_distance / 6.0,
                ],
                dtype=np.float32,
            )

        return TensorDict(
            {
                "policy": torch.as_tensor(policy, device=self.device),
                "privileged": torch.as_tensor(privileged, device=self.device),
            },
            batch_size=[self.num_envs],
        )

    @staticmethod
    def _hood_local_to_world(handles: EnvHandles, local_pos: np.ndarray) -> np.ndarray:
        rotation = handles.data.xmat[handles.hood_id].reshape(3, 3)
        return handles.data.xpos[handles.hood_id] + rotation @ local_pos

    @staticmethod
    def _orient_target_toward_shooter(handles: EnvHandles, target_pos: np.ndarray, shooter_pos: np.ndarray) -> None:
        forward = shooter_pos - target_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return

        z_axis = forward / forward_norm
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])

        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.flatten())
        handles.model.body_quat[handles.body_id] = quat
        mujoco.mj_forward(handles.model, handles.data)
