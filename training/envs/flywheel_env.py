"""Vectorized MuJoCo flywheel environment for RSL-RL."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from flywheel_rewards import (
    FLOOR_PENALTY,
    HIT_BONUS,
    timeout_terminal_reward,
    dense_distance_reward,
    dense_aim_reward,
    aim_spawn_bonus,
    AIM_ANGLE_SCALE_RAD,
    compute_target_distances,
    impact_distance_on_face,
)
from rsl_rl.env import VecEnv

HIDDEN_BALL_POS = np.array([0.0, 0.0, -10.0], dtype=np.float64)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "flywheel_test.xml")

# Full (hardest) target randomization envelope. Curriculum scale interpolates
# from a small, easy envelope centered near the shooter up to this full range.
TARGET_XY_RANGE = 10.0
TARGET_Z_MIN = 2.0
TARGET_Z_RANGE = 8.0
# Observation-only scale for distance features (not part of the reward). Kept at
# the old 12m reference so obs magnitudes stay in a similar range after the
# reward switched from linear to exp(-d).
OBS_DISTANCE_SCALE = 12.0


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
    target_lateral_half_extent: float
    ball_radius: float
    floor_z: float
    ball_qpos_adr: int
    ball_dof_adr: int
    spawn_local: np.ndarray


class FlywheelVecEnv(VecEnv):
    """Parallel flywheel shooter environments."""

    cfg = {
        "model_path": MODEL_PATH,
        # Minimum control periods of flywheel spin-up before spawn is allowed.
        # With frame_skip=5 and dt=0.002, 100 steps ≈ 1s — needed for gear=1800
        # to reach curriculum Z. Actual spawn may wait longer for aim gate.
        "spawn_delay_steps": 100,
        # Force spawn by this step even if still off-boresight (avoid infinite wait).
        "spawn_max_wait_steps": 250,
        # Only release the ball once |aim_err| is under this (≈15°), matching the
        # diagnosis band where deterministic hit rate jumped to ~39%.
        "aim_spawn_max_rad": AIM_ANGLE_SCALE_RAD,
        "flywheel_spawn_ctrl": -1.0,
        # Physics substeps per RL action. timestep=0.002 → 10ms control period.
        # frame_skip=1 (old) let the policy retune voltage every 2-3ms, before
        # contact had finished resolving, which favored chatter/oscillation.
        "frame_skip": 5,
    }

    def __init__(
        self,
        num_envs: int = 16,
        device: str = "cpu",
        # Counted in *control* steps (each runs frame_skip physics steps).
        # 900 * 5 * 0.002s ≈ 9s of sim time -- same flight budget as the old
        # 3000 * 0.003s episodes, enough for 25-38 m/s launches to land.
        max_episode_length: int = 900,
        seed: int | None = 0,
        episode_logger=None,
        curriculum_enabled: bool = False,
        curriculum_start_scale: float = 0.3,
        curriculum_max_scale: float = 1.0,
        curriculum_step: float = 0.05,
        curriculum_window: int = 200,
        curriculum_hit_threshold: float = 0.15,
    ) -> None:
        self.num_envs = num_envs
        self.num_actions = 2
        self.device = device
        self.max_episode_length = max_episode_length
        self.frame_skip = int(self.cfg["frame_skip"])
        self.episode_logger = episode_logger

        # Curriculum learning: start targets close/easy and widen the randomization
        # envelope once the policy is actually hitting often enough, rather than
        # forcing it to solve full-range precision aiming from episode 1. Disabled
        # by default so eval/distillation always see the full (hardest) envelope.
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_scale = curriculum_start_scale if curriculum_enabled else curriculum_max_scale
        self.curriculum_max_scale = curriculum_max_scale
        self.curriculum_step = curriculum_step
        self.curriculum_hit_threshold = curriculum_hit_threshold
        self._curriculum_outcomes: deque[float] = deque(maxlen=curriculum_window)

        self.rng = np.random.default_rng(seed)
        self.handles = [self._make_handles() for _ in range(num_envs)]

        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ball_spawned = np.zeros(num_envs, dtype=bool)
        # Tracks the best (smallest) miss_distance seen during flight. miss_distance
        # accounts for remaining depth-to-target until the ball passes the target's
        # plane, unlike pure lateral offset, which is ~0 at the muzzle by construction
        # (the target is always oriented to face the shooter spawn point) and would
        # otherwise saturate the reward on the very first post-spawn step.
        # Tracked for diagnostics/logging only (episode CSVs, curriculum) -- no
        # longer part of the reward calculation itself.
        self.best_miss = np.full(num_envs, np.inf, dtype=np.float64)
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
            target_lateral_half_extent=float(np.linalg.norm(model.geom_size[target_geom_id][:2])),
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
                # Teach yaw alignment before launch (dominant miss mode).
                aim_err = self._aim_error_rad(handles)
                rewards[env_id] += dense_aim_reward(aim_err)
                extras["log"]["/aim/error_rad"] = abs(aim_err)
                t = int(self.episode_length_buf[env_id].item())
                min_spin = int(self.cfg["spawn_delay_steps"])
                max_wait = int(self.cfg["spawn_max_wait_steps"])
                aim_gate = float(self.cfg["aim_spawn_max_rad"])
                if t >= min_spin and (abs(aim_err) <= aim_gate or t >= max_wait):
                    self._spawn_ball(handles)
                    self.ball_spawned[env_id] = True
                    rewards[env_id] += aim_spawn_bonus(aim_err)
                    extras["log"]["/aim/spawn_err_rad"] = abs(aim_err)
                    extras["log"]["/aim/spawn_gated"] = 1.0 if abs(aim_err) <= aim_gate else 0.0

            if self.ball_spawned[env_id]:
                rewards[env_id] += self._dense_reward(env_id, handles)

            # Hold ctrl fixed across frame_skip physics steps so contact can
            # resolve before the next policy action.
            for _ in range(self.frame_skip):
                mujoco.mj_step(handles.model, handles.data)
                if self.ball_spawned[env_id] and not dones[env_id]:
                    terminal_reward, done, outcome, impact_lateral = self._terminal_step(env_id, handles)
                    rewards[env_id] += terminal_reward
                    if done:
                        dones[env_id] = True
                        extras.setdefault("episode", {})
                        extras["log"][f"/episode/outcome_{outcome}"] = 1.0
                        episode_length = int(self.episode_length_buf[env_id].item()) + 1
                        best_miss = self._finite_best_miss(env_id)
                        self._record_episode_done(
                            extras,
                            env_id,
                            outcome,
                            impact_lateral=impact_lateral,
                            best_miss=best_miss,
                            episode_length=episode_length,
                        )
                        if self.episode_logger is not None:
                            self.episode_logger.record(
                                env_id,
                                outcome,
                                episode_length,
                                impact_lateral,
                                best_miss,
                                curriculum_scale=self.curriculum_scale,
                            )
                        self._update_curriculum(outcome == "hit")
                        break

            self.episode_length_buf[env_id] += 1

            if self.episode_length_buf[env_id].item() >= self.max_episode_length and not dones[env_id]:
                # Pay miss-quality terminal so a long near-miss beats dumping on
                # the floor for a short episode (see TIMEOUT_MISS_BONUS).
                best_miss = self._finite_best_miss(env_id)
                rewards[env_id] += timeout_terminal_reward(best_miss)
                dones[env_id] = True
                extras["log"]["/episode/timeout"] = 1.0
                episode_length = int(self.episode_length_buf[env_id].item())
                self._record_episode_done(
                    extras,
                    env_id,
                    "timeout",
                    best_miss=best_miss,
                    episode_length=episode_length,
                )
                if self.episode_logger is not None:
                    self.episode_logger.record(
                        env_id,
                        "timeout",
                        episode_length,
                        None,
                        best_miss,
                        curriculum_scale=self.curriculum_scale,
                    )
                self._update_curriculum(False)

            if dones[env_id]:
                self._reset_env(env_id)

        obs = self._build_obs()
        reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        extras["log"]["/curriculum/scale"] = self.curriculum_scale
        return obs, reward_t, done_t, extras

    def _update_curriculum(self, is_hit: bool) -> None:
        if not self.curriculum_enabled or self.curriculum_scale >= self.curriculum_max_scale:
            return
        self._curriculum_outcomes.append(1.0 if is_hit else 0.0)
        if len(self._curriculum_outcomes) < self._curriculum_outcomes.maxlen:
            return
        hit_rate = sum(self._curriculum_outcomes) / len(self._curriculum_outcomes)
        if hit_rate >= self.curriculum_hit_threshold:
            self.curriculum_scale = min(self.curriculum_max_scale, self.curriculum_scale + self.curriculum_step)
            # Reset the window so we judge the *new* difficulty level fresh, instead
            # of immediately ratcheting again on stale (easier-level) data.
            self._curriculum_outcomes.clear()

    def _reset_all(self) -> None:
        for env_id in range(self.num_envs):
            self._reset_env(env_id)

    def _reset_env(self, env_id: int) -> None:
        handles = self.handles[env_id]
        scale = self.curriculum_scale
        # Anchoring z at the *closest/lowest* corner made "easy" mode mean close, low
        # targets -- which need a flat, floor-skimming shot and turned out to be *more*
        # prone to clipping the floor than a farther/higher target's arcing shot, not
        # less. Centering on the middle of the full range and expanding symmetrically
        # avoids starting the curriculum at that physically awkward extreme.
        target_z_center = TARGET_Z_MIN + TARGET_Z_RANGE / 2.0
        random_x = self.rng.uniform(-TARGET_XY_RANGE * scale, TARGET_XY_RANGE * scale)
        random_y = self.rng.uniform(-TARGET_XY_RANGE * scale, TARGET_XY_RANGE * scale)
        random_z = self.rng.uniform(
            target_z_center - (TARGET_Z_RANGE / 2.0) * scale,
            target_z_center + (TARGET_Z_RANGE / 2.0) * scale,
        )
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
        self.best_miss[env_id] = np.inf

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

    def _aim_error_rad(self, handles: EnvHandles) -> float:
        """Signed off-boresight angle of the target in the hood frame (radians).

        Barrel aims roughly +Y (spawn site y≈1.2). Remaining yaw to put the
        target on the hood midplane is atan2(local_x, local_y).
        """
        hood_rot = handles.data.xmat[handles.hood_id].reshape(3, 3)
        hood_pos = handles.data.xpos[handles.hood_id]
        target_pos = handles.data.xpos[handles.body_id]
        target_local = hood_rot.T @ (target_pos - hood_pos)
        return float(np.arctan2(target_local[0], target_local[1]))

    def _dense_reward(self, env_id: int, handles: EnvHandles) -> float:
        """Per-step reward from exponential clamped distance score."""
        ball_pos = handles.data.xpos[handles.ball_id].copy()
        slab_pos = handles.data.xpos[handles.body_id].copy()
        plane_normal = handles.data.xmat[handles.body_id].reshape(3, 3)[:, 2]
        # Use miss_distance (not lateral_distance): lateral_distance is the purely
        # tangential offset from the target's boresight, which is ~0 right at the
        # muzzle by construction and would saturate the reward immediately. miss_distance
        # also accounts for the remaining depth to the target until the ball passes it.
        miss_distance, _, _ = compute_target_distances(
            ball_pos, slab_pos, plane_normal, handles.target_half_thickness
        )

        self.best_miss[env_id] = min(self.best_miss[env_id], miss_distance)
        return dense_distance_reward(miss_distance)

    def _terminal_step(
        self, env_id: int, handles: EnvHandles
    ) -> tuple[float, bool, str, float | None]:
        ball_pos = handles.data.xpos[handles.ball_id].copy()
        slab_pos = handles.data.xpos[handles.body_id].copy()
        plane_normal = handles.data.xmat[handles.body_id].reshape(3, 3)[:, 2]
        signed_distance = np.dot(ball_pos - slab_pos, plane_normal)
        crossed_front_face_plane = signed_distance <= handles.target_half_thickness + handles.ball_radius

        if self._ball_contacts_target(handles):
            # Flat bonus -- a hit is a hit, regardless of precision. impact_distance
            # is still returned for logging/diagnostics (e.g. episode CSVs), just not
            # used to scale the reward.
            impact_distance = impact_distance_on_face(
                self.pre_step_ball_pos[env_id],
                slab_pos,
                plane_normal,
                handles.target_half_thickness,
            )
            return HIT_BONUS, True, "hit", impact_distance

        if crossed_front_face_plane:
            # The plane extends infinitely, so only treat this as a hit if the
            # ball actually crossed within the target's footprint (not just
            # anywhere along the same depth). This also catches high-speed
            # tunneling through the thin target that mujoco's contact solver
            # might miss in a single step.
            impact_distance = impact_distance_on_face(
                self.pre_step_ball_pos[env_id],
                slab_pos,
                plane_normal,
                handles.target_half_thickness,
            )
            hit_radius = handles.target_lateral_half_extent + handles.ball_radius
            if impact_distance <= hit_radius:
                return HIT_BONUS, True, "hit", impact_distance
            # Fly-by miss: end the episode here. Previously these kept running to
            # max_episode_length (~900 control steps), so a short miss collected thousands of
            # extra dense-reward steps and the policy learned to coast to timeout
            # instead of aiming for hits.
            return timeout_terminal_reward(impact_distance), True, "miss", impact_distance

        if ball_pos[2] <= handles.floor_z + handles.ball_radius:
            # Flat penalty -- hitting the floor always costs the same, regardless
            # of how close the ball got.
            return -FLOOR_PENALTY, True, "floor", None

        return 0.0, False, "running", None

    def _finite_best_miss(self, env_id: int) -> float | None:
        best_miss = self.best_miss[env_id]
        if np.isinf(best_miss):
            return None
        return float(best_miss)

    @staticmethod
    def _record_episode_done(
        extras: dict,
        env_id: int,
        outcome: str,
        *,
        impact_lateral: float | None = None,
        best_miss: float | None = None,
        episode_length: int | None = None,
    ) -> None:
        extras.setdefault("episode_done", []).append(
            {
                "env_id": env_id,
                "outcome": outcome,
                "episode_length": episode_length,
                "impact_lateral": impact_lateral,
                "best_miss": best_miss,
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
            # The yaw hinge has no travel limit, so its raw qpos is a *cumulative*
            # rotation count, not a bounded angle -- it just keeps growing/shrinking
            # forever as the motor spins. Left unwrapped, this feature is wildly
            # non-stationary: over a long run it drifted to a running mean of ~-973pi
            # radians (~487 turns), which silently poisoned the observation normalizer
            # (fine while training continuously and adapting online, but catastrophic
            # the moment a checkpoint is reloaded into a fresh env that legitimately
            # starts near yaw=0 -- the policy saw a wildly out-of-distribution input
            # and degenerated to dropping the ball immediately). Wrapping to (-pi, pi]
            # here keeps the observation bounded and physically meaningful regardless
            # of how many total turns the joint has made.
            raw_yaw_angle = handles.data.qpos[handles.model.jnt_qposadr[yaw_joint]]
            yaw_angle = ((raw_yaw_angle + np.pi) % (2 * np.pi)) - np.pi

            target_pos = handles.data.xpos[handles.body_id]
            target_local = hood_rot.T @ (target_pos - hood_pos)
            plane_normal = handles.data.xmat[handles.body_id].reshape(3, 3)[:, 2]
            miss_distance, _, face_distance = compute_target_distances(
                ball_pos, target_pos, plane_normal, handles.target_half_thickness
            )

            best_miss = self.best_miss[env_id]
            if np.isinf(best_miss):
                best_miss = miss_distance

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
                    best_miss / OBS_DISTANCE_SCALE,
                    float(self.ball_spawned[env_id]),
                ],
                dtype=np.float32,
            )
            privileged[env_id] = np.array(
                [
                    target_local[0] / 15.0,
                    target_local[1] / 15.0,
                    target_local[2] / 15.0,
                    face_distance / OBS_DISTANCE_SCALE,
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
