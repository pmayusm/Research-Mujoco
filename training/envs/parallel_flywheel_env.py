"""Process-parallel wrapper around FlywheelVecEnv chunks."""

from __future__ import annotations

import multiprocessing as mp
from collections import deque

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from training.envs.mujoco_worker import flywheel_worker


def _split_counts(total: int, num_workers: int) -> list[int]:
    base, rem = divmod(total, num_workers)
    return [base + (1 if i < rem else 0) for i in range(num_workers)]


class ParallelFlywheelVecEnv(VecEnv):
    """Flywheel VecEnv that steps MuJoCo chunks in worker processes.

    Wall-clock speedup comes from true multi-core mj_step. Threads do not help
    here (MuJoCo work stays effectively serialized under the GIL for this stack).
    """

    cfg = {
        "model_path": None,  # unused; workers load flywheel_test.xml themselves
        "spawn_delay_steps": 100,
        "spawn_max_wait_steps": 250,
        "flywheel_spawn_ctrl": -1.0,
    }

    def __init__(
        self,
        num_envs: int = 32,
        num_workers: int | None = None,
        device: str = "cpu",
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
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self.num_envs = num_envs
        self.num_actions = 2
        self.device = device
        self.max_episode_length = max_episode_length
        self.episode_logger = episode_logger

        self.curriculum_enabled = curriculum_enabled
        self.curriculum_scale = curriculum_start_scale if curriculum_enabled else curriculum_max_scale
        self.curriculum_max_scale = curriculum_max_scale
        self.curriculum_step = curriculum_step
        self.curriculum_hit_threshold = curriculum_hit_threshold
        self._curriculum_outcomes: deque[float] = deque(maxlen=curriculum_window)

        if num_workers is None:
            # Leave a couple cores for the learner / OS on laptop-class machines.
            cpu = mp.cpu_count() or 4
            num_workers = max(1, min(num_envs, max(1, cpu - 2)))
        num_workers = max(1, min(int(num_workers), num_envs))
        self.num_workers = num_workers
        self._local_counts = _split_counts(num_envs, num_workers)

        ctx = mp.get_context("spawn")
        self._remotes = []
        self._processes = []
        base_seed = 0 if seed is None else int(seed)

        offset = 0
        for worker_id, local_n in enumerate(self._local_counts):
            if local_n <= 0:
                continue
            parent_remote, child_remote = ctx.Pipe()
            proc = ctx.Process(
                target=flywheel_worker,
                name=f"flywheel-worker-{worker_id}",
                args=(child_remote, parent_remote),
                kwargs={
                    "num_local_envs": local_n,
                    "seed": base_seed + 1000 * worker_id + offset,
                    "max_episode_length": max_episode_length,
                    "device": "cpu",  # physics workers stay on CPU
                    "curriculum_enabled": curriculum_enabled,
                    "curriculum_start_scale": self.curriculum_scale,
                    "curriculum_max_scale": curriculum_max_scale,
                    "curriculum_step": curriculum_step,
                    "curriculum_window": curriculum_window,
                    "curriculum_hit_threshold": curriculum_hit_threshold,
                },
                daemon=True,
            )
            proc.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._processes.append(proc)
            offset += local_n

        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._last_broadcast_scale = None
        # Sync initial observations / episode lengths from workers.
        self._broadcast_curriculum_scale(force=True)
        self.get_observations()

    def get_observations(self) -> TensorDict:
        for remote in self._remotes:
            remote.send(("get_observations", None))
        chunks = [remote.recv() for remote in self._remotes]
        return self._stack_obs(chunks)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        offset = 0
        for remote, local_n in zip(self._remotes, self._local_counts):
            chunk = actions_np[offset : offset + local_n]
            remote.send(("step", chunk))
            offset += local_n

        chunks = [remote.recv() for remote in self._remotes]
        obs = self._stack_obs(chunks)
        rewards = torch.as_tensor(
            np.concatenate([c["rewards"] for c in chunks], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            np.concatenate([c["dones"] for c in chunks], axis=0),
            dtype=torch.bool,
            device=self.device,
        )

        extras: dict = {"log": {"/curriculum/scale": self.curriculum_scale}}
        env_offset = 0
        for chunk, local_n in zip(chunks, self._local_counts):
            for key, value in chunk.get("log", {}).items():
                extras["log"][key] = value
            for ep in chunk.get("episode_done", []):
                global_env_id = env_offset + int(ep["env_id"])
                outcome = ep["outcome"]
                extras.setdefault("episode_done", []).append({**ep, "env_id": global_env_id})
                if self.episode_logger is not None:
                    self.episode_logger.record(
                        global_env_id,
                        outcome,
                        ep.get("episode_length"),
                        ep.get("impact_lateral"),
                        ep.get("best_miss"),
                        curriculum_scale=self.curriculum_scale,
                    )
                self._update_curriculum(outcome == "hit")
            env_offset += local_n

        # Curriculum may have advanced; push new scale so the next resets use it.
        self._broadcast_curriculum_scale()
        return obs, rewards, dones, extras

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for remote in self._remotes:
            try:
                remote.close()
            except OSError:
                pass
        for proc in self._processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()

    def _broadcast_curriculum_scale(self, force: bool = False) -> None:
        if not force and self._last_broadcast_scale == self.curriculum_scale:
            return
        for remote in self._remotes:
            remote.send(("set_curriculum_scale", self.curriculum_scale))
        for remote in self._remotes:
            remote.recv()
        self._last_broadcast_scale = self.curriculum_scale

    def _stack_obs(self, chunks: list[dict]) -> TensorDict:
        policy = np.concatenate([c["obs_policy"] for c in chunks], axis=0)
        privileged = np.concatenate([c["obs_privileged"] for c in chunks], axis=0)
        lengths = np.concatenate([c["episode_length_buf"] for c in chunks], axis=0)
        self.episode_length_buf = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        return TensorDict(
            {
                "policy": torch.as_tensor(policy, dtype=torch.float32, device=self.device),
                "privileged": torch.as_tensor(privileged, dtype=torch.float32, device=self.device),
            },
            batch_size=[self.num_envs],
        )

    def _update_curriculum(self, is_hit: bool) -> None:
        if not self.curriculum_enabled or self.curriculum_scale >= self.curriculum_max_scale:
            return
        self._curriculum_outcomes.append(1.0 if is_hit else 0.0)
        if len(self._curriculum_outcomes) < self._curriculum_outcomes.maxlen:
            return
        hit_rate = sum(self._curriculum_outcomes) / len(self._curriculum_outcomes)
        if hit_rate >= self.curriculum_hit_threshold:
            self.curriculum_scale = min(self.curriculum_max_scale, self.curriculum_scale + self.curriculum_step)
            self._curriculum_outcomes.clear()
