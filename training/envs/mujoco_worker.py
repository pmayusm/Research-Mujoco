"""Multiprocess MuJoCo worker for parallel FlywheelVecEnv stepping.

Kept in its own module so macOS spawn can import the worker target cleanly.
Each worker owns a serial FlywheelVecEnv chunk and speaks a simple pipe protocol.
"""

from __future__ import annotations

import os
import sys

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def flywheel_worker(
    remote,
    parent_remote,
    *,
    num_local_envs: int,
    seed: int,
    max_episode_length: int,
    device: str,
    curriculum_enabled: bool,
    curriculum_start_scale: float,
    curriculum_max_scale: float,
    curriculum_step: float,
    curriculum_window: int,
    curriculum_hit_threshold: float,
) -> None:
    """Process entrypoint: step/reset a local env chunk on command."""
    parent_remote.close()

    # Local import so parent process import cost stays light and spawn works.
    from training.envs.flywheel_env import FlywheelVecEnv

    env = FlywheelVecEnv(
        num_envs=num_local_envs,
        device=device,
        seed=seed,
        max_episode_length=max_episode_length,
        episode_logger=None,
        # Workers keep scale pinned; parent owns curriculum advancement and
        # broadcasts the current scale after each batch of outcomes.
        curriculum_enabled=False,
        curriculum_start_scale=curriculum_start_scale,
        curriculum_max_scale=curriculum_max_scale,
        curriculum_step=curriculum_step,
        curriculum_window=curriculum_window,
        curriculum_hit_threshold=curriculum_hit_threshold,
    )
    # Match parent's initial scale even with curriculum_enabled=False.
    env.curriculum_scale = curriculum_start_scale if curriculum_enabled else curriculum_max_scale

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                actions = torch.as_tensor(data, dtype=torch.float32, device=device)
                obs, rewards, dones, extras = env.step(actions)
                remote.send(
                    {
                        "obs_policy": obs["policy"].detach().cpu().numpy(),
                        "obs_privileged": obs["privileged"].detach().cpu().numpy(),
                        "rewards": rewards.detach().cpu().numpy(),
                        "dones": dones.detach().cpu().numpy(),
                        "episode_length_buf": env.episode_length_buf.detach().cpu().numpy(),
                        "episode_done": extras.get("episode_done", []),
                        "log": extras.get("log", {}),
                    }
                )
            elif cmd == "get_observations":
                obs = env.get_observations()
                remote.send(
                    {
                        "obs_policy": obs["policy"].detach().cpu().numpy(),
                        "obs_privileged": obs["privileged"].detach().cpu().numpy(),
                        "episode_length_buf": env.episode_length_buf.detach().cpu().numpy(),
                    }
                )
            elif cmd == "set_curriculum_scale":
                env.curriculum_scale = float(data)
                remote.send(True)
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown worker command: {cmd}")
    except KeyboardInterrupt:
        remote.close()
