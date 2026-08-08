#!/usr/bin/env python3
"""Smoke benchmark for serial vs parallel FlywheelVecEnv."""

from __future__ import annotations

import time

import torch

from training.envs.flywheel_env import FlywheelVecEnv
from training.envs.parallel_flywheel_env import ParallelFlywheelVecEnv


def bench(env, steps: int = 50, label: str = "") -> float:
    env.get_observations()
    actions = torch.zeros((env.num_envs, env.num_actions))
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    dt = time.perf_counter() - t0
    sps = env.num_envs * steps / dt
    print(f"{label}: {env.num_envs} envs x {steps} steps in {dt:.3f}s -> {sps:.0f} env-steps/s")
    return sps


def main() -> None:
    serial = FlywheelVecEnv(
        num_envs=16,
        seed=0,
        curriculum_enabled=True,
        curriculum_start_scale=0.45,
        curriculum_max_scale=0.45,
    )
    s1 = bench(serial, label="serial-16")

    for n_envs, n_workers in ((32, 8), (64, 8), (64, 10)):
        par = ParallelFlywheelVecEnv(
            num_envs=n_envs,
            num_workers=n_workers,
            seed=0,
            curriculum_enabled=True,
            curriculum_start_scale=0.45,
            curriculum_max_scale=0.45,
        )
        print("workers", par.num_workers, par._local_counts)
        s2 = bench(par, label=f"parallel-{n_envs}x{n_workers}")
        par.close()
        print(f"  ratio vs serial-16: {s2 / s1:.2f}x")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
