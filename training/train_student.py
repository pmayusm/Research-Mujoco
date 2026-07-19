#!/usr/bin/env python3
"""Distill a student policy from a trained teacher checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.configs.rl_configs import student_distillation_cfg
from training.envs.flywheel_env import FlywheelVecEnv
from training.episode_logger import EpisodeTableLogger
from rsl_rl.runners import DistillationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flywheel student policy via distillation")
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs", "student"))
    parser.add_argument(
        "--episode-log-every",
        type=int,
        default=100,
        help="Write a full episode table + summary every N completed episodes (0 to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    if not os.path.isfile(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    episode_logger = None
    if args.episode_log_every > 0:
        episode_logger = EpisodeTableLogger(
            output_dir=os.path.join(log_dir, "episode_tables"),
            log_every=args.episode_log_every,
        )

    env = FlywheelVecEnv(num_envs=args.num_envs, device=device, seed=args.seed, episode_logger=episode_logger)
    train_cfg = student_distillation_cfg()

    runner = DistillationRunner(env, train_cfg, log_dir=log_dir, device=device)
    runner.load(args.teacher_checkpoint, load_cfg={"teacher": True, "iteration": False}, strict=False)
    runner.add_git_repo_to_log(PROJECT_ROOT)

    print(f"Loaded teacher from: {args.teacher_checkpoint}")
    print(f"Training student policy for {args.max_iterations} iterations.")
    print(f"Logs and checkpoints: {log_dir}")
    if episode_logger is not None:
        print(f"Episode tables every {args.episode_log_every} episodes: {episode_logger.output_dir}")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    if episode_logger is not None:
        episode_logger.flush()
    print(f"Done. Final checkpoint: {log_dir}/model_{runner.current_learning_iteration}.pt")


if __name__ == "__main__":
    main()
