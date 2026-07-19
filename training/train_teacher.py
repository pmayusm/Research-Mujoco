#!/usr/bin/env python3
"""Train the privileged teacher policy with RSL-RL PPO."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.configs.rl_configs import teacher_ppo_cfg
from training.envs.flywheel_env import FlywheelVecEnv
from training.episode_logger import EpisodeTableLogger
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flywheel teacher policy with RSL-RL PPO")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs", "teacher"))
    parser.add_argument(
        "--episode-log-every",
        type=int,
        default=100,
        help="Write a full episode table + summary every N completed episodes (0 to disable)",
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start targets close/easy and widen the randomization envelope as hit rate improves",
    )
    parser.add_argument("--curriculum-start-scale", type=float, default=0.3)
    parser.add_argument("--curriculum-step", type=float, default=0.05)
    parser.add_argument("--curriculum-window", type=int, default=200)
    parser.add_argument("--curriculum-hit-threshold", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    episode_logger = None
    if args.episode_log_every > 0:
        episode_logger = EpisodeTableLogger(
            output_dir=os.path.join(log_dir, "episode_tables"),
            log_every=args.episode_log_every,
        )

    env = FlywheelVecEnv(
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        episode_logger=episode_logger,
        curriculum_enabled=args.curriculum,
        curriculum_start_scale=args.curriculum_start_scale,
        curriculum_step=args.curriculum_step,
        curriculum_window=args.curriculum_window,
        curriculum_hit_threshold=args.curriculum_hit_threshold,
    )
    train_cfg = teacher_ppo_cfg()

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    runner.add_git_repo_to_log(PROJECT_ROOT)
    print(f"Training teacher policy for {args.max_iterations} iterations.")
    print(f"Logs and checkpoints: {log_dir}")
    if episode_logger is not None:
        print(f"Episode tables every {args.episode_log_every} episodes: {episode_logger.output_dir}")
    if args.curriculum:
        print(
            f"Curriculum enabled: start_scale={args.curriculum_start_scale}, "
            f"step={args.curriculum_step}, window={args.curriculum_window}, "
            f"hit_threshold={args.curriculum_hit_threshold}"
        )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    if episode_logger is not None:
        episode_logger.flush()
    print(f"Done. Final checkpoint: {log_dir}/model_{runner.current_learning_iteration}.pt")


if __name__ == "__main__":
    main()
