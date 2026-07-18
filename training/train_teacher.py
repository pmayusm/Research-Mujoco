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
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flywheel teacher policy with RSL-RL PPO")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs", "teacher"))
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

    env = FlywheelVecEnv(num_envs=args.num_envs, device=device, seed=args.seed)
    train_cfg = teacher_ppo_cfg()

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    runner.add_git_repo_to_log(PROJECT_ROOT)
    print(f"Training teacher policy for {args.max_iterations} iterations.")
    print(f"Logs and checkpoints: {log_dir}")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    print(f"Done. Final checkpoint: {log_dir}/model_{runner.current_learning_iteration}.pt")


if __name__ == "__main__":
    main()
