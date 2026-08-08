#!/usr/bin/env python3
"""Train the privileged teacher policy with RSL-RL PPO."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.configs.rl_configs import teacher_ppo_cfg
from training.envs.flywheel_env import FlywheelVecEnv
from training.envs.parallel_flywheel_env import ParallelFlywheelVecEnv
from training.episode_logger import EpisodeTableLogger
from rsl_rl.runners import OnPolicyRunner


class BestTrackingRunner(OnPolicyRunner):
    """OnPolicyRunner that also keeps model_best.pt for the best low-floor window."""

    def __init__(self, *args, episode_logger: EpisodeTableLogger | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_logger = episode_logger
        self._best_score: float | None = None
        self._best_meta: dict | None = None

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos=infos)
        if self.episode_logger is None or self.logger.log_dir is None:
            return
        score = self.episode_logger.best_checkpoint_score(window=500, max_floor=0.15)
        if score is None:
            return
        if self._best_score is not None and score <= self._best_score:
            return

        rates = self.episode_logger.recent_window_rates(500)
        best_path = os.path.join(self.logger.log_dir, "model_best.pt")
        shutil.copy2(path, best_path)
        self._best_score = score
        self._best_meta = {
            "score": score,
            "source_checkpoint": os.path.basename(path),
            "iteration": self.current_learning_iteration,
            "episodes": len(self.episode_logger.records),
            "window": 500,
            "rates": rates,
        }
        meta_path = os.path.join(self.logger.log_dir, "model_best.json")
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(self._best_meta, meta_file, indent=2)
        print(
            f"[best] Updated model_best.pt from {os.path.basename(path)} "
            f"(score={score:.3f}, hit={rates['hit']:.1%}, floor={rates['floor']:.1%})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flywheel teacher policy with RSL-RL PPO")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Total parallel MuJoCo environments (split across --num-workers processes)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Process workers for physics (default: min(num_envs, cpu_count-2)). "
        "Use 1 with --no-parallel for the legacy in-process VecEnv.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Step env chunks in subprocesses (real multi-core speedup for MuJoCo)",
    )
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=900,
        help="Control steps per episode before timeout (900 * frame_skip=5 * dt=0.002 ≈ 9s)",
    )
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
        "--plot-every",
        type=int,
        default=100,
        help="Redraw the hit-accuracy-vs-episode chart every N completed episodes",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a model_*.pt checkpoint to resume training from",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="On resume, load actor/critic weights but start a fresh Adam optimizer "
        "(avoids carrying over collapsed adaptive moments)",
    )
    parser.add_argument(
        "--bump-std",
        type=float,
        default=None,
        help="On resume, raise the policy action std to at least this value so "
        "exploration can escape a floor-drop local optimum. Prefer ~0.30 with the "
        "current std_range floor of 0.25.",
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start targets close/easy and widen the randomization envelope as hit rate improves",
    )
    parser.add_argument("--curriculum-start-scale", type=float, default=0.3)
    parser.add_argument(
        "--curriculum-max-scale",
        type=float,
        default=1.0,
        help="Cap for curriculum scale. Set equal to --curriculum-start-scale to freeze difficulty.",
    )
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
            plot_every=args.plot_every,
        )

    env_kwargs = dict(
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        max_episode_length=args.max_episode_length,
        episode_logger=episode_logger,
        curriculum_enabled=args.curriculum,
        curriculum_start_scale=args.curriculum_start_scale,
        curriculum_max_scale=args.curriculum_max_scale,
        curriculum_step=args.curriculum_step,
        curriculum_window=args.curriculum_window,
        curriculum_hit_threshold=args.curriculum_hit_threshold,
    )
    if args.parallel and (args.num_workers is None or args.num_workers != 1):
        env = ParallelFlywheelVecEnv(num_workers=args.num_workers, **env_kwargs)
        print(f"Parallel MuJoCo: {args.num_envs} envs across {env.num_workers} workers {env._local_counts}")
    else:
        env = FlywheelVecEnv(**env_kwargs)
        print(f"In-process MuJoCo: {args.num_envs} serial envs")
    train_cfg = teacher_ppo_cfg()

    runner = BestTrackingRunner(env, train_cfg, log_dir=log_dir, device=device, episode_logger=episode_logger)
    runner.add_git_repo_to_log(PROJECT_ROOT)

    if args.resume:
        load_cfg = None
        if args.reset_optimizer:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            }
        runner.load(args.resume, load_cfg=load_cfg, map_location=device)
        # rsl_rl's PPO.load() restores the optimizer's state dict verbatim, which
        # includes the learning rate the checkpoint was saved with -- silently
        # overriding whatever "learning_rate" is set to in rl_configs.py. Since we
        # use schedule="fixed", nothing else ever touches it again afterwards, so
        # without this override every "resume" run would keep training at the OLD
        # checkpoint's LR forever, no matter what the config says.
        configured_lr = train_cfg["algorithm"]["learning_rate"]
        old_lr = runner.alg.optimizer.param_groups[0]["lr"]
        for param_group in runner.alg.optimizer.param_groups:
            param_group["lr"] = configured_lr
        runner.alg.learning_rate = configured_lr
        print(
            f"Resumed from {args.resume} at iteration {runner.current_learning_iteration}; "
            f"training for {args.max_iterations} more iterations."
        )
        if args.reset_optimizer:
            print("Loaded policy/value weights only -- optimizer state reset.")
        if old_lr != configured_lr:
            print(f"Overrode resumed optimizer learning rate: {old_lr:.2e} -> {configured_lr:.2e}")
        if args.bump_std is not None:
            dist = runner.alg.get_policy().distribution
            old_std = dist.std_param.detach().clone()
            with torch.no_grad():
                dist.std_param.clamp_(min=args.bump_std)
            print(f"Bumped action std_param: {old_std.tolist()} -> {dist.std_param.detach().tolist()}")
    else:
        print(f"Training teacher policy for {args.max_iterations} iterations.")

    print(f"Logs and checkpoints: {log_dir}")
    if episode_logger is not None:
        print(f"Episode tables every {args.episode_log_every} episodes: {episode_logger.output_dir}")
        print(f"Hit-accuracy plot every {args.plot_every} episodes: {episode_logger.output_dir}/hit_accuracy.png")
        print("Best low-floor checkpoint will be copied to model_best.pt when improved.")
    if args.curriculum:
        print(
            f"Curriculum enabled: start_scale={args.curriculum_start_scale}, "
            f"max_scale={args.curriculum_max_scale}, "
            f"step={args.curriculum_step}, window={args.curriculum_window}, "
            f"hit_threshold={args.curriculum_hit_threshold}"
        )
        if args.curriculum_start_scale >= args.curriculum_max_scale:
            print(
                f"Curriculum FROZEN at scale={args.curriculum_start_scale} "
                f"(start_scale >= max_scale)."
            )
    # init_at_random_ep_len=False when resuming: randomizing episode lengths is
    # meant to diversify the very first rollout of a fresh run, but here the
    # policy/value function are already trained, so start every env cleanly.
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=not args.resume)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    if episode_logger is not None:
        episode_logger.flush()
    print(f"Done. Final checkpoint: {log_dir}/model_{runner.current_learning_iteration}.pt")
    if runner._best_meta is not None:
        print(f"Best low-floor checkpoint: {log_dir}/model_best.pt ({runner._best_meta})")


if __name__ == "__main__":
    # Required on macOS for spawn-based MuJoCo workers.
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
