#!/usr/bin/env python3
"""Evaluate a trained teacher checkpoint."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flywheel_rewards import compute_hit_score, compute_miss_score
from training.configs.rl_configs import teacher_ppo_cfg
from training.envs.flywheel_env import FlywheelVecEnv
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained flywheel teacher checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the mean")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for episodes.csv and summary.txt (default: logs/eval/<timestamp>)",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    return f"mean={arr.mean():.3f}, median={np.median(arr):.3f}, min={arr.min():.3f}, max={arr.max():.3f}"


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": len(values),
    }


def enrich_episode(record: dict, episode_id: int) -> dict:
    impact_lateral = record.get("impact_lateral")
    best_lateral = record.get("best_lateral")
    return {
        "episode_id": episode_id,
        "env_id": record["env_id"],
        "outcome": record["outcome"],
        "episode_length": record.get("episode_length"),
        "return": record["return"],
        "impact_lateral_m": impact_lateral,
        "best_lateral_m": best_lateral,
        "hit_score": compute_hit_score(impact_lateral) if impact_lateral is not None else None,
        "miss_score": compute_miss_score(best_lateral) if best_lateral is not None else None,
    }


def build_report(episodes: list[dict], checkpoint: str, num_envs: int, stochastic: bool) -> tuple[str, dict]:
    outcomes = Counter(ep["outcome"] for ep in episodes)
    total = len(episodes)
    returns = [ep["return"] for ep in episodes]
    lengths = [float(ep["episode_length"]) for ep in episodes if ep["episode_length"] is not None]

    hit_impacts = [
        ep["impact_lateral_m"]
        for ep in episodes
        if ep["outcome"] == "hit" and ep["impact_lateral_m"] is not None
    ]
    hit_scores = [ep["hit_score"] for ep in episodes if ep["hit_score"] is not None]

    miss_best = [
        ep["best_lateral_m"]
        for ep in episodes
        if ep["outcome"] in {"floor", "timeout"} and ep["best_lateral_m"] is not None
    ]
    miss_scores = [ep["miss_score"] for ep in episodes if ep["miss_score"] is not None]
    all_best = [ep["best_lateral_m"] for ep in episodes if ep["best_lateral_m"] is not None]

    report_data = {
        "checkpoint": checkpoint,
        "num_episodes": total,
        "num_envs": num_envs,
        "stochastic": stochastic,
        "outcomes": {name: outcomes[name] for name in ("hit", "floor", "timeout")},
        "outcome_pct": {name: 100.0 * outcomes[name] / total for name in ("hit", "floor", "timeout")},
        "returns": stats(returns),
        "lengths": stats(lengths),
        "hit_impacts": stats(hit_impacts),
        "hit_scores": stats(hit_scores),
        "miss_best": stats(miss_best),
        "miss_scores": stats(miss_scores),
        "all_best": stats(all_best),
    }

    lines = [
        "Flywheel Teacher Evaluation",
        "===========================",
        f"Checkpoint: {checkpoint}",
        f"Episodes:   {total}",
        f"Envs:       {num_envs}",
        f"Stochastic: {stochastic}",
        "",
        "Episode Outcomes",
        "----------------",
    ]
    for outcome in ("hit", "floor", "timeout"):
        count = outcomes[outcome]
        pct = 100.0 * count / total
        lines.append(f"{outcome:>7}: {count:4d} ({pct:5.1f}%)")

    lines.extend(
        [
            "",
            "Returns and Length",
            "------------------",
            f"Mean episode return: {summarize(returns)}",
            f"Mean episode length: {summarize(lengths)}",
            "",
            "Accuracy",
            "--------",
        ]
    )

    if hit_impacts:
        lines.append(f"Hit impact distance (m): {summarize(hit_impacts)}")
        lines.append(f"Hit score:               {summarize(hit_scores)}")
    else:
        lines.append("Hit impact distance (m): no hits recorded")

    if miss_best:
        lines.append(f"Miss best lateral (m):     {summarize(miss_best)}")
        lines.append(f"Miss score:                {summarize(miss_scores)}")
    else:
        lines.append("Miss best lateral (m):     no floor/timeout misses with lateral data")

    if all_best:
        lines.append(f"All episodes best lateral: {summarize(all_best)}")

    return "\n".join(lines), report_data


def run_evaluation(
    checkpoint: str,
    num_envs: int,
    num_episodes: int,
    device: str,
    seed: int,
    stochastic: bool,
) -> list[dict]:
    env = FlywheelVecEnv(num_envs=num_envs, device=device, seed=seed)
    runner = OnPolicyRunner(
        env,
        teacher_ppo_cfg(),
        log_dir=os.path.join(PROJECT_ROOT, "logs", "eval"),
        device=device,
    )
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    completed_episodes: list[dict] = []

    with torch.inference_mode():
        while len(completed_episodes) < num_episodes:
            actions = policy(obs, stochastic_output=stochastic)
            obs, rewards, dones, extras = env.step(actions)

            episode_returns += rewards.detach().cpu().numpy()

            for record in extras.get("episode_done", []):
                env_id = record["env_id"]
                completed_episodes.append(
                    enrich_episode(
                        {**record, "return": float(episode_returns[env_id])},
                        episode_id=len(completed_episodes) + 1,
                    )
                )
                episode_returns[env_id] = 0.0

            if dones.any():
                done_ids = torch.nonzero(dones, as_tuple=False).flatten().tolist()
                for env_id in done_ids:
                    episode_returns[env_id] = 0.0

    return completed_episodes


def write_episodes_csv(path: str, episodes: list[dict]) -> None:
    fieldnames = [
        "episode_id",
        "env_id",
        "outcome",
        "episode_length",
        "return",
        "impact_lateral_m",
        "best_lateral_m",
        "hit_score",
        "miss_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episodes)


def write_summary_txt(path: str, summary_text: str) -> None:
    with open(path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)
        summary_file.write("\n")


def default_output_dir(checkpoint: str) -> str:
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PROJECT_ROOT, "logs", "eval", f"{checkpoint_name}_{timestamp}")


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output_dir = args.output_dir or default_output_dir(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Target episodes: {args.num_episodes} | parallel envs: {args.num_envs}")

    episodes = run_evaluation(
        checkpoint=args.checkpoint,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=device,
        seed=args.seed,
        stochastic=args.stochastic,
    )

    summary_text, _ = build_report(
        episodes,
        checkpoint=args.checkpoint,
        num_envs=args.num_envs,
        stochastic=args.stochastic,
    )
    print()
    print(summary_text)

    episodes_path = os.path.join(output_dir, "episodes.csv")
    summary_path = os.path.join(output_dir, "summary.txt")
    write_episodes_csv(episodes_path, episodes)
    write_summary_txt(summary_path, summary_text)

    print()
    print(f"Evaluated {len(episodes)} episodes.")
    print(f"Wrote episode table: {episodes_path}")
    print(f"Wrote summary text:  {summary_path}")


if __name__ == "__main__":
    main()
