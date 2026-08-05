#!/usr/bin/env python3
"""Log the ball's per-step distance to the target and export it as a table.

Unlike eval_teacher.py (which reports one row per *episode*), this script
records one row per *simulation step* so you can see how the ball-to-target
distance actually evolves during flight -- useful for sanity-checking reward
shaping and control behavior before committing to a long training run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flywheel_rewards import compute_target_distances
from training.configs.rl_configs import teacher_ppo_cfg
from training.envs.flywheel_env import FlywheelVecEnv
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log per-step ball-to-target distance")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional trained teacher checkpoint. If omitted, uses zero actions (no control).",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-episode-length", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=20000, help="Safety cap on total steps logged")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the mean")
    parser.add_argument(
        "--curriculum-scale",
        type=float,
        default=None,
        help=(
            "Pin the target-randomization envelope to this curriculum scale (0-1) instead "
            "of the default full/hardest range -- use the scale the checkpoint was actually "
            "trained at to see representative trajectories rather than harder-than-trained ones."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for trajectory.csv and trajectory_summary.txt (default: logs/trajectories/<timestamp>)",
    )
    return parser.parse_args()


def default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PROJECT_ROOT, "logs", "trajectories", timestamp)


def build_policy(checkpoint: str | None, env: FlywheelVecEnv, device: str):
    if checkpoint is None:
        return None
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    runner = OnPolicyRunner(
        env,
        teacher_ppo_cfg(),
        log_dir=os.path.join(PROJECT_ROOT, "logs", "eval"),
        device=device,
    )
    runner.load(checkpoint)
    return runner.get_inference_policy(device=device)


def collect_trajectory(
    env: FlywheelVecEnv,
    policy,
    num_episodes: int,
    max_steps: int,
    stochastic: bool,
) -> list[dict]:
    obs = env.get_observations()
    episode_id = np.ones(env.num_envs, dtype=np.int64)
    next_episode_id = env.num_envs + 1
    completed = 0
    rows: list[dict] = []

    with torch.inference_mode():
        for global_step in range(max_steps):
            if policy is not None:
                actions = policy(obs, stochastic_output=stochastic)
            else:
                actions = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)

            for env_id, handles in enumerate(env.handles):
                ball_pos = handles.data.xpos[handles.ball_id].copy()
                target_pos = handles.data.xpos[handles.body_id].copy()
                plane_normal = handles.data.xmat[handles.body_id].reshape(3, 3)[:, 2]
                miss_distance, lateral_distance, face_distance = compute_target_distances(
                    ball_pos, target_pos, plane_normal, handles.target_half_thickness
                )
                rows.append(
                    {
                        "episode_id": int(episode_id[env_id]),
                        "env_id": env_id,
                        "step": int(env.episode_length_buf[env_id].item()),
                        "ball_spawned": bool(env.ball_spawned[env_id]),
                        "ball_x": float(ball_pos[0]),
                        "ball_y": float(ball_pos[1]),
                        "ball_z": float(ball_pos[2]),
                        "target_x": float(target_pos[0]),
                        "target_y": float(target_pos[1]),
                        "target_z": float(target_pos[2]),
                        "miss_distance_m": float(miss_distance),
                        "lateral_distance_m": float(lateral_distance),
                        "face_distance_m": float(face_distance),
                    }
                )

            obs, rewards, dones, extras = env.step(actions)

            for record in extras.get("episode_done", []):
                env_id = record["env_id"]
                rows.append(
                    {
                        "episode_id": int(episode_id[env_id]),
                        "env_id": env_id,
                        "step": record["episode_length"],
                        "ball_spawned": True,
                        "ball_x": None,
                        "ball_y": None,
                        "ball_z": None,
                        "target_x": None,
                        "target_y": None,
                        "target_z": None,
                        "miss_distance_m": None,
                        "lateral_distance_m": None,
                        "face_distance_m": None,
                        "outcome": record["outcome"],
                    }
                )
                completed += 1
                episode_id[env_id] = next_episode_id
                next_episode_id += 1

            if completed >= num_episodes:
                break

    return rows


def write_trajectory_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "episode_id",
        "env_id",
        "step",
        "ball_spawned",
        "ball_x",
        "ball_y",
        "ball_z",
        "target_x",
        "target_y",
        "target_z",
        "miss_distance_m",
        "lateral_distance_m",
        "face_distance_m",
        "outcome",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_txt(path: str, rows: list[dict], checkpoint: str | None) -> str:
    flight_rows = [row for row in rows if row.get("ball_spawned") and row.get("miss_distance_m") is not None]
    episode_ids = sorted(set(row["episode_id"] for row in rows))

    lines = [
        "Flywheel Ball-to-Target Distance Trajectory",
        "============================================",
        f"Checkpoint: {checkpoint or '(none -- zero/no-op actions)'}",
        f"Episodes logged: {len(episode_ids)}",
        f"Total steps logged: {len(rows)}",
        "",
        "Per-Episode Distance Summary",
        "-----------------------------",
    ]

    for episode_id in episode_ids:
        episode_rows = [row for row in flight_rows if row["episode_id"] == episode_id]
        outcome_row = next(
            (row for row in rows if row["episode_id"] == episode_id and row.get("outcome")), None
        )
        outcome = outcome_row["outcome"] if outcome_row else "incomplete"
        if not episode_rows:
            lines.append(f"Episode {episode_id}: outcome={outcome} (no in-flight samples)")
            continue
        distances = [row["miss_distance_m"] for row in episode_rows]
        lines.append(
            f"Episode {episode_id}: outcome={outcome:8s} "
            f"start_dist={distances[0]:.3f}m  min_dist={min(distances):.3f}m  "
            f"end_dist={distances[-1]:.3f}m  samples={len(distances)}"
        )

    return_text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as summary_file:
        summary_file.write(return_text)
        summary_file.write("\n")
    return return_text


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    env_kwargs = {}
    if args.curriculum_scale is not None:
        env_kwargs = {
            "curriculum_enabled": True,
            "curriculum_start_scale": args.curriculum_scale,
            "curriculum_max_scale": args.curriculum_scale,
        }
    env = FlywheelVecEnv(
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        max_episode_length=args.max_episode_length,
        **env_kwargs,
    )
    policy = build_policy(args.checkpoint, env, device)

    print(f"Logging trajectory: {'policy=' + args.checkpoint if policy else 'no-op actions (zero control)'}")
    print(f"Target episodes: {args.num_episodes} | parallel envs: {args.num_envs}")

    rows = collect_trajectory(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        stochastic=args.stochastic,
    )

    output_dir = args.output_dir or default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    trajectory_path = os.path.join(output_dir, "trajectory.csv")
    summary_path = os.path.join(output_dir, "trajectory_summary.txt")

    write_trajectory_csv(trajectory_path, rows)
    summary_text = write_summary_txt(summary_path, rows, args.checkpoint)

    print()
    print(summary_text)
    print()
    print(f"Wrote per-step trajectory table: {trajectory_path}")
    print(f"Wrote summary text:              {summary_path}")


if __name__ == "__main__":
    main()
