#!/usr/bin/env python3
"""Classify miss modes for a checkpoint: yaw error vs power/height.

Deterministic rollouts; reports whether misses are left/right (aim) vs
short/low (launch energy), relative to the randomized target.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

import mujoco
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flywheel_rewards import impact_distance_on_face
from training.configs.rl_configs import teacher_ppo_cfg
from training.envs.flywheel_env import FlywheelVecEnv
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose flywheel miss modes")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-episodes", type=int, default=400)
    p.add_argument("--max-episode-length", type=int, default=900)
    p.add_argument("--curriculum-scale", type=float, default=0.45)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def wrap_angle(a: float) -> float:
    return float(((a + math.pi) % (2 * math.pi)) - math.pi)


class DiagnosingEnv(FlywheelVecEnv):
    """Records launch/aim diagnostics when episodes terminate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diag_records: list[dict] = []
        self._ep: list[dict] = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self._begin_ep(i)

    def _begin_ep(self, env_id: int) -> None:
        h = self.handles[env_id]
        target = h.data.xpos[h.body_id].copy()
        hood = h.data.xpos[h.hood_id].copy()
        hood_rot = h.data.xmat[h.hood_id].reshape(3, 3)
        target_local = hood_rot.T @ (target - hood)
        spawn = self._hood_local_to_world(h, h.spawn_local)
        # Barrel aims roughly +Y in hood frame (spawn y≈1.2). Remaining yaw to
        # put target on the midplane is atan2(local_x, local_y).
        aim_err = math.atan2(float(target_local[0]), float(target_local[1]))
        range_xy = float(np.linalg.norm(target[:2] - spawn[:2]))
        self._ep[env_id] = {
            "target_z": float(target[2]),
            "range_xy": range_xy,
            "aim_err_at_reset": aim_err,
            "aim_err_at_spawn": None,
            "yaw_at_spawn": None,
            "pre_ctrls": [],
            "max_z": -1e9,
            "max_speed": 0.0,
            "launch_speed": None,
            "launch_vz": None,
            "post_spawn_steps": 0,
        }

    def _spawn_ball(self, handles) -> None:
        env_id = self.handles.index(handles)
        yaw_j = mujoco.mj_name2id(handles.model, mujoco.mjtObj.mjOBJ_JOINT, "yaw joint")
        raw = float(handles.data.qpos[handles.model.jnt_qposadr[yaw_j]])
        self._ep[env_id]["yaw_at_spawn"] = wrap_angle(raw)
        hood = handles.data.xpos[handles.hood_id]
        hood_rot = handles.data.xmat[handles.hood_id].reshape(3, 3)
        target = handles.data.xpos[handles.body_id]
        target_local = hood_rot.T @ (target - hood)
        self._ep[env_id]["aim_err_at_spawn"] = math.atan2(
            float(target_local[0]), float(target_local[1])
        )
        super()._spawn_ball(handles)

    def _record_episode_done(
        self,
        extras: dict,
        env_id: int,
        outcome: str,
        *,
        impact_lateral: float | None = None,
        best_miss: float | None = None,
        episode_length: int | None = None,
    ) -> None:
        # Called before reset — ball/target still at terminal poses.
        super()._record_episode_done(
            extras,
            env_id,
            outcome,
            impact_lateral=impact_lateral,
            best_miss=best_miss,
            episode_length=episode_length,
        )
        h = self.handles[env_id]
        ep = self._ep[env_id]
        ball = h.data.xpos[h.ball_id].copy()
        target = h.data.xpos[h.body_id].copy()
        plane_normal = h.data.xmat[h.body_id].reshape(3, 3)[:, 2].copy()
        impact = (
            float(impact_lateral)
            if impact_lateral is not None
            else impact_distance_on_face(ball, target, plane_normal, h.target_half_thickness)
        )
        signed_z = float(ball[2] - target[2])
        right = np.cross(plane_normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        right = right / (np.linalg.norm(right) + 1e-8)
        signed_lat = float((ball - target) @ right)

        # Update max_z one last time at terminal.
        ep["max_z"] = max(ep["max_z"], float(ball[2]))
        z_margin = ep["max_z"] - ep["target_z"]

        yaw_err = ep.get("aim_err_at_spawn")
        if yaw_err is None:
            yaw_err = ep.get("aim_err_at_reset")

        pre = ep["pre_ctrls"]
        fly_mean = float(np.mean([a[0] for a in pre])) if pre else None
        yaw_mean = float(np.mean([a[1] for a in pre])) if pre else None

        if outcome == "miss":
            if abs(signed_lat) > max(abs(signed_z), 1.0):
                aim_mode = "yaw"
            elif z_margin < -0.5:
                aim_mode = "low_power"
            elif z_margin > 2.0:
                aim_mode = "high_or_lateral"
            else:
                aim_mode = "lateral_other"
        elif outcome == "hit":
            aim_mode = "hit"
        elif outcome == "floor":
            aim_mode = "low_power" if z_margin < 0 else "floor_other"
        else:
            aim_mode = "timeout"

        self.diag_records.append(
            {
                "outcome": outcome,
                "aim_mode": aim_mode,
                "impact_m": impact,
                "best_miss_m": best_miss,
                "signed_lat_m": signed_lat,
                "z_err_m": signed_z,
                "z_margin_m": z_margin,
                "max_z": ep["max_z"],
                "target_z": ep["target_z"],
                "range_xy": ep["range_xy"],
                "yaw_at_spawn": ep["yaw_at_spawn"],
                "yaw_err_rad": yaw_err,
                "yaw_err_deg": None if yaw_err is None else yaw_err * 180 / math.pi,
                "fly_ctrl_mean_pre": fly_mean,
                "yaw_ctrl_mean_pre": yaw_mean,
                "launch_speed": ep["launch_speed"],
                "launch_vz": ep["launch_vz"],
                "max_speed": ep["max_speed"],
                "episode_length": episode_length,
            }
        )

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        for env_id, h in enumerate(self.handles):
            ep = self._ep[env_id]
            if not self.ball_spawned[env_id]:
                ep["pre_ctrls"].append(actions_np[env_id].copy())
            else:
                ball_vel = h.data.qvel[h.ball_dof_adr : h.ball_dof_adr + 3]
                ep["max_z"] = max(ep["max_z"], float(h.data.xpos[h.ball_id][2]))
                ep["max_speed"] = max(ep["max_speed"], float(np.linalg.norm(ball_vel)))
                ep["post_spawn_steps"] += 1
                if ep["launch_speed"] is None and ep["post_spawn_steps"] == 8:
                    ep["launch_speed"] = float(np.linalg.norm(ball_vel))
                    ep["launch_vz"] = float(ball_vel[2])

        obs, rewards, dones, extras = super().step(actions)

        for env_id in range(self.num_envs):
            if bool(dones[env_id].item()):
                # Parent already recorded + reset; start fresh episode snapshot.
                self._begin_ep(env_id)

        return obs, rewards, dones, extras


def arr_stats(vals) -> str:
    a = np.asarray(vals, dtype=np.float64)
    return (
        f"mean={a.mean():+.2f} median={np.median(a):+.2f} "
        f"p10={np.percentile(a, 10):+.2f} p90={np.percentile(a, 90):+.2f}"
    )


def summarize(recs: list[dict]) -> str:
    n = len(recs)
    outcomes = Counter(r["outcome"] for r in recs)
    modes = Counter(r["aim_mode"] for r in recs)
    lines = [f"Episodes: {n}", "Outcomes:"]
    for k in ("hit", "floor", "miss", "timeout"):
        lines.append(f"  {k:8s} {outcomes[k]:4d} ({100 * outcomes[k] / n:5.1f}%)")
    lines.append("Aim/power mode (all eps):")
    for k, v in modes.most_common():
        lines.append(f"  {k:16s} {v:4d} ({100 * v / n:5.1f}%)")

    misses = [r for r in recs if r["outcome"] == "miss"]
    hits = [r for r in recs if r["outcome"] == "hit"]
    floors = [r for r in recs if r["outcome"] == "floor"]
    timeouts = [r for r in recs if r["outcome"] == "timeout"]

    if misses:
        lines.append(f"\nMisses only (n={len(misses)}):")
        miss_modes = Counter(r["aim_mode"] for r in misses)
        for k, v in miss_modes.most_common():
            lines.append(f"  mode {k:16s} {v:4d} ({100 * v / len(misses):5.1f}%)")
        yaw_errs = [r["yaw_err_deg"] for r in misses if r["yaw_err_deg"] is not None]
        if yaw_errs:
            lines.append(f"  yaw_err_deg: {arr_stats(yaw_errs)}")
            lines.append(f"  |yaw_err_deg|: {arr_stats([abs(x) for x in yaw_errs])}")
            big = 0
            for r in misses:
                if r["yaw_err_deg"] is None:
                    continue
                lat_from_yaw = abs(math.tan(r["yaw_err_rad"])) * max(r["range_xy"], 1e-3)
                if lat_from_yaw > 1.0:
                    big += 1
            lines.append(
                f"  misses where |yaw| alone predicts >1m lateral: "
                f"{big}/{len(yaw_errs)} ({100 * big / len(yaw_errs):.1f}%)"
            )
        lines.append(f"  signed_lat_m: {arr_stats([r['signed_lat_m'] for r in misses])}")
        lines.append(f"  z_err_at_end_m: {arr_stats([r['z_err_m'] for r in misses])}")
        lines.append(f"  z_margin (max_z - target_z): {arr_stats([r['z_margin_m'] for r in misses])}")
        lines.append(f"  impact_m: {arr_stats([r['impact_m'] for r in misses])}")
        ls = [r["launch_speed"] for r in misses if r["launch_speed"] is not None]
        if ls:
            lines.append(f"  launch_speed: {arr_stats(ls)}")
        low = sum(1 for r in misses if r["z_margin_m"] < -0.5)
        okz = sum(1 for r in misses if -0.5 <= r["z_margin_m"] <= 1.5)
        high = sum(1 for r in misses if r["z_margin_m"] > 1.5)
        lines.append(f"  height band: low={low} ok={okz} high={high}")

    if hits:
        lines.append(f"\nHits only (n={len(hits)}):")
        yaw_errs = [r["yaw_err_deg"] for r in hits if r["yaw_err_deg"] is not None]
        if yaw_errs:
            lines.append(f"  yaw_err_deg: {arr_stats(yaw_errs)}")
            lines.append(f"  |yaw_err_deg|: {arr_stats([abs(x) for x in yaw_errs])}")
        lines.append(f"  z_margin: {arr_stats([r['z_margin_m'] for r in hits])}")
        lines.append(f"  impact_m: {arr_stats([r['impact_m'] for r in hits])}")
        ls = [r["launch_speed"] for r in hits if r["launch_speed"] is not None]
        if ls:
            lines.append(f"  launch_speed: {arr_stats(ls)}")

    if floors:
        lines.append(f"\nFloors (n={len(floors)}):")
        lines.append(f"  z_margin: {arr_stats([r['z_margin_m'] for r in floors])}")
        ls = [r["launch_speed"] for r in floors if r["launch_speed"] is not None]
        if ls:
            lines.append(f"  launch_speed: {arr_stats(ls)}")

    if timeouts:
        lines.append(f"\nTimeouts (n={len(timeouts)}):")
        lines.append(f"  z_margin: {arr_stats([r['z_margin_m'] for r in timeouts])}")
        yaw_errs = [r["yaw_err_deg"] for r in timeouts if r["yaw_err_deg"] is not None]
        if yaw_errs:
            lines.append(f"  |yaw_err_deg|: {arr_stats([abs(x) for x in yaw_errs])}")

    with_yaw = [r for r in recs if r["yaw_err_deg"] is not None]
    if with_yaw:
        lines.append("\nHit rate by |yaw_err| at spawn:")
        for lo, hi, label in [
            (0, 5, "0-5deg"),
            (5, 15, "5-15deg"),
            (15, 30, "15-30deg"),
            (30, 180, "30+deg"),
        ]:
            bucket = [r for r in with_yaw if lo <= abs(r["yaw_err_deg"]) < hi]
            if not bucket:
                continue
            hits_b = sum(1 for r in bucket if r["outcome"] == "hit")
            lines.append(f"  {label:8s} n={len(bucket):3d} hit={100 * hits_b / len(bucket):5.1f}%")

        lines.append("Hit rate by z_margin bucket:")
        for lo, hi, label in [(-99, -0.5, "low"), (-0.5, 1.5, "ok_height"), (1.5, 99, "high")]:
            bucket = [r for r in recs if lo <= r["z_margin_m"] < hi]
            if not bucket:
                continue
            hits_b = sum(1 for r in bucket if r["outcome"] == "hit")
            lines.append(f"  {label:10s} n={len(bucket):3d} hit={100 * hits_b / len(bucket):5.1f}%")

    # Pre-spawn control habits
    fly = [r["fly_ctrl_mean_pre"] for r in recs if r["fly_ctrl_mean_pre"] is not None]
    yaw_c = [r["yaw_ctrl_mean_pre"] for r in recs if r["yaw_ctrl_mean_pre"] is not None]
    if fly:
        lines.append(f"\nPre-spawn mean flywheel ctrl: {arr_stats(fly)}")
    if yaw_c:
        lines.append(f"Pre-spawn mean yaw ctrl: {arr_stats(yaw_c)}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    scale = args.curriculum_scale
    env = DiagnosingEnv(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        max_episode_length=args.max_episode_length,
        curriculum_enabled=True,
        curriculum_start_scale=scale,
        curriculum_max_scale=scale,
    )
    runner = OnPolicyRunner(
        env,
        teacher_ppo_cfg(),
        log_dir=os.path.join(PROJECT_ROOT, "logs", "eval"),
        device=args.device,
    )
    runner.load(args.checkpoint, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    obs = env.get_observations()
    for i in range(env.num_envs):
        env._begin_ep(i)

    while len(env.diag_records) < args.num_episodes:
        with torch.no_grad():
            actions = policy(obs, stochastic_output=False)
        obs, _, _, _ = env.step(actions)

    recs = env.diag_records[: args.num_episodes]
    report = summarize(recs)
    print(report)
    out_dir = os.path.join(PROJECT_ROOT, "logs", "eval", "miss_diagnosis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
