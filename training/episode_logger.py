"""Periodic episode-table logger.

Accumulates one record per completed episode (outcome, distances, scores) and,
every `log_every` completed episodes, writes the full accumulated table plus a
short text summary to disk. This lets you monitor a long training or eval run
-- including real hit/floor/timeout rates and distances, not just PPO's
internal loss/reward prints -- without waiting for the run to finish.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import datetime

from flywheel_rewards import compute_hit_score, compute_miss_score

FIELDNAMES = [
    "episode_id",
    "env_id",
    "outcome",
    "episode_length",
    "impact_lateral_m",
    "best_miss_m",
    "hit_score",
    "miss_score",
    "curriculum_scale",
]


class EpisodeTableLogger:
    """Accumulates episode records and periodically snapshots them to disk."""

    def __init__(self, output_dir: str | None = None, log_every: int = 100) -> None:
        self.log_every = max(1, log_every)
        self.output_dir = output_dir or os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "logs",
            "episode_tables",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.records: list[dict] = []
        self._last_flushed_count = 0

    def record(
        self,
        env_id: int,
        outcome: str,
        episode_length: int | None,
        impact_lateral: float | None = None,
        best_miss: float | None = None,
        curriculum_scale: float | None = None,
    ) -> None:
        hit_score = compute_hit_score(impact_lateral) if impact_lateral is not None else None
        miss_score = compute_miss_score(best_miss) if best_miss is not None else None
        self.records.append(
            {
                "episode_id": len(self.records) + 1,
                "env_id": env_id,
                "outcome": outcome,
                "episode_length": episode_length,
                "impact_lateral_m": impact_lateral,
                "best_miss_m": best_miss,
                "hit_score": hit_score,
                "miss_score": miss_score,
                "curriculum_scale": curriculum_scale,
            }
        )
        if len(self.records) - self._last_flushed_count >= self.log_every:
            self.flush()

    def flush(self) -> None:
        """Write the full accumulated table + summary to disk, regardless of interval."""
        if not self.records:
            return
        self._last_flushed_count = len(self.records)
        total = len(self.records)

        self._write_csv(os.path.join(self.output_dir, f"episodes_{total:06d}.csv"))
        self._write_csv(os.path.join(self.output_dir, "episodes_latest.csv"))
        self._write_summary(os.path.join(self.output_dir, "summary_latest.txt"))

    def _write_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.records)

    def _write_summary(self, path: str) -> None:
        total = len(self.records)
        outcomes = Counter(r["outcome"] for r in self.records)

        lines = [
            f"Episode table snapshot -- {total} episodes completed",
            "=" * 48,
        ]
        for name in ("hit", "floor", "timeout"):
            count = outcomes.get(name, 0)
            pct = 100.0 * count / total if total else 0.0
            lines.append(f"{name:>7}: {count:6d} ({pct:5.1f}%)")

        hit_scores = [r["hit_score"] for r in self.records if r["hit_score"] is not None]
        if hit_scores:
            lines.append(f"Mean hit score:  {sum(hit_scores) / len(hit_scores):.3f} (out of 5)")

        miss_scores = [r["miss_score"] for r in self.records if r["miss_score"] is not None]
        if miss_scores:
            lines.append(f"Mean miss score: {sum(miss_scores) / len(miss_scores):.3f} (out of 1)")

        best_misses = [r["best_miss_m"] for r in self.records if r["best_miss_m"] is not None]
        if best_misses:
            lines.append(f"Mean best miss distance: {sum(best_misses) / len(best_misses):.3f} m")

        current_scale = self.records[-1].get("curriculum_scale")
        if current_scale is not None:
            lines.append(f"Curriculum scale (current): {current_scale:.3f}")

        # Also show the trend over the most recent window, so improvement over
        # time is visible even once `records` has grown large.
        window = self.records[-self.log_every :]
        window_outcomes = Counter(r["outcome"] for r in window)
        lines.append("")
        lines.append(f"Last {len(window)} episodes:")
        for name in ("hit", "floor", "timeout"):
            count = window_outcomes.get(name, 0)
            pct = 100.0 * count / len(window) if window else 0.0
            lines.append(f"{name:>7}: {count:6d} ({pct:5.1f}%)")

        with open(path, "w", encoding="utf-8") as summary_file:
            summary_file.write("\n".join(lines))
            summary_file.write("\n")
