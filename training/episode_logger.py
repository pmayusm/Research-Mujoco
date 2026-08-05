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

from flywheel_rewards import FLOOR_PENALTY, HIT_BONUS, timeout_terminal_reward

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: never tries to open a window
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

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

    def __init__(self, output_dir: str | None = None, log_every: int = 100, plot_every: int = 100) -> None:
        self.log_every = max(1, log_every)
        self.plot_every = max(1, plot_every)
        self.output_dir = output_dir or os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "logs",
            "episode_tables",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.records: list[dict] = []
        self._last_flushed_count = 0
        self._last_plotted_count = 0
        # One (episode_count, hit_accuracy) point per plot_every-episode bucket,
        # accumulated over the whole run so the plot shows the full trend.
        self._accuracy_points: list[tuple[int, float]] = []
        if not _MATPLOTLIB_AVAILABLE:
            print("[EpisodeTableLogger] matplotlib not installed -- skipping hit-accuracy plot (pip install matplotlib).")

    def record(
        self,
        env_id: int,
        outcome: str,
        episode_length: int | None,
        impact_lateral: float | None = None,
        best_miss: float | None = None,
        curriculum_scale: float | None = None,
    ) -> None:
        # Terminal scores mirror env payout: hit -> HIT_BONUS, floor -> -FLOOR_PENALTY,
        # miss/timeout -> TIMEOUT_MISS_BONUS * closeness(...).
        hit_score = HIT_BONUS if outcome == "hit" else None
        if outcome == "floor":
            miss_score = -FLOOR_PENALTY
        elif outcome == "miss":
            miss_score = timeout_terminal_reward(
                impact_lateral if impact_lateral is not None else best_miss
            )
        elif outcome == "timeout":
            miss_score = timeout_terminal_reward(best_miss)
        else:
            miss_score = None
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
        if len(self.records) - self._last_plotted_count >= self.plot_every:
            self._update_accuracy_plot()

    def flush(self) -> None:
        """Write the latest full table + summary to disk, regardless of interval.

        Only keeps episodes_latest.csv / summary_latest.txt (overwritten in place).
        Writing a new full cumulative CSV every flush used to fill the disk -- at
        ~100k episodes that is O(n^2) bytes across snapshots.
        """
        if not self.records:
            return
        self._last_flushed_count = len(self.records)

        self._write_csv(os.path.join(self.output_dir, "episodes_latest.csv"))
        self._write_summary(os.path.join(self.output_dir, "summary_latest.txt"))
        # Flush any leftover partial bucket so the chart includes every episode,
        # not just whole plot_every-sized buckets.
        if self._last_plotted_count < len(self.records):
            self._update_accuracy_plot()

    def _update_accuracy_plot(self) -> None:
        """Record a new (episode_count, hit_accuracy) point for the most recent
        bucket of `plot_every` episodes and redraw the full hit-accuracy chart."""
        bucket = self.records[self._last_plotted_count :]
        self._last_plotted_count = len(self.records)
        if not bucket:
            return

        hit_rate = 100.0 * sum(1 for r in bucket if r["outcome"] == "hit") / len(bucket)
        self._accuracy_points.append((len(self.records), hit_rate))

        with open(os.path.join(self.output_dir, "hit_accuracy.csv"), "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["episode", "hit_accuracy_pct"])
            writer.writerows(self._accuracy_points)

        if not _MATPLOTLIB_AVAILABLE:
            return

        episodes = [p[0] for p in self._accuracy_points]
        accuracies = [p[1] for p in self._accuracy_points]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, accuracies, marker="o", linestyle="-", markersize=4, color="tab:blue")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Hit accuracy (%)")
        ax.set_title(f"Hit accuracy per {self.plot_every}-episode bucket")
        ax.set_ylim(0, max(10, max(accuracies) * 1.2))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "hit_accuracy.png"), dpi=120)
        plt.close(fig)

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
        for name in ("hit", "floor", "miss", "timeout"):
            count = outcomes.get(name, 0)
            pct = 100.0 * count / total if total else 0.0
            lines.append(f"{name:>7}: {count:6d} ({pct:5.1f}%)")

        # hit_score/miss_score mirror the terminal reward actually paid out.
        terminal_rewards = [
            r["hit_score"] if r["hit_score"] is not None else r["miss_score"]
            for r in self.records
            if r["hit_score"] is not None or r["miss_score"] is not None
        ]
        if terminal_rewards:
            lines.append(f"Mean terminal reward: {sum(terminal_rewards) / len(terminal_rewards):.3f}")

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
        for name in ("hit", "floor", "miss", "timeout"):
            count = window_outcomes.get(name, 0)
            pct = 100.0 * count / len(window) if window else 0.0
            lines.append(f"{name:>7}: {count:6d} ({pct:5.1f}%)")

        with open(path, "w", encoding="utf-8") as summary_file:
            summary_file.write("\n".join(lines))
            summary_file.write("\n")

    def recent_window_rates(self, window: int = 500) -> dict[str, float] | None:
        """Hit/floor/miss/timeout rates over the most recent `window` episodes."""
        if len(self.records) < window:
            return None
        chunk = self.records[-window:]
        total = len(chunk)
        counts = Counter(r["outcome"] for r in chunk)
        return {name: counts.get(name, 0) / total for name in ("hit", "floor", "miss", "timeout")}

    def best_checkpoint_score(self, window: int = 500, max_floor: float = 0.15) -> float | None:
        """Score for keeping a checkpoint: hit rate if floor is under control, else None.

        Rejects windows with floor_rate > max_floor so we never promote a
        half-collapsed policy just because a few hits sneaked through.
        """
        rates = self.recent_window_rates(window)
        if rates is None or rates["floor"] > max_floor:
            return None
        return rates["hit"] - 0.25 * rates["floor"]
