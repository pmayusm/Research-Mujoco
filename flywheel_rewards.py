"""Shared reward functions for flywheel simulation and RL training.

Episode return is:
  1. Dense distance reward every step after the ball spawns.
  2. A flat bonus if the ball hits the target.
  3. A flat penalty if the ball hits the floor.
  4. On timeout: a miss-quality bonus from best approach distance (not zero).

Dense score uses a softened exponential so mid-range misses still produce a
usable gradient. Plain exp(-d) collapses to ~0 by d≈4-5m (typical early miss
distances), which left only the floor penalty as a learning signal and the
policy collapsed to 100% floor-drops within ~1000 episodes.
"""

from __future__ import annotations

import math

# Soft length scale (meters) inside exp(-d / L). L≈6 keeps mid-range misses
# (~3-6m) clearly above near-zero so the dense term still shapes aiming.
DISTANCE_LENGTH_SCALE = 6.0

# Clamped band for the raw distance score (before per-step scaling).
DENSE_REWARD_MIN = 0.0
DENSE_REWARD_MAX = 1.0
# Per-step multiplier. Cut 0.002 -> 0.001 after v6b: a ~2700-step coast at
# typical miss (~4m, closeness≈0.5) was paying ~2.7 dense + timeout bonus ≈ hit,
# so the policy preferred timeout-coasting over committing to hits.
DENSE_REWARD_SCALE = 0.001
# Raised 16 -> 24: overnight run solved floor (~0-3%) but miss still ~74%.
# Only training the main (privileged) policy now — push hit vs miss harder.
HIT_BONUS = 24.0
# Floor already rare under gear=1800/spawn=100 physics; keep strong enough that
# under-launch cannot become cheaper than a miss again.
FLOOR_PENALTY = 8.0
# Cut 0.35 -> 0.1: close miss ~+0.05, hit +24. Stops "good enough miss" plateau.
TIMEOUT_MISS_BONUS = 0.1

# Pre-spawn yaw alignment (diagnosis: ~80% of misses are off-boresight; hit rate
# jumps to ~39% when |aim_err| < 15° at spawn). Dense reward while the ball is
# still hidden teaches the policy to finish aiming before launch.
# Angle scale ~15°: full credit near 0, ~e^-2 at 30°, near-zero by 60°+.
AIM_ANGLE_SCALE_RAD = 0.26
# Per control-step. 100 well-aimed spin-up steps ≈ +5, clearly worth learning
# but still well below HIT_BONUS so the agent cannot farm aim forever.
AIM_ALIGN_REWARD_SCALE = 0.05
# Flat bonus at the spawn step if aim is already inside the gate band.
AIM_SPAWN_BONUS = 2.0


def aim_alignment_score(aim_error_rad: float) -> float:
    """1 at perfect boresight, falls with |aim_error| / AIM_ANGLE_SCALE_RAD."""
    return math.exp(-abs(float(aim_error_rad)) / AIM_ANGLE_SCALE_RAD)


def dense_aim_reward(aim_error_rad: float) -> float:
    """Per-step pre-spawn reward for pointing the hood at the target."""
    return AIM_ALIGN_REWARD_SCALE * aim_alignment_score(aim_error_rad)


def aim_spawn_bonus(aim_error_rad: float) -> float:
    """Bonus paid once when the ball spawns, scaled by aim quality."""
    return AIM_SPAWN_BONUS * aim_alignment_score(aim_error_rad)


def closeness(distance: float) -> float:
    """Exponential distance score, clamped into [DENSE_REWARD_MIN, DENSE_REWARD_MAX].

    score = clamp(exp(-distance / DISTANCE_LENGTH_SCALE), min, max)

    1.0 at d=0; falls toward 0 as distance grows. Far early flight pays near the
    lower limit; reward rises toward the upper limit as the ball gets close.
    """
    raw = math.exp(-float(distance) / DISTANCE_LENGTH_SCALE)
    if raw < DENSE_REWARD_MIN:
        return DENSE_REWARD_MIN
    if raw > DENSE_REWARD_MAX:
        return DENSE_REWARD_MAX
    return raw


def dense_distance_reward(distance: float) -> float:
    """Per-step dense reward from miss distance."""
    return DENSE_REWARD_SCALE * closeness(distance)


def timeout_terminal_reward(best_miss: float | None) -> float:
    """Terminal reward for max-length episodes based on closest approach."""
    if best_miss is None:
        return 0.0
    try:
        miss = float(best_miss)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(miss):
        return 0.0
    return TIMEOUT_MISS_BONUS * closeness(miss)


def compute_target_distances(ball_pos, slab_pos, plane_normal, target_half_thickness):
    """Return (miss_distance, lateral_distance, face_distance) to the target.

    Geometry helper, not a reward formula.
    """
    front_face_center = slab_pos + target_half_thickness * plane_normal
    offset = ball_pos - front_face_center
    normal_component = offset @ plane_normal
    tangential = offset - normal_component * plane_normal
    lateral_distance = float((tangential @ tangential) ** 0.5)

    if normal_component >= 0:
        miss_distance = float((normal_component**2 + lateral_distance**2) ** 0.5)
    else:
        miss_distance = lateral_distance

    face_distance = max(0.0, float(normal_component))
    return miss_distance, lateral_distance, face_distance


def impact_distance_on_face(ball_pos, slab_pos, plane_normal, target_half_thickness):
    """Lateral distance from center on the shooter-facing front face (logging only)."""
    front_face_center = slab_pos + target_half_thickness * plane_normal
    offset = ball_pos - front_face_center
    tangential = offset - (offset @ plane_normal) * plane_normal
    return float((tangential @ tangential) ** 0.5)
