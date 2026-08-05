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
HIT_BONUS = 8.0
# Raised 5 -> 8: late resumes kept collapsing into floor-dumps because a short
# floor episode was cheaper than a long near-miss flight. Floor must clearly
# lose to any timeout/miss that got reasonably close.
FLOOR_PENALTY = 8.0
# Terminal bonus on timeout/miss = TIMEOUT_MISS_BONUS * closeness(distance).
# Cut 1.5 -> 0.75: collapses were NOT the policy preferring floor (−8) over
# coast (~+2) — floor is worse. The healthy mode was long coast/miss; when a
# mean drift caused under-launches, exploration was already starved so it
# trapped. Keeping miss early-stop, devaluing coast/miss, and raising std/entropy
# (see rl_configs) is the anti-collapse package. Hit (+8) still clearly wins;
# miss/timeout stay above floor for close approaches (~+0.4 at d≈4m).
TIMEOUT_MISS_BONUS = 0.75


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
