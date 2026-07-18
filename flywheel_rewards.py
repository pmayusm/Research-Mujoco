"""Shared reward functions for flywheel simulation and RL training."""

REWARD_ZERO_DISTANCE = 6.0
HIT_SCORE_MAX = 5.0
MISS_SCORE_MAX = 1.0
DENSE_REWARD_SCALE = 0.1


def compute_target_reward(lateral_distance):
    """Return accuracy in [0, 1]. Zero beyond 6 m, increasing toward the bullseye."""
    if lateral_distance >= REWARD_ZERO_DISTANCE:
        return 0.0
    progress = 1.0 - (lateral_distance / REWARD_ZERO_DISTANCE)
    return progress ** 2


def compute_hit_score(impact_lateral):
    """Reward in [0, 5], based only on where the ball actually hits the face."""
    return HIT_SCORE_MAX * compute_target_reward(impact_lateral)


def compute_miss_score(best_lateral_distance):
    """Reward in [0, 1], based on best aim during the flight."""
    return MISS_SCORE_MAX * compute_target_reward(best_lateral_distance)


def compute_closeness(ball_pos, slab_pos, plane_normal, target_half_thickness):
    """Return miss distance, lateral distance, and face distance to the target."""
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
    """Lateral miss distance on the shooter-facing front face of the target box."""
    front_face_center = slab_pos + target_half_thickness * plane_normal
    offset = ball_pos - front_face_center
    tangential = offset - (offset @ plane_normal) * plane_normal
    return float((tangential @ tangential) ** 0.5)
