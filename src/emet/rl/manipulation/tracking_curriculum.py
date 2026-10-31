"""Validation-gated reward precision and later motion regularization."""

import math

from emet.rl.manipulation.evaluation import tracking_precision_ready


def tracking_updates(config, summary, episodes, *, lift_summary=None, current_motion_scale=None):
    """All available training strata must pass; loaded failures cannot be hidden."""
    if not config.tracking_reward or not tracking_precision_ready(summary, config.tracking_precision, episodes):
        return {}
    if config.grasp_lift_probability and (
        lift_summary is None or not tracking_precision_ready(lift_summary, config.tracking_precision, episodes)
    ):
        return {}
    # Pose precision must not advance while the base violates its active
    # heading contract. The loaded subtype always has the stricter fixed base.
    for groups, heading_limit in (
        (summary, config.base_heading_tolerance),
        (lift_summary or {}, min(0.01, config.base_heading_tolerance)),
    ):
        for group in groups.values():
            if group.get("base_target_count", 0):
                heading = group.get("base_heading_error_rad_p95")
                if heading is None or not math.isfinite(heading) or heading > heading_limit:
                    return {}
    updates = {}
    level = min(4, config.tracking_precision + 1)
    if level != config.tracking_precision:
        updates["tracking_precision"] = level
    scale = config.motion_penalty_scale if current_motion_scale is None else current_motion_scale
    if config.smoothness_curriculum:
        next_scale = max(scale, (0.05, 0.1, 0.1, 0.4, 1.0)[level])
        if next_scale != scale:
            updates["motion_penalty_scale"] = next_scale
    return updates
