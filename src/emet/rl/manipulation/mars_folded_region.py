"""Conservative fold exposure check for piecewise-linear Mars joint paths."""

import numpy as np

from emet.rl.manipulation.reward_config import mars_folded_pose_cost


def audit_mars_folded_path(positions_rad, *, uncertainty_rad=np.deg2rad(2), risk_limit=0.5):
    """Bound every segment with a joint-space box padded for tracking error.

    An initially exposed arm may leave the region by moving both fold axes
    monotonically away from it. An exposed endpoint never passes.
    """
    q = np.asarray(positions_rad, dtype=float)
    if q.ndim != 2 or q.shape[1] != 5 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("expected at least two finite five-axis poses")
    if not np.isfinite(uncertainty_rad) or uncertainty_rad < 0 or not 0 < risk_limit < 1:
        raise ValueError("expected finite nonnegative uncertainty and a risk limit inside (0, 1)")
    pad = np.array([0, -uncertainty_rad, uncertainty_rad, 0, 0])
    pose_bounds = mars_folded_pose_cost(q + pad)
    corners = q[:-1].copy()
    corners[:, 1] = np.minimum(q[:-1, 1], q[1:, 1])
    corners[:, 2] = np.maximum(q[:-1, 2], q[1:, 2])
    segment_bounds = mars_folded_pose_cost(corners + pad)
    violations = []
    for index, bound in enumerate(segment_bounds):
        if bound <= risk_limit:
            continue
        moves_outward = q[index + 1, 1] < q[index, 1] or q[index + 1, 2] > q[index, 2]
        if pose_bounds[index] <= risk_limit or moves_outward:
            violations.append(index)
    return {
        "accepted": not violations and bool(pose_bounds[-1] <= risk_limit),
        "violating_segments": violations,
        "start_upper_bound": float(pose_bounds[0]),
        "end_upper_bound": float(pose_bounds[-1]),
        "peak_upper_bound": float(segment_bounds.max()),
        "uncertainty_rad": float(uncertainty_rad),
        "risk_limit": risk_limit,
    }
