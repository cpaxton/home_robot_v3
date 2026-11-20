"""Episode-weighted coverage of declared simulator goal and path workspaces."""

from collections import Counter
from math import prod

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.trajectory_metrics import rotations


def bin_indices(points, edges):
    """Assign finite vectors to fixed bins. Values outside get index -1."""
    boundaries = tuple(np.asarray(axis, dtype=float) for axis in edges)
    if not boundaries or any(
        axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0)
        for axis in boundaries
    ):
        raise ValueError("Each coordinate needs finite, strictly increasing bin edges")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != len(boundaries) or not np.isfinite(points).all():
        raise ValueError("Expected finite points with one column per bin axis")
    shape = tuple(len(axis) - 1 for axis in boundaries)
    indices = np.column_stack(
        [
            np.minimum(np.searchsorted(axis, points[:, i], side="right") - 1, shape[i] - 1)
            for i, axis in enumerate(boundaries)
        ]
    )
    valid = np.ones(len(points), dtype=bool)
    for i, axis in enumerate(boundaries):
        valid &= (points[:, i] >= axis[0]) & (points[:, i] <= axis[-1])
    result = np.full(len(points), -1, dtype=int)
    result[valid] = np.ravel_multi_index(indices[valid].T, shape)
    return result, shape


def _distribution(mass, shape, eligible):
    count = prod(shape)
    allowed = None if eligible is None else set(eligible)
    if allowed is not None and (not allowed or any(not 0 <= i < count for i in allowed)):
        raise ValueError("Expected nonempty eligible bins inside the workspace grid")
    eligible_count = count if allowed is None else len(allowed)
    included = {i: m for i, m in mass.items() if i >= 0 and (allowed is None or i in allowed)}
    total = sum(included.values())
    proportions = np.asarray(list(included.values()), dtype=float) / total if total else np.empty(0)
    entropy = -np.sum(proportions * np.log(proportions))
    uniform = 1 / eligible_count
    return {
        "eligible_bins": eligible_count,
        "occupied_bins": len(included),
        "occupied_fraction": len(included) / eligible_count,
        "effective_bins": float(np.exp(entropy)) if total else 0.0,
        "distance_to_uniform": (
            float((np.abs(proportions - uniform).sum() + (eligible_count - len(included)) * uniform) / 2)
            if total
            else None
        ),
        "bins": [
            {"index": list(map(int, np.unravel_index(index, shape))), "mass": float(value)}
            for index, value in sorted(included.items())
        ],
    }


def goal_coverage(points, edges, *, eligible=None):
    """Count each episode's endpoint once against a frozen grid."""
    indices, shape = bin_indices(points, edges)
    allowed = None if eligible is None else set(eligible)
    counts = Counter(map(int, indices))
    return {
        **_distribution(counts, shape, allowed),
        "episodes": len(indices),
        "outside_grid": counts.get(-1, 0),
        "outside_eligible_workspace": sum(n for i, n in counts.items() if i >= 0 and i not in allowed)
        if allowed is not None
        else 0,
    }


def path_coverage(paths, edges, *, eligible=None):
    """Give each trajectory one vote, split across distinct eligible cells visited."""
    allowed = None if eligible is None else set(eligible)
    mass = Counter()
    shape = None
    paths_in_workspace = count = 0
    for path in paths:
        indices, shape = bin_indices(path, edges)
        visited = {int(i) for i in indices if i >= 0 and (allowed is None or i in allowed)}
        count += 1
        if visited:
            paths_in_workspace += 1
            mass.update({i: 1 / len(visited) for i in visited})
    if shape is None:
        raise ValueError("At least one trajectory is required")
    return {
        **_distribution(mass, shape, allowed),
        "episodes": count,
        "episodes_without_eligible_visits": count - paths_in_workspace,
    }


def balanced_goal_weights(points, edges, *, eligible=None):
    """Sampling weights that choose an occupied goal cell before its episodes."""
    indices, shape = bin_indices(points, edges)
    allowed = None if eligible is None else set(eligible)
    if allowed is not None and (not allowed or any(not 0 <= i < prod(shape) for i in allowed)):
        raise ValueError("Expected nonempty eligible bins inside the workspace grid")
    valid = (indices >= 0) & (
        np.asarray([int(i) in allowed for i in indices]) if allowed is not None else np.ones(len(indices), dtype=bool)
    )
    counts = Counter(indices[valid])
    if not counts:
        raise ValueError("No episodes occupy an eligible cell")
    weights = np.zeros(len(indices))
    weights[valid] = [1 / (len(counts) * counts[index]) for index in indices[valid]]
    return weights


def spatial_path(hand, base, frame):
    """Return position and orientation in the selected expert's goal frame."""
    base = np.asarray(base, dtype=float)
    hand = np.asarray(hand, dtype=float)
    if (
        base.ndim != 2
        or base.shape[1] != 3
        or hand.shape != (len(base), 9)
        or not len(base)
        or not np.isfinite(base).all()
        or not np.isfinite(hand).all()
    ):
        raise ValueError("Expected paired base and hand poses")
    yaw = Rotation.from_euler("z", base[:, 2])
    if frame == "base_start":
        return yaw[0].inv().apply(np.column_stack((base[:, :2] - base[0, :2], np.zeros(len(base)))))[:, :2], yaw[
            0
        ].inv() * yaw
    hand_rotation = Rotation.from_matrix(rotations(hand))
    if frame == "initial_hand":
        return hand_rotation[0].inv().apply(hand[:, :3] - hand[0, :3]), hand_rotation[0].inv() * hand_rotation
    if frame in {"body_hand", "start_base_hand"}:
        xy = base[:, :2] if frame == "body_hand" else np.broadcast_to(base[0, :2], (len(base), 2))
        orientation = yaw if frame == "body_hand" else yaw[0]
        position = hand[:, :3] - np.column_stack((xy, np.zeros(len(base))))
        return orientation.inv().apply(position), orientation.inv() * hand_rotation
    raise ValueError(f"Unknown goal frame {frame!r}")


def nearest_orientation_gaps(orientations):
    """Nearest-neighbor SO(3) angular gaps in radians, including duplicate goals."""
    quaternions = orientations.as_quat()
    if quaternions.ndim != 2:
        raise ValueError("Expected a sequence of rotations")
    n = len(quaternions)
    if n < 2:
        return np.empty(0)
    distance, indices = cKDTree(np.vstack((quaternions, -quaternions))).query(quaternions, k=3)
    valid = indices % n != np.arange(n)[:, None]
    nearest = np.min(np.where(valid, distance, np.inf), axis=1)
    return 4 * np.arcsin(np.clip(nearest / 2, 0, 1))
