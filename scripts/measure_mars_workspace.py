"""Measure how evenly Mars expert goals and paths cover declared workspaces."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.workspace_coverage import (
    balanced_goal_weights,
    bin_indices,
    goal_coverage,
    nearest_orientation_gaps,
    path_coverage,
    spatial_path,
)


def quantiles(values):
    return dict(zip(("min", "median", "p90", "max"), np.quantile(values, (0, 0.5, 0.9, 1)).tolist(), strict=True))


def measure(spec):
    episodes = spec["episodes"]
    edges, frame = spec["edges"], spec["frame"]
    eligible = spec.get("eligible_bins")
    poses = [spatial_path(row["hand"], row["base"], frame) for row in episodes]
    paths = [pose[0] for pose in poses]
    starts = np.asarray([path[0] for path in paths])
    goals = np.asarray([path[-1] for path in paths])
    lengths = np.asarray([np.linalg.norm(np.diff(path, axis=0), axis=1).sum() for path in paths])
    displacements = np.asarray([np.linalg.norm(path[-1] - path[0]) for path in paths])
    angles = np.asarray([(r[:-1].inv() * r[1:]).magnitude().sum() for _, r in poses])
    orientation_gap = nearest_orientation_gaps(Rotation.concatenate([r[-1] for _, r in poses]))
    endpoint = goal_coverage(goals, edges, eligible=eligible)
    start_bins, _ = bin_indices(starts, edges)
    goal_bins, _ = bin_indices(goals, edges)
    allowed = None if eligible is None else set(eligible)
    transitions = {
        (int(i), int(j))
        for i, j in zip(start_bins, goal_bins, strict=True)
        if i >= 0 and j >= 0 and (allowed is None or {i, j} <= allowed)
    }
    result = {
        "frame": frame,
        "source_scope": spec.get("source_scope"),
        "edges": edges,
        "eligible_workspace": "declared cells" if eligible is not None else "full grid, reachability unclassified",
        "goal_coverage": endpoint,
        "start_coverage": goal_coverage(starts, edges, eligible=eligible),
        "distinct_start_goal_bin_pairs": len(transitions),
        "path_coverage": path_coverage(paths, edges, eligible=eligible),
        "path_length_m": quantiles(lengths),
        "endpoint_displacement_m": quantiles(displacements),
        "orientation_travel_rad": quantiles(angles),
        "nearest_goal_orientation_rad": quantiles(orientation_gap) if len(orientation_gap) else None,
        "weights": [],
    }
    if endpoint["occupied_bins"]:
        weights = balanced_goal_weights(goals, edges, eligible=eligible)
        result["weights"] = [
            {"id": row["id"], "goal_m": goal.tolist(), "weight": float(weight)}
            for row, goal, weight in zip(episodes, goals, weights, strict=True)
        ]
    if frame == "base_start":
        bearings = [rot.as_euler("zyx")[:, 0] for _, rot in poses]
        forward = [
            (np.diff(path, axis=0) * np.column_stack((np.cos(yaw[:-1]), np.sin(yaw[:-1])))).sum(axis=1)
            for path, yaw in zip(paths, bearings, strict=True)
        ]
        result["reverse_path_fraction"] = quantiles(
            [float(-delta[delta < 0].sum() / abs(delta).sum()) if np.any(delta) else 0.0 for delta in forward]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.workspace.read_text())
    result = {"families": {}}
    for name, family in spec["families"].items():
        if not family["episodes"]:
            raise ValueError(f"No episodes for {name}")
        result["families"][name] = measure(family)
        print(name, result["families"][name]["goal_coverage"]["occupied_bins"])
    save(args.output, result)


if __name__ == "__main__":
    main()
