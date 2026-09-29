# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bounded camera-view estimates for frontier ranking, never collision certificates."""

import math

import numpy as np


def visible_unknown_cells(obstacles, explored, *, camera_xy, heading, fov, max_range, resolution, to_cell):
    """Raycast a horizontal camera sector; occupied cells occlude unknown cells.

    This 2D estimate ignores vertical occlusion and must be checked against a
    fresh arrival observation. Count unique cells, not rays or padded free space.
    """
    if not np.isfinite([*camera_xy, heading, fov, max_range, resolution]).all():
        raise ValueError("Finite camera geometry required")
    if not 0 < fov < math.pi or min(max_range, resolution) <= 0:
        raise ValueError("Positive range/resolution and valid horizontal FOV required")
    cells = set()
    ray_count = max(2, int(math.ceil(2 * fov * max_range / resolution)) + 1)
    distances = np.arange(0, max_range + resolution / 2, resolution / 2)
    for angle in np.linspace(heading - fov / 2, heading + fov / 2, ray_count):
        direction = np.array([math.cos(angle), math.sin(angle)])
        for distance in distances:
            i, j = to_cell(np.asarray(camera_xy) + distance * direction)
            if not (0 <= i < obstacles.shape[0] and 0 <= j < obstacles.shape[1]) or obstacles[i, j]:
                break
            if not explored[i, j]:
                cells.add((i, j))
    return cells


def make_frontier_evaluator(agent, start, goals, diagnostics):
    """Score full executable routes using the latest calibrated forward camera.

    Called after look_front. Head stays in that measured configuration; rotation
    of the base changes the camera yaw/offset accordingly. No sensor geometry is
    invented when calibration is missing.
    """

    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    obs = agent.robot.get_observation()
    geometry = None
    if obs is not None and all(getattr(obs, key, None) is not None for key in ("camera_K", "camera_pose", "rgb")):
        intrinsics, pose = array(obs.camera_K), array(obs.camera_pose)
        width = array(obs.rgb).shape[1]
        max_range = float(agent.voxel_map.max_depth)
        if (
            intrinsics.shape == (3, 3)
            and pose.shape == (4, 4)
            and np.isfinite(intrinsics).all()
            and np.isfinite(pose).all()
            and intrinsics[0, 0] > 0
            and max_range > 0
        ):
            fov = 2 * math.atan(width / (2 * intrinsics[0, 0]))
            geometry = (pose, fov, max_range)
    obstacles, explored = (array(v).astype(bool) for v in agent.space.get_navigation_map())
    resolution = agent.voxel_map.grid_resolution
    footprint = agent.space._footprint
    radius = math.hypot(
        footprint.width / 2 + abs(footprint.width_offset), footprint.length / 2 + abs(footprint.length_offset)
    )
    seen = set()

    def evaluate(path, index):
        heading = float(goals[index][2])
        waypoints = agent.planner.clean_path_for_xy([[*xy, heading] for xy in path], start_yaw=float(start[2]))
        record = {"index": index, "requested_goal": list(goals[index]), "resolved_goal": list(waypoints[-1])}
        diagnostics.append(record)
        from emet.core.navigation_result import NAVIGATION_POLICIES

        policy = NAVIGATION_POLICIES["exploration"]
        yaw_error = abs(math.atan2(math.sin(heading - start[2]), math.cos(heading - start[2])))
        if (
            np.linalg.norm(np.asarray(path[-1]) - start[:2]) <= policy.xy_tolerance
            and yaw_error <= policy.yaw_tolerance
        ):
            record["reason"] = "already_satisfied_view"
            return None
        key = tuple(round(float(v), 6) for v in waypoints[-1])
        if key in seen:
            record["reason"] = "duplicate_resolved_view"
            return None
        seen.add(key)
        _, reason, clearance = agent._filter_unsafe_nav_traj(waypoints, start_xyt=start, explore_goal=True)
        if reason:
            record.update(reason=reason, footprint=dict(getattr(agent, "_last_nav_sweep_failure", {}) or {}))
            return None
        if geometry is None:
            record["reason"] = "missing_view_calibration"
            return None
        pose, fov, max_range = geometry
        delta = heading - float(start[2])
        rotation = np.array([[math.cos(delta), -math.sin(delta)], [math.sin(delta), math.cos(delta)]])
        camera_xy = np.asarray(path[-1]) + rotation @ (pose[:2, 3] - np.asarray(start[:2]))
        camera_yaw = math.atan2(pose[1, 2], pose[0, 2]) + delta
        cells = visible_unknown_cells(
            obstacles,
            explored,
            camera_xy=camera_xy,
            heading=camera_yaw,
            fov=fov,
            max_range=max_range,
            resolution=resolution,
            to_cell=agent.planner.to_pt,
        )
        distance = sum(
            float(np.linalg.norm(np.asarray(b)[:2] - np.asarray(a)[:2]))
            for a, b in zip(waypoints, waypoints[1:], strict=False)
        )
        turn = 0.0
        previous = float(start[2])
        for waypoint in waypoints:
            turn += abs(math.atan2(math.sin(waypoint[2] - previous), math.cos(waypoint[2] - previous)))
            previous = waypoint[2]
        gain = len(cells) * resolution**2
        record.update(estimated_gain_m2=gain, path_m=distance, turn_rad=turn, min_clearance_m=clearance)
        if not cells:
            record["reason"] = "no_estimated_view_gain"
            return None
        score = gain / (distance + radius * turn + resolution)
        record.update(reason="eligible", score=score)
        return score

    return evaluate
