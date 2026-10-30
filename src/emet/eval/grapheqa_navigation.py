# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sensor-only navigation adapter for the external GraphEQA qualification.

The planner and sweep checks are the shared emet implementations. The renderer
uses Habitat's collision filter internally; no global pose, navmesh, full-scene
bounds or semantic annotations are exposed to this adapter's planning path.
"""

import time
from dataclasses import asdict

import numpy as np

from emet.eval.grapheqa_sensor import command_camera_pose
from emet.mapping.rgbd_odometry import RGBDOdometry


def base_xyt(camera_pose):
    """Local Z-up heading from the calibrated forward optical axis."""
    return np.array([*camera_pose[:2, 3], np.arctan2(camera_pose[1, 2], camera_pose[0, 2])])


def predicted_camera_pose(camera_pose, action, amount=None):
    """Small command prior, never used as a measured pose."""
    pose = np.asarray(camera_pose).copy()
    yaw = base_xyt(pose)[2]
    if action == "move_forward":
        pose[:2, 3] += (0.1 if amount is None else amount) * np.array([np.cos(yaw), np.sin(yaw)])
    elif action in {"turn_left", "turn_right"}:
        angle = np.deg2rad((5 if amount is None else amount) * (1 if action == "turn_left" else -1))
        c, s = np.cos(angle), np.sin(angle)
        pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ pose[:3, :3]
    elif action in {"look_down", "look_up"}:
        angle = np.deg2rad((5 if amount is None else amount) * (-1 if action == "look_down" else 1))
        c, s = np.cos(angle), np.sin(angle)
        pose[:3, :3] = pose[:3, :3] @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        raise ValueError(f"Unsupported motion: {action}")
    return pose


def stationary_floor_reference(frame, *, support_radius_m=0.1):
    """Measure one flat surface under a declared safe, stationary spawn.

    This is startup calibration for the Habitat body-frame height convention,
    not a travelling ground estimator. It must not be used to erase a newly
    encountered drop. Ambiguous, sloped or insufficient support is rejected.
    """
    depth, k, pose = frame["depth"], frame["camera_K"], frame["camera_pose"]
    if pose is None:
        raise ValueError("Floor calibration requires accepted camera odometry")
    v, u = np.mgrid[0 : depth.shape[0] : 2, 0 : depth.shape[1] : 2]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0.1) & (z < 6)
    xyz = np.stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z), axis=-1)
    xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    within = valid & (np.linalg.norm(xyz[..., :2] - pose[:2, 3], axis=-1) <= support_radius_m)
    points = xyz[within]
    if len(points) < 64 or np.ptp(points[:, :2], axis=0).min() < 0.05:
        raise ValueError("Insufficient measured floor under the stationary footprint")
    low, high = np.percentile(points[:, 2], [5, 95])
    reference = float(np.median(points[:, 2]))
    if abs(reference) > 0.2 or high - low > 0.02:
        raise ValueError("Ambiguous or out-of-range stationary floor calibration")
    return {
        "reference_z_m": reference,
        "support_points": len(points),
        "height_spread_m": float(high - low),
        "timestamp_ns": int(frame["timestamp_ns"]),
    }


class SensorOnlyHabitat:
    """Minimal renderer boundary: prescribed spawn in, registered RGB-D out."""

    def __init__(self, config, spawn):
        import habitat_sim
        from habitat_sim.utils.common import quat_from_angle_axis

        agent = config.agents[0]
        agent.radius, agent.height = 0.1, 1.5
        agent.action_space = {
            name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
            for name, amount in [
                ("move_forward", 0.1),
                ("turn_left", 5.0),
                ("turn_right", 5.0),
                ("look_down", 5.0),
                ("look_up", 5.0),
            ]
        }
        config.sim_cfg.allow_sliding = False
        self.sim = habitat_sim.Simulator(config)
        if not self.sim.pathfinder.is_loaded:
            self.sim.close()
            raise RuntimeError("Renderer collision filter is unavailable; refusing unchecked base motion")
        state = habitat_sim.AgentState()
        state.position = [spawn.x, spawn.y, spawn.z]
        state.rotation = quat_from_angle_axis(spawn.heading, np.array([0.0, 1.0, 0.0]))
        self.sim.initialize_agent(0, state)
        focal = 320.0 / np.tan(np.deg2rad(60))
        self.k = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1]], dtype=float)
        self.odometry = RGBDOdometry(self.k, command_camera_pose(yaw_deg=0, tilt_deg=-30, sensor_height=1.5))
        self.sequence = 0
        self.last = self._capture(self.sim.get_sensor_observations(), None)

    def _capture(self, raw, guess):
        rgb = np.ascontiguousarray(raw["rgb"][..., :3], dtype=np.uint8)
        depth = np.ascontiguousarray(raw["depth"], dtype=np.float32)
        stamp = time.monotonic_ns()
        receipt = self.odometry.update(rgb, depth, stamp, source_to_target_guess=guess)
        self.sequence += 1
        record = asdict(receipt)
        if receipt.camera_pose is not None:
            record["camera_pose"] = receipt.camera_pose.tolist()
        result = {
            "rgb": rgb,
            "depth": depth,
            "camera_K": self.k,
            "timestamp_ns": stamp,
            "camera_pose": receipt.camera_pose,
            "sequence": self.sequence,
            "collided": bool(raw.get("collided", False)),
            "odometry": record,
        }
        self.last = result
        return result

    def step(self, action, amount=None):
        if self.last["camera_pose"] is None:
            raise RuntimeError("Tracking lost: refusing further motion")
        limit = 0.1 if action == "move_forward" else 5.0
        amount = limit if amount is None else amount
        if not np.isfinite(amount) or not 0 < amount <= limit:
            raise ValueError("Primitive exceeds the calibrated small-step bound")
        self.sim.get_agent(0).agent_config.action_space[action].actuation.amount = float(amount)
        predicted = predicted_camera_pose(self.last["camera_pose"], action, amount)
        elevation = np.rad2deg(np.arcsin(np.clip(predicted[2, 2], -1, 1)))
        if not -65.1 <= elevation <= -14.9:
            raise ValueError("Head motion exceeds the declared inspection envelope")
        guess = np.linalg.inv(predicted) @ self.last["camera_pose"]
        return self._capture(self.sim.step(action), guess)

    def capture(self):
        return self._capture(self.sim.get_sensor_observations(), None)

    def close(self):
        self.sim.close()


class ObservedSpaceNavigation:
    """Shared geometry map, observed-space search and swept-footprint admission."""

    def __init__(self):
        from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap
        from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace
        from emet.robots.footprint import CircularFootprint

        self.map = SparseVoxelMap(
            resolution=0.01,
            grid_size=(256, 256),
            grid_resolution=0.05,
            min_depth=0.1,
            max_depth=6.0,
            obs_min_height=0.08,
            obs_max_height=1.5,
            obs_min_density=3,
            smooth_kernel_size=0,
            add_local_radius_points=False,
            min_points_per_voxel=1,
            device="cpu",
            map_2d_device="cpu",
            run_eqa=False,
            parameters={"floor_evidence": {"enabled": True, "reference_z_m": 0.0}},
        )
        self.footprint = CircularFootprint(0.1)
        self.support = np.zeros((256, 256), dtype=bool)
        self.space = SparseVoxelMapNavigationSpace(
            self.map,
            obstacle_map_mode="physical",
            footprint=self.footprint,
            dilate_frontier_size=0,
            dilate_obstacle_size=0,
        )
        self.pose = None
        self.floor_calibration = None

    def calibrate_stationary_floor(self, frame):
        import torch

        if self.floor_calibration is not None:
            raise RuntimeError("Startup floor calibration is fixed for the entire episode")
        if self.pose is None or np.linalg.norm(self.pose[:2]) > 0.025:
            raise RuntimeError("Floor calibration requires the original stationary spawn")
        self.floor_calibration = stationary_floor_reference(frame)
        self.map.parameters["floor_evidence"]["reference_z_m"] = self.floor_calibration["reference_z_m"]
        # Invalidate only the 2D projection, retaining the measured 3D cloud.
        self.map._map2d = None
        obstacles, explored = self.map.get_navigation_map()
        self.map._physical_map2d = (obstacles, explored | torch.tensor(self.support))
        return dict(self.floor_calibration)

    def update(self, frame):
        import torch

        if frame["camera_pose"] is None:
            raise RuntimeError("Cannot map a frame without accepted odometry")
        self.pose = base_xyt(frame["camera_pose"])
        # Only the occupied footprint is proprioceptive support. It cannot
        # clear obstacles/drops, nor certify a surrounding disk as known floor.
        cells = self.footprint.grid_cells(0.05, self.pose, self.map.grid_origin.numpy())
        if (cells < 0).any() or (cells >= np.asarray(self.support.shape)).any():
            raise RuntimeError("Estimated footprint outside the declared local map")
        self.support[cells[:, 0], cells[:, 1]] = True
        k = frame["camera_K"].copy()
        k[:2] /= 2
        self.map.add(
            camera_pose=torch.tensor(frame["camera_pose"], dtype=torch.float32),
            rgb=torch.tensor(frame["rgb"][::2, ::2].copy()),
            depth=torch.tensor(frame["depth"][::2, ::2].copy()),
            camera_K=torch.tensor(k, dtype=torch.float32),
            base_pose=torch.tensor(self.pose, dtype=torch.float32),
        )
        obstacles, explored = self.map.get_navigation_map()
        # This adapter's visited support is explicit and uses the same grid and
        # physical footprint as admission. Terrain remains in obstacles.
        self.map._physical_map2d = (obstacles, explored | torch.tensor(self.support))

    def _navigable(self):
        from scipy.ndimage import binary_erosion

        obstacles, explored = (m.numpy() for m in self.map.get_navigation_map())
        kernel = self.footprint.get_conservative_rotated_mask(0.05)
        return binary_erosion(explored & ~obstacles, structure=kernel), explored

    def frontiers(self, limit=8):
        from scipy.ndimage import binary_dilation, label

        nav, explored = self._navigable()
        edge = nav & binary_dilation(~explored, iterations=4)
        components, count = label(edge)
        candidates = []
        for component in range(1, count + 1):
            cells = np.argwhere(components == component)
            center = cells.mean(axis=0)
            chosen = cells[np.argmin(np.linalg.norm(cells - center, axis=1))]
            xy = (chosen - self.map.grid_origin.numpy()[:2]) * 0.05
            if np.linalg.norm(xy - self.pose[:2]) > 0.3:
                candidates.append((len(cells), [*xy.tolist(), 0.0]))
        # Reachability is measured on the same observed grid. Do not advertise
        # disconnected free-looking islands as actionable frontiers.
        from emet.motion.base_goal_rank import plan_grid_multi_goal

        origin = self.map.grid_origin.numpy()[:2]
        start = tuple(np.rint(self.pose[:2] / 0.05 + origin).astype(int))
        candidates.sort(key=lambda x: -x[0])
        goals = [tuple(np.rint(np.asarray(point[:2]) / 0.05 + origin).astype(int)) for _, point in candidates]
        result = plan_grid_multi_goal(start, goals, navigable=nav, stop_at_first=False)
        reachable = {index for index, ok, _ in result.goal_scores if ok}
        return [point for index, (_, point) in enumerate(candidates) if index in reachable][:limit]

    def plan(self, target_xyz, *, object_target):
        from emet.motion.base_goal_rank import plan_grid_multi_goal
        from emet.motion.navigation_sweep import differential_drive_waypoints, validate_navigation_sweep

        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Finite observed target XYZ required")
        nav, _ = self._navigable()
        origin = self.map.grid_origin.numpy()[:2]
        start = tuple(np.rint(self.pose[:2] / 0.05 + origin).astype(int))
        if object_target:
            cells = np.argwhere(nav)
            xy = (cells - origin) * 0.05
            distance = np.linalg.norm(xy - target[:2], axis=1)
            goals = cells[(distance >= 0.6) & (distance <= 1.2)]
        else:
            goals = [tuple(np.rint(target[:2] / 0.05 + origin).astype(int))]
        result = plan_grid_multi_goal(start, [tuple(g) for g in goals], navigable=nav)
        if not result.success:
            return [], result.reason
        route, current = [], self.pose.copy()
        for cell in result.path_ij[1:]:
            xy = (np.array(cell) - origin) * 0.05
            heading = np.arctan2(xy[1] - current[1], xy[0] - current[0])
            chunk = differential_drive_waypoints(current, np.r_[xy, heading])
            route.extend(chunk)
            current = np.asarray(chunk[-1])
        # A frontier goal coincides with the endpoint: atan2(0, 0) would turn
        # every arrival back toward local +X. Retain the incoming travel yaw;
        # objects instead retain a distinct standoff target to face.
        final_heading = np.arctan2(target[1] - current[1], target[0] - current[0]) if object_target else current[2]
        route.extend(differential_drive_waypoints(current, np.r_[current[:2], final_heading]))
        valid, reason = validate_navigation_sweep(self.space, self.pose, route)
        return (route, "ok") if valid else ([], reason)


class HabitatMotionExecutor:
    """Drive the shared route executor using fresh RGB-D after every primitive."""

    def __init__(self, sensor, navigation, on_frame=lambda frame: None):
        self.sensor, self.navigation, self.on_frame = sensor, navigation, on_frame
        self.primitive_count = 0
        self._command_receipt = {}

    def cancel_navigation(self):
        # Habitat primitives are synchronous; no command remains in flight.
        return True

    def move_base_to(self, goal, *, blocking, world_frame, timeout, **kwargs):
        from emet.motion.navigation_sweep import validate_navigation_sweep

        if not blocking or not world_frame:
            raise ValueError("This adapter requires blocking local-world XYT goals")
        deadline = time.monotonic() + timeout
        goal = np.asarray(goal, dtype=float)
        rotation_only = np.linalg.norm(goal[:2] - base_xyt(self.sensor.last["camera_pose"])[:2]) <= 0.025
        reason = "primitive_budget_exhausted"
        for _ in range(80):
            if self.sensor.last["camera_pose"] is None:
                reason = "tracking_lost"
                break
            current = base_xyt(self.sensor.last["camera_pose"])
            delta = goal[:2] - current[:2]
            distance = np.linalg.norm(delta)
            if rotation_only and distance > 0.025:
                reason = "rotation_position_drift"
                break
            heading = np.arctan2(delta[1], delta[0]) if not rotation_only and distance > 0.025 else goal[2]
            yaw = np.arctan2(np.sin(heading - current[2]), np.cos(heading - current[2]))
            if distance <= 0.025 and abs(yaw) <= np.deg2rad(2):
                self._command_receipt = {"result": {"reason": "ok"}}
                return True
            if time.monotonic() >= deadline:
                reason = "waypoint_timeout"
                break
            if abs(yaw) > np.deg2rad(2):
                action, amount = ("turn_left" if yaw > 0 else "turn_right"), min(5.0, abs(np.rad2deg(yaw)))
            else:
                action, amount = "move_forward", min(0.1, distance)
            predicted = base_xyt(predicted_camera_pose(self.sensor.last["camera_pose"], action, amount))
            valid, rejection = validate_navigation_sweep(self.navigation.space, current, [predicted])
            if not valid:
                reason = rejection
                break
            frame = self.sensor.step(action, amount)
            self.primitive_count += 1
            if frame["camera_pose"] is None:
                reason = "tracking_lost"
                self.on_frame(frame)
                break
            self.navigation.update(frame)
            self.on_frame(frame)
            if frame["collided"]:
                reason = "collision"
                break
        self._command_receipt = {"result": {"reason": reason}}
        return False

    def execute(self, route, *, max_replans=0):
        from emet.motion.navigation_sweep import execute_measured_route

        if not route:
            raise ValueError("A nonempty admitted route is required")
        events = []
        result = execute_measured_route(
            self,
            goal=route[-1],
            measure=lambda: base_xyt(self.sensor.last["camera_pose"]),
            plan_route=lambda current, goal: [],
            space=self.navigation.space,
            max_replans=max_replans,
            position_tolerance_m=0.025,
            yaw_tolerance_rad=np.deg2rad(3),
            waypoint_timeout_s=30,
            initial_route=route,
            event=lambda **event: events.append(event),
        )
        return {**asdict(result), "events": events}
