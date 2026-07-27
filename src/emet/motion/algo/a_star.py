# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import heapq
import math
import time
from collections.abc import Callable

import numpy as np

from emet.motion import ConfigurationSpace, Planner, PlanResult
from emet.motion import Node as BaseNode
from emet.motion.algo.node import TreeNode as Node

# Soft fallback when skfmm has no zero contour (empty obstacle set in explored).
_DEFAULT_CLEARANCE_M = 10.0


def neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    return [(pt[0] + dx, pt[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]


def unwrap_yaw(prev: float, target: float) -> float:
    """Return ``target`` unwrapped so the turn from ``prev`` is in ``(-pi, pi]``."""
    delta = float(np.arctan2(np.sin(target - prev), np.cos(target - prev)))
    return float(prev + delta)


def default_min_clearance_m(footprint_width_m: float, *, margin_m: float = 0.05) -> float:
    """Half footprint width plus a small margin (Stretch width 0.34 → ~0.22 m)."""
    return 0.5 * float(footprint_width_m) + float(margin_m)


class AStar(Planner):
    """Define A* motion planning problem and parameters"""

    def __init__(
        self,
        space: ConfigurationSpace,
        validate_fn: Callable = None,
        *,
        min_clearance_m: float | None = 0.22,
        clearance_cost_weight: float = 1.0,
        grid_resolution_m: float | None = None,
    ):
        """Create A* planner with configuration.

        Args:
            min_clearance_m: Hard-reject cells closer than this to obstacles (meters).
                ``None`` or ``<= 0`` disables the hard gate (soft cost may still apply).
            clearance_cost_weight: Soft cost ``weight / clearance_m`` so paths prefer open space.
            grid_resolution_m: Meters per grid cell for EDT. Inferred from the voxel map when omitted.
        """
        if validate_fn is None:
            validate_fn = space.is_valid
        super().__init__(space, validate_fn)
        self.min_clearance_m = float(min_clearance_m) if min_clearance_m is not None else 0.0
        self.clearance_cost_weight = float(clearance_cost_weight)
        self._grid_resolution_override = grid_resolution_m
        self._clearance_m: np.ndarray | None = None
        self._navigable: np.ndarray | None = None
        self.reset()
        if validate_fn is not None:
            self.validate = validate_fn  # type:ignore
        else:
            self.validate = self.space.is_valid  # type:ignore

    def _grid_resolution(self) -> float:
        if self._grid_resolution_override is not None:
            return float(self._grid_resolution_override)
        vm = getattr(self.space, "voxel_map", None)
        if vm is not None:
            res = getattr(vm, "grid_resolution", None)
            if res is not None:
                return float(res)
        return 0.05

    def compute_theta(self, cur_x, cur_y, end_x, end_y):
        """Heading of the segment from (cur) to (end), in ``(-pi, pi]``."""
        return float(np.arctan2(float(end_y) - float(cur_y), float(end_x) - float(cur_x)))

    def reset(self):
        obs, exp = self.space.voxel_map.get_2d_map()
        if hasattr(obs, "cpu"):
            obs = obs.cpu().numpy()
        if hasattr(exp, "cpu"):
            exp = exp.cpu().numpy()
        obs = np.asarray(obs, dtype=bool)
        exp = np.asarray(exp, dtype=bool)
        self._navigable = (~obs) & exp
        self._clearance_m = self._build_clearance_field(obs, exp)
        if self.min_clearance_m > 0 and self._clearance_m is not None:
            # Hard gate: treat low-clearance free cells as non-navigable for search.
            self._navigable = self._navigable & (self._clearance_m >= self.min_clearance_m)
        self.start_time = time.time()

    def _build_clearance_field(self, obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
        """EDT clearance in meters from obstacles within explored free space."""
        h, w = obs.shape
        clearance = np.full((h, w), _DEFAULT_CLEARANCE_M, dtype=np.float64)
        if not np.any(exp):
            return clearance
        # Zero contour at obstacles; distance grows into free explored cells.
        phi = np.ones((h, w), dtype=np.float64)
        phi[obs] = 0.0
        # Mask anything we should not plan through (unexplored).
        masked = np.ma.masked_array(phi, mask=~exp)
        try:
            import skfmm

            if not np.any(obs & exp):
                # No obstacle contour in explored → leave large clearance on free cells.
                clearance[exp & ~obs] = _DEFAULT_CLEARANCE_M
                clearance[obs] = 0.0
                return clearance
            # skfmm needs a zero contour present in the unmasked region.
            dist_cells = skfmm.distance(masked, dx=1)
            res = self._grid_resolution()
            if np.ma.isMaskedArray(dist_cells):
                filled = np.ma.filled(dist_cells, _DEFAULT_CLEARANCE_M / max(res, 1e-6))
            else:
                filled = np.asarray(dist_cells, dtype=np.float64)
            clearance = filled * res
            clearance[obs] = 0.0
            clearance[~exp] = 0.0
        except Exception:
            # Fallback: binary free vs obs only (no soft preference).
            clearance[exp & ~obs] = _DEFAULT_CLEARANCE_M
            clearance[obs] = 0.0
            clearance[~exp] = 0.0
        return clearance

    def clearance_at_xy(self, xy: tuple[float, float] | list[float] | np.ndarray) -> float:
        """Clearance in meters at a world XY (0 if out of map / unexplored)."""
        if self._clearance_m is None:
            self.reset()
        pt = self.to_pt((float(xy[0]), float(xy[1])))
        return self.clearance_at_pt(pt)

    def clearance_at_pt(self, pt: tuple[int, int]) -> float:
        if self._clearance_m is None:
            return _DEFAULT_CLEARANCE_M
        i, j = int(pt[0]), int(pt[1])
        h, w = self._clearance_m.shape
        if not (0 <= i < h and 0 <= j < w):
            return 0.0
        return float(self._clearance_m[i, j])

    def is_explored_xy(self, xy: tuple[float, float] | list[float] | np.ndarray) -> bool:
        obs, exp = self.space.voxel_map.get_2d_map()
        if hasattr(exp, "cpu"):
            exp = exp.cpu().numpy()
        exp = np.asarray(exp, dtype=bool)
        pt = self.to_pt((float(xy[0]), float(xy[1])))
        i, j = int(pt[0]), int(pt[1])
        h, w = exp.shape
        if not (0 <= i < h and 0 <= j < w):
            return False
        return bool(exp[i, j])

    def point_is_occupied(self, x: int, y: int) -> bool:
        """Checks if a point is occupied (obstacle, unexplored, or below min clearance).

        Args:
            x: The x coordinate.
            y: The y coordinate.

        Returns:
            Whether the point is occupied.
        """
        h, w = self._navigable.shape
        if not (0 <= x < h and 0 <= y < w):
            return True
        return not bool(self._navigable[x][y])

    def to_pt(self, xy: tuple[float, float]) -> tuple[int, int]:
        """Converts a point from continuous, world xy coordinates to grid coordinates.

        Args:
            xy: The point in continuous xy coordinates.

        Returns:
            The point in discrete grid coordinates.
        """
        # # type: ignore to bypass mypy checking
        return self.space.to_pt(xy)

    def to_xy(self, pt: tuple[int, int]) -> tuple[float, float]:
        """Converts a point from grid coordinates to continuous, world xy coordinates.

        Args:
            pt: The point in grid coordinates.

        Returns:
            The point in continuous xy coordinates.
        """
        # # type: ignore to bypass mypy checking
        return self.space.to_xy(pt)

    def compute_dis(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Compute distance between two points a and b.

        Args:
            a: The first point.
            b: The second point.

        Returns:
            The distance between the two points.
        """
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _clearance_soft_cost(self, pt: tuple[int, int]) -> float:
        if self.clearance_cost_weight <= 0 or self._clearance_m is None:
            return 0.0
        c = self.clearance_at_pt(pt)
        return self.clearance_cost_weight / max(c, 1e-3)

    def compute_obstacle_punishment(self, a: tuple[int, int], weight: int, avoid: int) -> float:
        """Legacy local 3x3 obstacle penalty (kept for tests / fallbacks). Prefer clearance field."""
        obstacle_punishment = 0
        for i in range(-avoid, avoid + 1):
            for j in range(-avoid, avoid + 1):
                if self.point_is_occupied(a[0] + i, a[1] + j):
                    b = [a[0] + i, a[1] + j]
                    obs_dis = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                    obstacle_punishment = max((weight / max(obs_dis, 1)), obstacle_punishment)
        return obstacle_punishment

    def compute_heuristic(self, a: tuple[int, int], b: tuple[int, int], weight=6, avoid=3) -> float:
        # Soft clearance cost on both ends; hard gate already in point_is_occupied.
        return self.compute_dis(a, b) + self._clearance_soft_cost(a) + self._clearance_soft_cost(b)

    def step_cost(self, current: tuple[int, int], nxt: tuple[int, int]) -> float:
        return self.compute_dis(current, nxt) + self._clearance_soft_cost(nxt)

    def is_in_line_of_sight(self, start_pt: tuple[int, int], end_pt: tuple[int, int]) -> bool:
        """Checks if there is a line-of-sight between two points.

        Implements using Bresenham's line algorithm.

        Args:
            start_pt: The starting point.
            end_pt: The ending point.

        Returns:
            Whether there is a line-of-sight between the two points.
        """

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) > abs(dy):
            if dx < 0:
                start_pt, end_pt = end_pt, start_pt
            for x in range(start_pt[0], end_pt[0] + 1):
                yf = start_pt[1] + (x - start_pt[0]) / dx * dy
                for y in list({math.floor(yf), math.ceil(yf)}):
                    if self.point_is_occupied(x, y):
                        return False

        else:
            if dy < 0:
                start_pt, end_pt = end_pt, start_pt
            for y in range(start_pt[1], end_pt[1] + 1):
                xf = start_pt[0] + (y - start_pt[1]) / dy * dx
                for x in list({math.floor(xf), math.ceil(xf)}):
                    if self.point_is_occupied(x, y):
                        return False

        return True

    def is_a_line(self, a, b, c):
        if a[0] == b[0]:
            return c[0] == a[0]
        if b[0] == c[0]:
            return False
        return ((c[1] - b[1]) / (c[0] - b[0])) == ((b[1] - a[1]) / (b[0] - a[0]))

    def clean_path_for_xy(self, waypoints, start_yaw: float | None = None):
        """Simplify path and assign continuous yaw via shortest-turn unwrap from start heading."""
        goal = waypoints[-1]
        if start_yaw is None:
            g0 = np.asarray(waypoints[0], dtype=np.float64).reshape(-1)
            start_yaw = float(g0[2]) if g0.size >= 3 and np.isfinite(g0[2]) else 0.0
        waypts = [self.to_pt(waypoint) for waypoint in waypoints]
        waypts = self.clean_path(waypts)
        waypoints_xy = [self.to_xy(waypt) for waypt in waypts]
        traj = []
        prev_yaw = float(start_yaw)
        for i in range(len(waypoints_xy) - 1):
            theta = self.compute_theta(
                waypoints_xy[i][0], waypoints_xy[i][1], waypoints_xy[i + 1][0], waypoints_xy[i + 1][1]
            )
            theta = unwrap_yaw(prev_yaw, theta)
            traj.append([waypoints_xy[i][0], waypoints_xy[i][1], float(theta)])
            prev_yaw = theta
        goal_arr = np.asarray(goal, dtype=np.float64).reshape(-1)
        goal_yaw = float(goal_arr[2]) if goal_arr.size >= 3 and np.isfinite(goal_arr[2]) else prev_yaw
        goal_yaw = unwrap_yaw(prev_yaw, goal_yaw)
        traj.append([waypoints_xy[-1][0], waypoints_xy[-1][1], float(goal_yaw)])
        return traj

    def clean_path(self, path) -> list[tuple[int, int]]:
        """Cleans up the final path.

        This implements a simple algorithm where, given some current position,
        we find the last point in the path that is in line-of-sight, and then
        we set the current position to that point. This is repeated until we
        reach the end of the path. This is not particularly efficient, but
        it's simple and works well enough.

        Args:
            path: The path to clean up.

        Returns:
            The cleaned up path.
        """
        cleaned_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if self.is_in_line_of_sight(path[i][:2], path[j][:2]):
                    break
            else:
                j = i + 1
            # Include the mid waypoint to avoid the collision
            if j - i >= 2 and self.point_is_occupied((path[i][0] + path[j][0]) // 2, (path[i][1] + path[j][1]) // 2):
                cleaned_path.append(path[(i + j) // 2])
            cleaned_path.append(path[j])
            i = j
        return cleaned_path

    def get_unoccupied_neighbor(self, pt: tuple[int, int], goal_pt=None, max_ring: int = 4) -> tuple[int, int] | None:
        if not self.point_is_occupied(*pt):
            return pt

        # If the start cell is marked occupied (pose noise / dilation), search outward by Chebyshev ring.
        h, w = self._navigable.shape
        for ring in range(1, max_ring + 1):
            ring_pts: list[tuple[int, int]] = []
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    if max(abs(di), abs(dj)) != ring:
                        continue
                    ni, nj = pt[0] + di, pt[1] + dj
                    if not (0 <= ni < h and 0 <= nj < w):
                        continue
                    if not self.point_is_occupied(ni, nj):
                        ring_pts.append((ni, nj))
            if goal_pt is not None and ring_pts:
                ring_pts.sort(key=lambda n: self.compute_heuristic(n, goal_pt))
            for neighbor_pt in ring_pts:
                return neighbor_pt
        print("The robot might stand on a non navigable point, so check obstacle map and restart roslaunch")
        return None
        # raise ValueError("The robot might stand on a non navigable point, so check obstacle map and restart roslaunch")

    def get_reachable_points(self, start_pt: tuple[int, int]) -> set[tuple[int, int]]:
        """Gets all reachable points from a given starting point.

        Args:
            start_pt: The starting point

        Returns:
            The set of all reachable points
        """

        self.reset()
        # Obstacle dilation + pose noise often marks the robot cell occupied;
        # search farther than the default 4-cell ring before giving up.
        start_pt = self.get_unoccupied_neighbor(start_pt, max_ring=8)
        if start_pt is None:
            return set()

        reachable_points: set[tuple[int, int]] = set()
        to_visit = [start_pt]
        while to_visit:
            pt = to_visit.pop()
            if pt in reachable_points:
                continue
            reachable_points.add(pt)
            for new_pt in neighbors(pt):
                if new_pt in reachable_points:
                    continue
                if self.point_is_occupied(new_pt[0], new_pt[1]):
                    continue
                to_visit.append(new_pt)
        return reachable_points

    def run_astar(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        remove_line_of_sight_points: bool = False,
    ) -> list[tuple[float, float]]:

        start_pt, end_pt = self.to_pt(start_xy), self.to_pt(end_xy)

        # Checks that both points are unoccupied.
        start_pt = self.get_unoccupied_neighbor(start_pt)
        end_pt = self.get_unoccupied_neighbor(end_pt, start_pt)
        # print('A* formally starts ', time.time() - self.start_time, ' seconds after path planning starts')
        if start_pt is None or end_pt is None:
            return None

        # Implements A* search.
        q = [(0, start_pt)]
        came_from: dict = {start_pt: None}
        cost_so_far: dict[tuple[int, int], float] = {start_pt: 0.0}
        while q:
            _, current = heapq.heappop(q)

            if current == end_pt:
                break

            for nxt in neighbors(current):
                if self.point_is_occupied(nxt[0], nxt[1]):
                    continue
                new_cost = cost_so_far[current] + self.step_cost(current, nxt)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.compute_dis(end_pt, nxt)
                    heapq.heappush(q, (priority, nxt))  # type: ignore
                    came_from[nxt] = current

        else:
            return None

        # Reconstructs the path.
        path = []
        current = end_pt
        while current != start_pt:
            path.append(current)
            prev = came_from[current]
            if prev is None:
                break
            current = prev
        path.append(start_pt)
        path.reverse()

        # Clean up the path.
        if remove_line_of_sight_points:
            path = self.clean_path(path)

        # return [start_xy] + [self.to_xy(pt) for pt in path[1:-1]] + [end_xy]
        return [start_xy] + [self.to_xy(pt) for pt in path[1:]]

    def run_astar_multi_goal(
        self,
        start_xy: tuple[float, float],
        goals_xy: list[tuple[float, float]],
        *,
        stop_at_first: bool = True,
    ):
        """One shared A*/Dijkstra search toward a set of goal XYs.

        Returns ``(waypoints_xy, goal_index)`` or ``(None, None)`` if no goal is reachable.
        """
        from emet.motion.base_goal_rank import plan_grid_multi_goal

        start_pt = self.get_unoccupied_neighbor(self.to_pt(start_xy), max_ring=8)
        if start_pt is None:
            return None, None

        goal_ijs: list[tuple[int, int] | None] = []
        snapped_goals: list[tuple[float, float]] = []
        for gxy in goals_xy:
            gpt = self.get_unoccupied_neighbor(self.to_pt(gxy), start_pt)
            goal_ijs.append(gpt)
            snapped_goals.append(gxy)

        result = plan_grid_multi_goal(
            start_pt,
            goal_ijs,
            navigable=self._navigable,
            stop_at_first=stop_at_first,
        )
        if not result.success or result.goal_index is None:
            return None, None

        path_xy = [start_xy] + [self.to_xy(pt) for pt in result.path_ij[1:-1]]
        gi = int(result.goal_index)
        path_xy.append(snapped_goals[gi])
        return path_xy, gi

    def plan(self, start, goal, verbose: bool = True, goals=None, **kwargs) -> PlanResult:
        """Plan from start to ``goal``, or to the nearest of ``goals`` (multi-goal).

        When ``goals`` is a non-empty sequence of XY(T) states, runs one shared grid
        search and returns a trajectory to the nearest reachable goal (final theta from
        that goal if provided).
        """
        self.reset()
        chosen_goal = goal
        chosen_index: int | None = None
        if goals is not None:
            goal_list = list(goals)
            if not goal_list:
                return PlanResult(False, reason="no_goals")
            goals_xy = [(float(g[0]), float(g[1])) for g in goal_list]
            waypoints, gi = self.run_astar_multi_goal(start[:2], goals_xy, stop_at_first=True)
            if waypoints is None or gi is None:
                if verbose:
                    print("A* multi-goal fails, check obstacle map")
                return PlanResult(False, reason="A* multi-goal fails, check obstacle map")
            chosen_index = int(gi)
            chosen_goal = goal_list[chosen_index]
        else:
            waypoints = self.run_astar(start[:2], goal[:2])

        if waypoints is None:
            if verbose:
                print("A* fails, check obstacle map")
            return PlanResult(False, reason="A* fails, check obstacle map")
        trajectory: list[BaseNode] = []
        start_yaw = float(start[2]) if len(start) > 2 else 0.0
        prev_yaw = start_yaw
        for i in range(len(waypoints) - 1):
            theta = self.compute_theta(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])
            theta = unwrap_yaw(prev_yaw, theta)
            if i > 0:
                parent = trajectory[-1]
            else:
                parent = None
            trajectory.append(Node(np.array([waypoints[i][0], waypoints[i][1], float(theta)]), parent=parent))
            prev_yaw = theta
        if len(trajectory) <= 0:
            parent = None
        else:
            parent = trajectory[-1]
        goal_yaw = float(chosen_goal[-1]) if len(chosen_goal) > 2 else prev_yaw
        goal_yaw = unwrap_yaw(prev_yaw, goal_yaw)
        trajectory.append(Node(np.array([waypoints[-1][0], waypoints[-1][1], float(goal_yaw)]), parent=parent))

        # Save the nodes for this planner
        self.nodes = trajectory

        return PlanResult(True, trajectory=trajectory, goal_index=chosen_index)
