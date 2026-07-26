# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline motion planning against synthetic / SparseVoxelMap-shaped obstacle grids.

No ZMQ, no GPU, no pickle maps — exercises the same grid convention as agent voxel nav.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest
import torch

from emet.mapping.grid.grid import GridParams
from emet.motion.algo import get_planner
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.base import ConfigurationSpace
from emet.motion.mujoco_arm_ik import (
    RBY1_LEFT_ARM_JOINTS,
    RBY1_LEFT_EE_BODY,
    interpolate_arm_waypoints,
    joint_qpos_addrs,
)
from emet.motion.voxel_arm_collision import (
    VoxelMapArmCollisionChecker,
    link_samples_collide_2d,
    world_xy_to_grid,
)
from emet.robots.rby1 import Rby1Backend


@dataclass
class FakeSparseVoxelMap:
    """Minimal stand-in for SparseVoxelMap.get_2d_map + GridParams indexing."""

    obstacles: np.ndarray
    resolution: float = 0.05

    def __post_init__(self) -> None:
        h, w = self.obstacles.shape[:2]
        self.explored = np.ones_like(self.obstacles, dtype=bool)
        self.grid_resolution = float(self.resolution)
        self.grid = GridParams((h, w), self.resolution, device=torch.device("cpu"))

    def get_2d_map(self):
        return self.obstacles, self.explored


def _wall_map(*, size: int = 80, res: float = 0.05) -> FakeSparseVoxelMap:
    """Open free space with a vertical wall and a gap at the bottom (SparseVoxelMap indexing)."""
    obs = np.zeros((size, size), dtype=bool)
    # Wall near map center in X (cell index), spanning most of Y, gap at low j
    mid = size // 2
    obs[mid : mid + 2, 8:] = True
    return FakeSparseVoxelMap(obs, resolution=res)


def test_world_xy_to_grid_matches_grid_params():
    grid = GridParams((64, 64), 0.1, device=torch.device("cpu"))
    go = grid.grid_origin[:2].cpu().numpy()
    # World origin maps to grid center
    gi, gj = world_xy_to_grid(0.0, 0.0, grid_origin=go, resolution=0.1, convention="grid_params")
    assert gi == int(go[0])
    assert gj == int(go[1])


def test_from_voxel_map_uses_grid_params_convention():
    fake = _wall_map(size=40, res=0.1)
    checker = VoxelMapArmCollisionChecker.from_voxel_map(fake, link_bodies=["left_arm_link6"], inflate_cells=0)
    assert checker is not None
    assert checker.convention == "grid_params"
    # Free cell left of wall (wall at mid=20)
    free_xy = fake.grid.grid_coords_to_xy(torch.tensor([10.0, 10.0]))
    free_xy = np.asarray(free_xy, dtype=np.float64).reshape(2)
    assert not link_samples_collide_2d(
        checker.obstacles,
        grid_origin=checker.grid_origin,
        resolution=checker.resolution,
        sample_xy=[(float(free_xy[0]), float(free_xy[1]))],
        inflate_cells=0,
        convention="grid_params",
    )
    # Point on the wall
    wall_xy = fake.grid.grid_coords_to_xy(torch.tensor([20.0, 20.0]))
    wall_xy = np.asarray(wall_xy, dtype=np.float64).reshape(2)
    assert link_samples_collide_2d(
        checker.obstacles,
        grid_origin=checker.grid_origin,
        resolution=checker.resolution,
        sample_xy=[(float(wall_xy[0]), float(wall_xy[1]))],
        inflate_cells=0,
        convention="grid_params",
    )


def test_rrt_connect_base_avoids_voxel_wall():
    """Base XY planning on a SparseVoxelMap-shaped obstacle grid (offline)."""
    fake = _wall_map(size=60, res=0.1)
    go = fake.grid.grid_origin[:2].cpu().numpy()
    res = fake.resolution
    obs = fake.obstacles

    def world_free(xy: np.ndarray) -> bool:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1)
        x, y = float(xy[0]), float(xy[1])
        return not link_samples_collide_2d(
            obs,
            grid_origin=go,
            resolution=res,
            sample_xy=[(x, y)],
            inflate_cells=0,
            convention="grid_params",
        )

    # Start left of wall, goal right — both free; midpoint on wall occupied
    start = np.asarray(fake.grid.grid_coords_to_xy(torch.tensor([15.0, 30.0])), dtype=np.float64).reshape(2)
    goal = np.asarray(fake.grid.grid_coords_to_xy(torch.tensor([45.0, 30.0])), dtype=np.float64).reshape(2)
    mid = 0.5 * (start + goal)
    assert world_free(start) and world_free(goal)
    assert not world_free(mid)

    space = ConfigurationSpace(
        2,
        mins=np.array([-3.0, -3.0]),
        maxs=np.array([3.0, 3.0]),
        step_size=0.12,
    )
    np.random.seed(0)
    planner = get_planner("rrt_connect", space, world_free, max_iter=1200, goal_tolerance=0.15)
    result = planner.plan(start, goal, verbose=False)
    assert result.success, "RRT-Connect should route around the voxel wall"
    for node in result.trajectory:
        assert world_free(node.state)


def _sealed_wall_map(*, size: int = 60, res: float = 0.1) -> FakeSparseVoxelMap:
    """Thick vertical wall with no gap — right side unreachable from the left.

    Wall is several meters thick so RRT step_size cannot jump over free cells.
    """
    obs = np.zeros((size, size), dtype=bool)
    mid = size // 2
    obs[mid - 4 : mid + 5, :] = True  # ~0.9 m thick at res=0.1
    return FakeSparseVoxelMap(obs, resolution=res)


def test_multi_frontier_goals_pick_reachable_reject_sealed():
    """Multi-option frontier MP: sealed-wall decoy fails; open-side goal is chosen."""
    from emet.motion.base_goal_rank import choose_first_reachable, rank_xy_goals_by_plan

    fake = _sealed_wall_map(size=60, res=0.1)
    go = fake.grid.grid_origin[:2].cpu().numpy()
    res = fake.resolution
    obs = fake.obstacles
    h, w = obs.shape

    def world_free(xy: np.ndarray) -> bool:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1)
        gi, gj = world_xy_to_grid(float(xy[0]), float(xy[1]), grid_origin=go, resolution=res, convention="grid_params")
        # Out-of-map must be invalid or RRT skirts the wall in unbounded free space.
        if gi < 0 or gj < 0 or gi >= h or gj >= w:
            return False
        return not bool(obs[gi, gj])

    start = np.asarray(fake.grid.grid_coords_to_xy(torch.tensor([15.0, 30.0])), dtype=np.float64).reshape(2)
    # Decoy first: free cell on the far side of the sealed wall (no path).
    decoy = np.asarray(fake.grid.grid_coords_to_xy(torch.tensor([45.0, 30.0])), dtype=np.float64).reshape(2)
    # Reachable: another free cell on the start side.
    good = np.asarray(fake.grid.grid_coords_to_xy(torch.tensor([12.0, 40.0])), dtype=np.float64).reshape(2)
    assert world_free(start) and world_free(decoy) and world_free(good)

    # Bound the planner to the map so it cannot leave the grid.
    corner0 = np.asarray(fake.grid.grid_coords_to_xy(torch.tensor([0.0, 0.0])), dtype=np.float64).reshape(2)
    corner1 = np.asarray(fake.grid.grid_coords_to_xy(torch.tensor([59.0, 59.0])), dtype=np.float64).reshape(2)
    mins = np.minimum(corner0, corner1)
    maxs = np.maximum(corner0, corner1)

    scores = rank_xy_goals_by_plan(
        start,
        [decoy, good],
        is_valid=world_free,
        planner="rrt_connect",
        max_iter=800,
        seed=1,
        bounds=(mins, maxs),
    )
    assert any(i == 0 and not ok for i, ok, _r in scores), scores
    assert any(i == 1 and ok for i, ok, _r in scores), scores
    chosen = choose_first_reachable(scores)
    assert chosen == 1, scores


def test_arm_rrt_avoids_voxel_wall_linear_collides():
    """Joint-space RRT-Connect with VoxelMapArmCollisionChecker: linear hits wall, RRT does not."""
    from emet.motion.mujoco_arm_ik import solve_position_ik

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    joints = RBY1_LEFT_ARM_JOINTS
    qadr = joint_qpos_addrs(model, joints)

    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        data.qpos[int(model.jnt_qposadr[jid])] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RBY1_LEFT_EE_BODY)
    p_seed = np.asarray(data.body(ee).xpos, dtype=np.float64).copy()

    t0 = p_seed + np.array([-0.12, 0.0, 0.05])
    t1 = p_seed + np.array([0.12, 0.0, 0.05])
    r0 = solve_position_ik(
        model, data, ee_body=RBY1_LEFT_EE_BODY, joint_names=joints, target_pos=t0, tol_m=0.03, max_iters=120
    )
    if not r0.success:
        pytest.skip(f"IK to t0 failed err={r0.pos_error_m}")
    q0 = np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)
    p0 = np.asarray(data.body(ee).xpos[:2], dtype=np.float64).copy()

    r1 = solve_position_ik(
        model, data, ee_body=RBY1_LEFT_EE_BODY, joint_names=joints, target_pos=t1, tol_m=0.03, max_iters=120
    )
    if not r1.success:
        pytest.skip(f"IK to t1 failed err={r1.pos_error_m}")
    q1 = np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)
    p1 = np.asarray(data.body(ee).xpos[:2], dtype=np.float64).copy()
    if float(np.linalg.norm(p1 - p0)) < 0.08:
        pytest.skip("EE XY motion too small between IK targets")

    linear = interpolate_arm_waypoints(q0, q1, n_steps=24)
    # Place wall on EE XY of the middle linear waypoint (guarantees linear collides)
    mid_q = linear[len(linear) // 2]
    for a, v in zip(qadr, mid_q, strict=True):
        data.qpos[a] = float(v)
    mujoco.mj_forward(model, data)
    mid_xy = np.asarray(data.body(ee).xpos[:2], dtype=np.float64).copy()

    size, res = 160, 0.05
    fake = FakeSparseVoxelMap(np.zeros((size, size), dtype=bool), resolution=res)
    go = fake.grid.grid_origin[:2].cpu().numpy()
    gi_m, gj_m = world_xy_to_grid(
        float(mid_xy[0]), float(mid_xy[1]), grid_origin=go, resolution=res, convention="grid_params"
    )
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ii, jj = gi_m + di, gj_m + dj
            if 0 <= ii < size and 0 <= jj < size:
                fake.obstacles[ii, jj] = True

    checker = VoxelMapArmCollisionChecker.from_voxel_map(fake, link_bodies=[RBY1_LEFT_EE_BODY], inflate_cells=0)
    assert checker is not None
    # Mid config must collide; endpoints should remain free (farther than wall blob)
    for a, v in zip(qadr, mid_q, strict=True):
        data.qpos[a] = float(v)
    assert checker.configuration_collides(model, data), "wall must cover mid EE"
    for a, v in zip(qadr, q0, strict=True):
        data.qpos[a] = float(v)
    assert not checker.configuration_collides(model, data), "start must stay free"
    for a, v in zip(qadr, q1, strict=True):
        data.qpos[a] = float(v)
    assert not checker.configuration_collides(model, data), "goal must stay free"

    hit = checker.trajectory_collides(model, data, joint_names=joints, arm_waypoints=linear)
    assert hit is not None, "linear path must collide with wall at mid waypoint"

    np.random.seed(2)
    plan = plan_arm_joint_path(
        model,
        data,
        joint_names=joints,
        q_start=q0,
        q_goal=q1,
        collision=checker,
        planner="rrt_connect",
        max_iter=1200,
        linear_fallback=False,
        shortcut=True,
        goal_tolerance=0.08,
    )
    assert plan.success, f"RRT-Connect should avoid voxel wall; got {plan}"
    assert plan.planner == "rrt_connect"
    hit2 = checker.trajectory_collides(model, data, joint_names=joints, arm_waypoints=plan.waypoints)
    assert hit2 is None, f"RRT path still collides at waypoint {hit2}"
