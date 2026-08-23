# Copyright (c) Chris Paxton 2026

"""Rotating in place must grow the voxel map, never shrink it.

``clear_points`` and ``voxel_pcd.add`` are two halves of the same refresh pass.
Gating only the add (the old rotate-in-place obstacle guard) turned every spin into
a map eraser: HM-EQA q104 lost ~85% of its explored floor during the opening scan.

Free-space carving must also be conservative: a single truncated depth pixel at a
discontinuity is not evidence that static floor geometry moved.
"""

import tempfile

import numpy as np
import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap
from emet.utils.voxel import VoxelizedPointcloud

H, W = 120, 160
K = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float32)
K_T = torch.tensor(K, dtype=torch.float32)


def _pose_at(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    base_r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cam_r = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = base_r @ cam_r
    pose[:3, 3] = [x, y, 1.0]
    return pose.astype(np.float32)


def _opencv_cam_pose() -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return pose


def _n_points(vm: SparseVoxelMap) -> int:
    pts = vm.voxel_pcd._points
    return 0 if pts is None else int(pts.shape[0])


def test_spin_in_place_grows_voxel_map():
    vm = SparseVoxelMap(log=tempfile.mkdtemp(), image_shape=None, use_instance_memory=False)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 1.5, dtype=np.float32)

    counts = []
    for yaw in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
        vm.process_rgbd_images(rgb, depth, K, _pose_at(0.0, 0.0, float(yaw)), base_xyt=np.array([0.0, 0.0, float(yaw)]))
        counts.append(_n_points(vm))

    assert counts[0] > 0
    assert counts == sorted(counts), f"spin erased mapped points: {counts}"
    # A full 360 deg scan with a ~74 deg FOV must see far more than the first frame.
    assert counts[-1] > 3 * counts[0], f"spin added almost nothing: {counts}"


def test_clear_points_keeps_geometry_where_depth_has_no_return():
    """Mesh holes / sky / past-range pixels are misses, not evidence the geometry is gone."""
    pts = torch.tensor([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)
    pose = _opencv_cam_pose()

    for bad in (0.0, float("nan"), float("inf"), 99.0):
        vox_case = VoxelizedPointcloud(voxel_size=0.1)
        vox_case.add(pts, features=None, rgb=torch.zeros((3, 3), dtype=torch.float32), min_weight_per_voxel=1)
        depth = torch.full((H, W), bad, dtype=torch.float32)
        vox_case.clear_points(depth, K_T, pose, max_depth=4.5)
        assert vox_case._points is not None and vox_case._points.shape[0] == 3, f"depth={bad} erased points"


def test_clear_points_still_removes_geometry_that_moved_away():
    """A valid reading past a stored point is real evidence, so dynamic removal still works."""
    vox = VoxelizedPointcloud(voxel_size=0.1)
    pts = torch.tensor([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)
    vox.add(pts, features=None, rgb=torch.zeros((3, 3), dtype=torch.float32), min_weight_per_voxel=1)

    pose = _opencv_cam_pose()
    depth = torch.full((H, W), 3.0, dtype=torch.float32)

    vox.clear_points(depth, K_T, pose, max_depth=4.5)
    assert vox._points is None or vox._points.shape[0] == 0


def test_clear_points_ignores_single_far_pixel_at_discontinuity():
    """``.int()`` truncation onto one far pixel must not carve static geometry."""
    vox = VoxelizedPointcloud(voxel_size=0.05)
    # Points along optical axis at ~1.5 m; project near image center.
    pts = torch.tensor(
        [
            [1.50, 0.00, 0.00],
            [1.50, 0.02, 0.00],
            [1.50, -0.02, 0.00],
            [1.50, 0.00, 0.02],
            [1.50, 0.00, -0.02],
        ],
        dtype=torch.float32,
    )
    vox.add(pts, features=None, rgb=torch.zeros((pts.shape[0], 3), dtype=torch.float32), min_weight_per_voxel=1)

    pose = _opencv_cam_pose()
    depth = torch.full((H, W), 1.5, dtype=torch.float32)
    # One pixel reads "empty" farther away — classic floor-edge / truncation artifact.
    depth[H // 2, W // 2] = 3.5

    before = int(vox._points.shape[0])
    vox.clear_points(depth, K_T, pose, max_depth=4.5)
    assert vox._points is not None
    assert int(vox._points.shape[0]) == before, "discontinuity pixel carved static points"


def test_clear_points_respects_depth_is_valid_mask():
    """Pixels rejected for insertion must not count as free-space evidence."""
    vox = VoxelizedPointcloud(voxel_size=0.1)
    pts = torch.tensor([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)
    vox.add(pts, features=None, rgb=torch.zeros((3, 3), dtype=torch.float32), min_weight_per_voxel=1)

    pose = _opencv_cam_pose()
    depth = torch.full((H, W), 3.0, dtype=torch.float32)
    valid = torch.zeros((H, W), dtype=torch.bool)  # nowhere trusted

    vox.clear_points(depth, K_T, pose, depth_is_valid=valid, max_depth=4.5)
    assert vox._points is not None and vox._points.shape[0] == 3


def test_static_floor_survives_filtered_spin():
    """With Habitat-like median/derivative filters, a static floor must not shrink on spin."""
    vm = SparseVoxelMap(
        log=tempfile.mkdtemp(),
        image_shape=None,
        use_instance_memory=False,
        use_median_filter=True,
        median_filter_size=5,
        median_filter_max_error=0.01,
        use_derivative_filter=True,
        derivative_filter_threshold=0.2,
        max_depth=4.5,
        min_depth=0.25,
        point_update_threshold=0.0,  # keep all inserted points so clear is the only shrink path
    )
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # Smooth floor plus a hard depth step (discontinuity) that used to false-carve.
    depth = np.full((H, W), 1.5, dtype=np.float32)
    depth[:, W // 2 :] = 2.2

    areas = []
    for yaw in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
        vm.process_rgbd_images(rgb, depth, K, _pose_at(0.0, 0.0, float(yaw)), base_xyt=np.array([0.0, 0.0, float(yaw)]))
        _, explored = vm.get_2d_map()
        areas.append(int(explored.sum().item()))

    assert areas[0] > 0
    # Explored cells from a static mesh must be monotone non-decreasing across the spin.
    assert areas == sorted(areas), f"static floor explored shrank during spin: {areas}"
