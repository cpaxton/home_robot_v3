# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# In-process sim: pole ring + rotate + fuse sensor depth → world; geometric circle fit vs ground truth.
# Rerun: ``emet debug-circle-rerun`` (same geometry, streams to Rerun).
#
# Run: uv run emet test src/test/simulation/test_pointcloud_circle_alignment.py -v
# Skip: RUN_SIM_TESTS=0

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from emet.simulation.circle_calibration import (
    POLE_INNER_WALL_RADIUS,
    POLE_RING_RADIUS,
    build_merged_model_with_pole_ring,
    fit_circle_layout_perfect_depth,
    fit_circle_xy_geometric_ls,
    sample_annulus_points_head_camera,
)
from emet.simulation.mujoco_server import _load_default_scene_with_robot

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def test_fit_circle_xy_geometric_ls_noiseless_synthetic():
    """Sanity: LS circle fit recovers ground truth on synthetic samples."""
    rng = np.random.default_rng(0)
    c_true = np.array([0.12, -0.34], dtype=np.float64)
    r_true = 0.58
    n = 500
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    noise = rng.normal(scale=1e-5, size=(n, 2))
    xy = c_true + np.stack([r_true * np.cos(ang), r_true * np.sin(ang)], axis=1) + noise
    c_fit, r_fit, _ = fit_circle_xy_geometric_ls(xy, c_true + 0.02, r_true + 0.01)
    assert np.linalg.norm(c_fit - c_true) < 1e-4
    assert abs(r_fit - r_true) < 1e-4


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(45)
def test_innate_mars_rotate_fused_depth_circle_matches_layout():
    """Sensor depth + poses should match MJCF ring (tight; pole-color + geometric LS)."""
    pytest.importorskip("mujoco")
    import mujoco

    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")

    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    base = _load_default_scene_with_robot("innate_mars")
    if base is None:
        pytest.skip("Merged innate_mars scene not available")

    data0 = mujoco.MjData(base)
    mujoco.mj_forward(base, data0)
    bid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if bid < 0:
        pytest.skip("base_link missing")
    cx0 = float(data0.body(bid).xpos[0])
    cy0 = float(data0.body(bid).xpos[1])

    model = build_merged_model_with_pole_ring(cx=cx0, cy=cy0)

    spec = InnateMarsBackend().get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
        use_remote_computer=False,
    )
    server._load_model()
    server._stabilize_physics_state_after_load()
    server._initial_xyt = server.get_base_xyt().copy()

    jid = mujoco.mj_name2id(server._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")
    assert jid >= 0
    qadr = int(server._mjmodel.jnt_qposadr[jid])

    ann_c = np.array([cx0, cy0], dtype=np.float64)
    fused_xyz: list[np.ndarray] = []
    fused_cols: list[np.ndarray] = []
    rng = np.random.default_rng(42)
    n_yaw_steps = 24
    steps_per_yaw = 12
    for _ in range(n_yaw_steps):
        with server._mj_lock:
            server._mjdata.qpos[qadr] += float(2.0 * np.pi / n_yaw_steps)
            mujoco.mj_forward(server._mjmodel, server._mjdata)
        for _ in range(steps_per_yaw):
            with server._mj_lock:
                mujoco.mj_step(server._mjmodel, server._mjdata)
        xyz, cols = sample_annulus_points_head_camera(
            server,
            stride=4,
            annulus_c=ann_c,
            annulus_r=POLE_RING_RADIUS,
            annulus_half_width=0.22,
            pole_color_only=True,
        )
        if xyz.shape[0] > 0:
            fused_xyz.append(xyz)
            fused_cols.append(cols)

    assert fused_xyz, "expected at least one frame with pole-colored points in the annulus"
    all_pts = np.vstack(fused_xyz)
    all_cols = np.vstack(fused_cols)

    c_fit, r_fit, diag = fit_circle_layout_perfect_depth(
        all_pts[:, :2], all_cols, annulus_c=ann_c, ring_radius=POLE_RING_RADIUS, rng=rng
    )

    err_c = float(np.linalg.norm(c_fit - ann_c))
    err_r_axis = abs(r_fit - POLE_RING_RADIUS)
    err_r_inner = abs(r_fit - POLE_INNER_WALL_RADIUS)

    assert err_c < 0.002, (
        f"center error {err_c:.4f} m vs base_link / ring layout (expected ≪ pole radius); "
        f"c_fit={c_fit}, ann_c={ann_c}, r_fit={r_fit:.4f}, diag={diag}"
    )
    assert err_r_inner < 0.008, (
        f"radius vs inner wall {err_r_inner:.4f} m (pole paint samples the inward-facing wall at "
        f"≈ R − pole_r = {POLE_INNER_WALL_RADIUS:.4f}, not pole axis R = {POLE_RING_RADIUS}); "
        f"r_fit={r_fit:.4f}, |r_fit−R_axis|={err_r_axis:.4f}, diag={diag}"
    )
    assert float(diag["rmse_circle_residual_m"]) < 0.008, (
        f"RMSE to fitted circle {diag['rmse_circle_residual_m']:.4f} m too large; diag={diag}"
    )
    assert int(diag["n_xy_pole_color"]) > 400, f"too few pole-colored points: {diag}"
