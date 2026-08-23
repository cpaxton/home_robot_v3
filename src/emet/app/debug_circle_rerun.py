# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# In-process pole-ring calibration scene streamed to Rerun (no ZMQ). Same geometry as
# ``src/test/simulation/test_pointcloud_circle_alignment.py``.

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click
import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[3]


@click.command("debug-circle-rerun")
@click.option(
    "--headless",
    is_flag=True,
    help="Do not spawn the Rerun viewer; use rr.serve() and open the web viewer URL printed below.",
)
@click.option("--frames", type=int, default=24, help="Number of yaw steps (full rotation spread across them).")
@click.option("--steps-per-frame", type=int, default=12, help="MuJoCo mj_step calls after each yaw bump.")
@click.option("--stride", type=int, default=5, help="Pixel stride when subsampling depth for point cloud.")
def main(headless: bool, frames: int, steps_per_frame: int, stride: int) -> None:
    """Build innate_mars + pole ring in MuJoCo, spin base yaw, log annulus points to Rerun, print circle fit."""
    import mujoco
    import rerun as rr

    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + os.environ.get("PYTHONPATH", "").split(os.pathsep))

    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation.circle_calibration import (
        POLE_INNER_WALL_RADIUS,
        POLE_RING_RADIUS,
        build_merged_model_with_pole_ring,
        fit_circle_layout_perfect_depth,
        sample_annulus_points_head_camera,
    )
    from emet.simulation.mujoco_server import _load_default_scene_with_robot
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    spawn = not headless
    rr.init("circle_calibration", spawn=spawn)
    if not spawn:
        rr.serve(open_browser=not os.environ.get("RERUN_HEADLESS", "").strip())
        host = os.environ.get("RERUN_SERVE_HOST", "127.0.0.1")
        click.echo(
            f"Rerun web: http://{host}:9090?url=ws://{host}:9877 (set RERUN_HEADLESS=1 to skip opening a browser)"
        )

    base = _load_default_scene_with_robot("innate_mars")
    if base is None:
        raise click.ClickException("Merged innate_mars scene not available (missing assets?).")

    data0 = mujoco.MjData(base)
    mujoco.mj_forward(base, data0)
    bid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if bid < 0:
        raise click.ClickException("base_link not found in merged model.")
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
    if jid < 0:
        raise click.ClickException("base_yaw joint missing.")
    qadr = int(server._mjmodel.jnt_qposadr[jid])

    ann_c = np.array([cx0, cy0], dtype=np.float64)
    fused_xyz: list[np.ndarray] = []
    fused_cols: list[np.ndarray] = []
    rng = np.random.default_rng(42)

    for i in range(frames):
        with server._mj_lock:
            server._mjdata.qpos[qadr] += float(2.0 * np.pi / max(frames, 1))
            mujoco.mj_forward(server._mjmodel, server._mjdata)
        for _ in range(steps_per_frame):
            with server._mj_lock:
                mujoco.mj_step(server._mjmodel, server._mjdata)
        xyz, cols = sample_annulus_points_head_camera(
            server,
            stride=stride,
            annulus_c=ann_c,
            annulus_r=POLE_RING_RADIUS,
            annulus_half_width=0.22,
            pole_color_only=True,
        )
        rr.set_time_sequence("frame", i)
        if xyz.shape[0] > 0:
            rr.log(
                "calibration/annulus",
                rr.Points3D(
                    positions=np.ascontiguousarray(xyz.astype(np.float32)),
                    colors=np.ascontiguousarray(cols),
                    radii=np.full(xyz.shape[0], 0.012, dtype=np.float32),
                ),
            )
            fused_xyz.append(xyz)
            fused_cols.append(cols)
        time.sleep(0.01)

    if not fused_xyz:
        raise click.ClickException("No points in annulus — check camera view of poles or annulus_half_width.")

    all_xyz = np.vstack(fused_xyz)
    all_cols = np.vstack(fused_cols)
    c_fit, r_fit, diag = fit_circle_layout_perfect_depth(
        all_xyz[:, :2], all_cols, annulus_c=ann_c, ring_radius=POLE_RING_RADIUS, rng=rng
    )
    err_c = float(np.linalg.norm(c_fit - ann_c))
    err_r_inner = abs(r_fit - POLE_INNER_WALL_RADIUS)
    summary = (
        f"## Circle calibration (sensor depth, pole-color + geometric LS)\n\n"
        f"- Ground-truth **pole axis** circle center (base XY): ({ann_c[0]:.4f}, {ann_c[1]:.4f})\n"
        f"- Pole axis radius: {POLE_RING_RADIUS:.4f} m | inner wall ≈ {POLE_INNER_WALL_RADIUS:.4f} m\n"
        f"- Fitted center: ({c_fit[0]:.4f}, {c_fit[1]:.4f})\n"
        f"- Fitted radius: {r_fit:.4f} m (expect ≈ inner wall when camera is inside the ring)\n"
        f"- Center error: {err_c:.4f} m | |r_fit − inner wall|: {err_r_inner:.4f} m\n"
        f"- Diagnostics: {diag}\n"
    )
    rr.set_time_sequence("frame", frames)
    rr.log("calibration/summary", rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN))
    click.echo(summary)
    click.echo("Done. Close Rerun or Ctrl+C when finished.")
