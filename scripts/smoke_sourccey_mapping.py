#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Drive Sourccey with the production sim controller and map rendered RGB-D.

Run: MUJOCO_GL=egl uv run python scripts/smoke_sourccey_mapping.py --out /tmp/sourccey-mapping
Uses simulator pose/depth, prescribed waypoints and the DynaMem geometric mapper.
This does not test SLAM, autonomous exploration, or physical collision avoidance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/emet-mpl")
os.environ["EMET_SIM_NAV_TELEPORT"] = "0"

import cv2
import mujoco
import numpy as np
import torch

from emet.mapping.navgrid_compare import world_raster_from_voxel_map
from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap
from emet.robots.sourccey import SourcceyBackend
from emet.simulation.mujoco_server import _load_default_scene_with_robot
from emet.simulation.robosuite_server import RobosuiteZmqServer


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    np.random.seed(0)
    torch.manual_seed(0)
    model = _load_default_scene_with_robot("sourccey")
    server = RobosuiteZmqServer(
        robot_spec=SourcceyBackend().get_spec(),
        scene_model=model,
        navigation_xy_tolerance=0.03,
        navigation_yaw_tolerance=0.04,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
    )
    mapper = SparseVoxelMap(
        log=str(out / "observations"),
        image_shape=None,
        resolution=0.03,
        grid_resolution=0.05,
        grid_size=(400, 400),
        max_depth=4.5,
        device="cpu",
        use_instance_memory=False,
        add_local_radius_points=False,
    )
    trace, frames, goals = [], [], []
    video = None
    chase = None
    try:
        server._load_model()
        server._stabilize_physics_state_after_load()
        server._initial_xyt = server.get_base_xyt().copy()
        server._running = True
        start = server.get_base_xyt().copy()
        assert np.linalg.norm(start[:2]) < 0.01
        video = cv2.VideoWriter(str(out / "drive.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 240))
        assert video.isOpened(), "Could not create smoke-test video"
        chase = mujoco.Renderer(model, height=240, width=320)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.2, 0, 0.5]
        camera.distance, camera.azimuth, camera.elevation = 3.4, 135, -30

        def snapshot(map_frame: bool) -> None:
            pose = server.get_base_xyt().copy()
            rgb, depth, K = server._primary_rgb_and_depth("front_left")
            h, w = depth.shape
            rgb = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_NEAREST)
            K = np.diag([320 / w, 240 / h, 1]) @ K
            valid = np.isfinite(depth) & (depth > 0.25) & (depth < 4.5)
            assert valid.sum() > 200, "No usable front-camera depth"
            chase.update_scene(server._mjdata, camera=camera)
            view = chase.render().copy()
            video.write(cv2.cvtColor(np.concatenate([view, rgb], axis=1), cv2.COLOR_RGB2BGR))
            if not map_frame:
                return
            mapper.process_rgbd_images(
                rgb,
                depth,
                K.astype(np.float32),
                server._camera_pose_world("front_left").astype(np.float32),
                base_xyt=pose,
                full_perception=False,
            )
            obstacles, explored = mapper.get_2d_map()
            frames.append(
                {
                    "pose": pose.tolist(),
                    "explored_cells": int(explored.sum()),
                    "obstacle_cells": int(obstacles.sum()),
                    "valid_depth_pixels": int(valid.sum()),
                }
            )

        snapshot(True)
        first_explored = frames[0]["explored_cells"]
        # Scan all headings, then translate and turn around a clear square in +Y.
        waypoints = [(0, 0, float(yaw)) for yaw in np.linspace(np.pi / 4, 2 * np.pi, 8)]
        waypoints += [(0.8, 0, np.pi / 2), (0.8, 0.8, np.pi), (0, 0.8, -np.pi / 2), (0, 0, -np.pi / 2)]
        for index, waypoint in enumerate(waypoints):
            goal = np.asarray(waypoint)
            before = server.get_base_xyt().copy()
            server.handle_action({"xyt": goal.tolist(), "nav_world": True})
            assert np.linalg.norm(server.get_base_xyt()[:2] - before[:2]) < 0.001, "Unexpected nav teleport"
            reached = False
            for tick in range(500):
                server._step_base_navigation_drive()
                for _ in range(server._mj_substeps_per_tick):
                    server._mj_step_once()
                pose = server.get_base_xyt().copy()
                trace.append(pose.tolist())
                assert np.isfinite(server._mjdata.qacc).all()
                yaw_error = abs(np.arctan2(np.sin(goal[2] - pose[2]), np.cos(goal[2] - pose[2])))
                xy_error = float(np.linalg.norm(goal[:2] - pose[:2]))
                if tick % 5 == 0:
                    snapshot(tick % 20 == 0)
                if xy_error < 0.05 and yaw_error < 0.06:
                    reached = True
                    break
            assert reached, f"Waypoint {index} failed: goal={goal}, pose={pose}"
            snapshot(True)
            goals.append(
                {
                    "goal": goal.tolist(),
                    "actual": pose.tolist(),
                    "xy_error_m": xy_error,
                    "yaw_error_rad": float(yaw_error),
                }
            )
            print(
                f"waypoint {index + 1}/{len(waypoints)}: error={xy_error:.3f}m, mapped={frames[-1]['explored_cells']} cells",
                flush=True,
            )
        chase.close()
        chase = None
        raster = world_raster_from_voxel_map(mapper, (-2.5, 2.5, -2.5, 2.5), resolution_m=0.05)
        xs = -2.5 + (np.arange(raster.explored.shape[1]) + 0.5) * 0.05
        ys = -2.5 + (np.arange(raster.explored.shape[0]) + 0.5) * 0.05
        xx, yy = np.meshgrid(xs, ys)
        table = (abs(xx) < 0.6) & (yy > -1.5) & (yy < -0.5)
        table_hits = int((raster.obstacles & table).sum())
        table_padded = (abs(xx) < 0.75) & (yy > -1.65) & (yy < -0.35)
        table_alignment = float((raster.obstacles & table_padded).sum() / max(1, raster.obstacles.sum()))
        assert table_alignment > 0.95, f"Map obstacle locations do not align with scene: {table_alignment}"
        trace_arr = np.asarray(trace)
        distance = float(np.linalg.norm(np.diff(trace_arr[:, :2], axis=0), axis=1).sum())
        assert distance > 2.8, f"Base did not traverse the loop: {distance}m"
        assert frames[-1]["explored_cells"] > first_explored * 2, "Scan/drive did not grow the map"
        assert table_hits > 20, f"Known table missing from occupancy map: {table_hits} cells"
        assert len(mapper.voxel_pcd._points) > 1000
        report = {
            "status": "passed",
            "scene": "default_table",
            "robot": "sourccey",
            "pose_source": "MuJoCo ground truth",
            "depth_source": "rendered front_left RGB-D",
            "navigation": "production velocity controller, no teleport, prescribed waypoints",
            "mapper": "DynaMem SparseVoxelMap (geometric occupancy, no semantic models)",
            "distance_m": distance,
            "waypoints_reached": len(goals),
            "rgbd_frames": len(frames),
            "first_explored_cells": first_explored,
            "final_explored_cells": frames[-1]["explored_cells"],
            "explored_area_m2": frames[-1]["explored_cells"] * 0.05**2,
            "obstacle_cells": frames[-1]["obstacle_cells"],
            "table_obstacle_cells": table_hits,
            "obstacle_alignment_fraction": table_alignment,
            "voxel_points": len(mapper.voxel_pcd._points),
            "goals": goals,
            "frames": frames,
        }
        np.savez_compressed(
            out / "map.npz",
            explored=raster.explored,
            obstacles=raster.obstacles,
            trajectory=trace_arr,
            resolution_m=0.05,
            origin_xy=[-2.5, -2.5],
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        grid = np.zeros_like(raster.explored, dtype=np.uint8)
        grid[raster.explored] = 1
        grid[raster.obstacles] = 2
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            grid,
            origin="lower",
            extent=(-2.5, 2.5, -2.5, 2.5),
            vmin=0,
            vmax=2,
            cmap=ListedColormap(["#d5d9df", "#fafafa", "#344152"]),
        )
        ax.plot(trace_arr[:, 0], trace_arr[:, 1], color="#ed7624", linewidth=2, label="Driven path")
        ax.scatter([start[0]], [start[1]], color="#008c73", s=70, label="Start / finish")
        ax.add_patch(
            plt.Rectangle((-0.6, -1.5), 1.2, 1, fill=False, edgecolor="#557eb9", linestyle="--", label="Known table")
        )
        ax.set(xlabel="World X (m)", ylabel="World Y (m)", title="Sourccey: RGB-D occupancy map + driven path")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "map.png", dpi=150)
        plt.close(fig)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("goals", "frames")}, indent=2), flush=True)
        return report
    finally:
        if video is not None:
            video.release()
        if chase is not None:
            chase.close()
        server._running = False
        server._close_renderers()
        server.close_zmq_resources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sourccey-mapping"))
    run(parser.parse_args().out)
