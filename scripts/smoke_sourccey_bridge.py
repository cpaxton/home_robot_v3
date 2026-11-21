#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sourccey camera and optional turn smoke through the normal MuJoCo ZMQ bridge."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from emet.app.robot_cli import create_robot_client_from_cli
from emet.config.sim_launch_config import SimLaunchDefaultMujoco
from emet.core.command_client import command_receipt
from emet.robots.sourccey import SOURCCEY_CAMERA_NAMES, SOURCCEY_JOINT_NAMES
from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
from emet.simulation.sim_subprocess import wait_for_sim_tcp_port
from emet.utils.process_tree import popen_session, terminate_process_tree


def run(output, offset, *, table_view=False):
    output.mkdir(parents=True, exist_ok=False)
    # The simulator CLI can recycle occupied ports. Require dedicated free ports.
    for port in range(4401 + offset, 4405 + offset):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    config = SimLaunchDefaultMujoco(robot="sourccey", headless=True, port_offset=offset, seed=0)
    env = {
        **os.environ,
        "EMET_SIM_NAV_TELEPORT": "0",
        "EMET_MOLMOSPACES_NAV_TELEPORT": "0",
        "EMET_PHYSICAL_EXECUTION": "1",
        "EMET_PHYSICAL_AUDIT": str(output.resolve() / "actuation.jsonl"),
    }
    robot = server = None
    try:
        with (output / "server.log").open("w") as log:
            server = popen_session(
                [
                    sys.executable,
                    "-m",
                    "emet.simulation.mujoco_server",
                    *prepare_mujoco_server_argv(config),
                    "--use_remote_computer",
                    "false",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            wait_for_sim_tcp_port("127.0.0.1", 4401 + offset, proc=server, timeout_sec=60)
            robot = create_robot_client_from_cli(
                "sourccey",
                "127.0.0.1",
                port_offset=offset,
                enable_rerun_server=False,
                start_immediately=True,
                allow_missing_depth=False,
            )
            session = robot.get_emet_session()
            assert robot.peek_emet_robot_id() == "sourccey" and session["is_simulation"] is True, session
            deadline = time.monotonic() + 20.0
            stamps = []
            while time.monotonic() < deadline:
                obs = robot.get_observation()
                with robot._obs_lock:
                    stamp = (robot._obs or {}).get("image_sim_time")
                if obs is not None and isinstance(stamp, (int, float)) and (not stamps or stamp > stamps[-1]):
                    stamps.append(stamp)
                    if len(stamps) >= 3 and obs.head_rgb_right is not None:
                        break
                time.sleep(0.05)
            assert len(stamps) >= 3 and obs is not None and obs.head_rgb_right is not None, stamps
            assert obs.rgb.shape[:2] == obs.head_rgb_right.shape[:2] == obs.depth.shape
            assert obs.camera_K.shape == (3, 3) and np.isfinite(obs.camera_K).all()
            assert obs.camera_pose.shape == (4, 4) and np.isfinite(obs.camera_pose).all()
            assert len(obs.joint) == len(SOURCCEY_JOINT_NAMES) and np.isfinite(obs.joint).all()
            valid = np.isfinite(obs.depth) & (obs.depth > 0.1) & (obs.depth < 8.0)
            Image.fromarray(obs.rgb).save(output / "front-left.png")
            Image.fromarray(obs.head_rgb_right).save(output / "front-right.png")
            np.savez_compressed(
                output / "first-depth.npz", depth=obs.depth, camera_K=obs.camera_K, camera_pose=obs.camera_pose
            )
            result = {
                "scope": "live Sourccey simulation bridge with simulator localization",
                "configured_model_cameras": SOURCCEY_CAMERA_NAMES,
                "verified_streams": ["front_left RGB-D", "front_right RGB", "joint positions"],
                "rgb_shape": list(obs.rgb.shape),
                "valid_depth_pixels": int(valid.sum()),
                "rgb_spatial_std": [
                    float(np.std(image, axis=(0, 1)).mean()) for image in (obs.rgb, obs.head_rgb_right)
                ],
                "depth_range_m": [float(obs.depth.min()), float(obs.depth.max())],
                "joint_count": len(obs.joint),
                "advancing_capture_sim_times": stamps,
                "depth_source": "rendered metric depth",
                "pose_source": "simulator pose",
            }
            (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
            assert result["valid_depth_pixels"] > 200, result
            if table_view:
                initial = np.asarray(robot.get_base_pose(), dtype=float)
                # Absolute episode goal exercises the planar velocity controller.
                goal = [*initial[:2], initial[2] - np.pi / 2]
                assert robot.move_base_to(
                    goal,
                    blocking=True,
                    timeout=25.0,
                    navigation_policy="exploration",
                )
                receipt = command_receipt(robot, robot._last_navigation_command)
                pose = np.asarray(robot.get_base_pose(), dtype=float)
                view = robot.get_observation()
                yaw = float(np.rad2deg(pose[2] - initial[2]))
                contrast = float(np.std(view.rgb, axis=(0, 1)).mean())
                Image.fromarray(view.rgb).save(output / "table-front-left.png")
                result["table_turn"] = {
                    "measured_yaw_deg": yaw,
                    "translation_m": float(np.linalg.norm(pose[:2] - initial[:2])),
                    "rgb_spatial_std": contrast,
                    "receipt": receipt,
                }
                (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
                assert receipt["result"]["motion_mode"] == "velocity_drive" and abs(yaw + 90) < 9, receipt
                assert contrast > 3, result
            return result
    finally:
        try:
            if robot is not None:
                robot.stop()
        finally:
            terminate_process_tree(server, grace_s=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-offset", type=int, default=2020)
    parser.add_argument("--table-view", action="store_true", help="Turn the simulated base toward the table.")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.port_offset, table_view=args.table_view), indent=2))
