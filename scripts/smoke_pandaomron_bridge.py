#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Exercise PandaOmron's production RoboCasa server and shared robot client."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from emet.app.robot_cli import create_robot_client_from_cli
from emet.config.sim_launch_config import load_sim_launch_config_from_path
from emet.core.command_client import command_receipt
from emet.robots.pandaomron import CONTROL_JOINTS
from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
from emet.simulation.sim_subprocess import wait_for_sim_tcp_port
from emet.utils.process_tree import popen_session, terminate_process_tree


def wait_for(predicate, description, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value is not None:
            return value
        time.sleep(0.05)
    raise TimeoutError(description)


def advance_observation(robot, previous_time):
    with robot._obs_lock:
        packet = robot._obs
        stamp = None if packet is None else packet.get("image_sim_time")
    if stamp is not None and stamp > previous_time:
        obs = robot.get_observation()
        if obs is not None and obs.ee_rgb is not None:
            return stamp, obs
    return None


def check_motion(robot, goal, *, open_gripper):
    action = robot.send_action({"joint": goal.tolist(), "gripper": 0.05 if open_gripper else -0.1})
    receipt = command_receipt(robot, action)
    if receipt is None or receipt["status"] != "succeeded":
        raise RuntimeError(f"native goal receipt was not accepted: {receipt}")

    def settled():
        measured = robot.get_joint_positions(timeout=0.3)
        if measured is None or np.max(np.abs(measured[: len(goal)] - goal)) > 0.006:
            return None
        jaw = measured[-2] - measured[-1]
        if (open_gripper and jaw > 0.06) or (not open_gripper and jaw < 0.018):
            return measured
        return None

    return receipt, wait_for(settled, "native joint or gripper target did not settle")


def run(output, offset, *, sim_pythonpath=None):
    output.mkdir(parents=True, exist_ok=False)
    for port in range(4401 + offset, 4405 + offset):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    client = None
    server_env = {
        **os.environ,
        "EMET_PHYSICAL_EXECUTION": "1",
        "EMET_PHYSICAL_AUDIT": str(output.resolve() / "actuation.jsonl"),
    }
    if sim_pythonpath is not None:
        server_env["PYTHONPATH"] = sim_pythonpath
    profile = Path(__file__).resolve().parents[1] / "configs/sim/robocasa_sink_pandaomron.yaml"
    config = replace(load_sim_launch_config_from_path(str(profile)), port_offset=offset)
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
            env=server_env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_sim_tcp_port("127.0.0.1", 4401 + offset, proc=server, timeout_sec=60)
            client = create_robot_client_from_cli(
                "pandaomron",
                "127.0.0.1",
                port_offset=offset,
                zmq_startup_timeout=20,
            )
            first_time, _ = wait_for(lambda: advance_observation(client, -1), "missing first RGB-D and wrist frame")
            second_time, obs = wait_for(
                lambda: advance_observation(client, first_time), "image simulator timestamp did not advance"
            )
            session = client.get_emet_session()
            assert session["runtime_kind"] == "robocasa_native"
            assert "sim_object_placements" not in session and "sim_articulations" not in session
            assert obs.rgb.shape[:2] == obs.depth.shape
            assert obs.ee_rgb.shape[:2] == obs.ee_depth.shape
            assert np.count_nonzero(obs.ee_depth > 0) > obs.ee_depth.size // 2
            assert (
                obs.image_timing["ee_cam"]["depth_capture_interval_ns"]
                == obs.image_timing["head_cam_left"]["capture_interval_ns"]
            )
            assert np.count_nonzero(np.isfinite(obs.depth) & (obs.depth > 0)) > obs.depth.size // 2
            assert obs.camera_K[0, 0] > 0 and obs.ee_camera_K[0, 0] > 0
            assert np.isfinite(obs.camera_pose).all() and not np.allclose(obs.ee_pose, np.eye(4))
            from emet.agent.tools import PENDING_DISCORD_IMAGE_KEY, get_tools

            context = {"robot": client}
            tools = {tool.name: tool for tool in get_tools(context)}
            photo = tools["send_image"].func()
            assert photo.ok and photo.status == "captured"
            Image.fromarray(context[PENDING_DISCORD_IMAGE_KEY]).save(output / "agent-photo.png")
            assert not (output / "actuation.jsonl").exists()
            initial_base = client.get_base_pose()
            initial = client.get_joint_positions()
            assert initial.shape == (len(CONTROL_JOINTS) + 2,)
            limits = np.asarray([session["native_joint_goal_limits"][name] for name in CONTROL_JOINTS])
            home = np.clip(initial[: len(CONTROL_JOINTS)], limits[:, 0], limits[:, 1])
            goal = home.copy()
            goal[0] += 0.05
            goal[-1] += 0.02
            closed_receipt, closed = check_motion(client, goal, open_gripper=False)
            open_receipt, opened = check_motion(client, home, open_gripper=True)
            for invalid, reason in (
                ([0.0] * 6, "seven finite arm joint targets"),
                ([*home[:-1], limits[-1, 1] + 0.1], "outside the native joint limits"),
            ):
                try:
                    client.send_action({"joint": invalid})
                except RuntimeError as exc:
                    assert reason in str(exc)
                else:
                    raise AssertionError("invalid native joint command was accepted")
            assert client.arm_to(home[:7] + [0.035, 0, 0, 0, 0, 0, 0], timeout=5.0)
            assert client.arm_to(home[:7], timeout=5.0)
            finger_goal = np.array([0.021, -0.021])
            assert client.command_gripper_joints(finger_goal)

            def fingers_arrived():
                measured = client.get_joint_positions(timeout=0.3)
                if measured is not None and np.max(np.abs(measured[-2:] - finger_goal)) <= 0.001:
                    return measured[-2:].tolist()
                return None

            measured_finger_target = wait_for(fingers_arrived, "finite finger target did not settle")
            with client._obs_lock:
                frame = client._obs
                assert frame["image_sim_time"] == frame["ee_image_sim_time"]
            base_drift = float(np.linalg.norm(client.get_base_pose() - initial_base))
            assert base_drift < 0.003
            Image.fromarray(obs.rgb).save(output / "front.png")
            Image.fromarray(obs.ee_rgb).save(output / "wrist.png")
            np.savez(
                output / "rgbd.npz",
                depth=obs.depth,
                wrist_depth=obs.ee_depth,
                K=obs.camera_K,
                camera_pose=obs.camera_pose,
                ee_pose=obs.ee_pose,
                joint=obs.joint,
            )
            report = {
                "passed": True,
                "session": session,
                "image_sim_times": [first_time, second_time],
                "base_pose_delta_norm": base_drift,
                "initial_joint_positions": initial.tolist(),
                "measured_closed": closed.tolist(),
                "measured_open": opened.tolist(),
                "measured_finger_target": measured_finger_target,
                "receipts": [closed_receipt, open_receipt],
                "invalid_goal_rejected": True,
                "photo_tool": photo.to_dict(),
                "synchronized_wrist_rgb": True,
            }
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            if client is not None:
                client.stop()
            terminate_process_tree(server)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port-offset", type=int, default=2420)
    parser.add_argument("--sim-pythonpath", help="Dependency path for a separately pinned RoboCasa process")
    args = parser.parse_args()
    run(args.output, args.port_offset, sim_pythonpath=args.sim_pythonpath)
