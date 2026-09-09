#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Gated hold/known-route probe through the production simulation ZMQ bridge.

Physics runs. The production server holds planar base pose while idle and drives
it holonomically during navigation; this is not a wheel-actuator dynamics test.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from probe_stationary_objects import setup_scene


def pose_error(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    angle = actual[2] - desired[2]
    return float(np.linalg.norm(actual[:2] - desired[:2])), float(abs(np.arctan2(np.sin(angle), np.cos(angle))))


def load_live_scene(config, scene):
    import xml.etree.ElementTree as ET

    import mujoco

    from emet.simulation.mujoco_stationary_control import compute_stationary_ctrl_vector

    spec, model, data = setup_scene(config)
    for key in range(model.nkey):
        model.key_qpos[key] = data.qpos
        model.key_ctrl[key] = compute_stationary_ctrl_vector(model, data)
    # Recompile edited fixture geometry so collision bounds/inertia constants
    # agree with the larger diagnostic objects, rather than changing render sizes only.
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            bid, adr = int(model.jnt_bodyid[jid]), int(model.jnt_qposadr[jid])
            model.body_pos[bid] = data.qpos[adr : adr + 3]
            model.body_quat[bid] = data.qpos[adr + 3 : adr + 7]
    mujoco.mj_saveLastXML(str(scene), model)
    tree = ET.parse(scene)
    compiler = tree.getroot().find("compiler")
    compiler.set("assetdir", str(Path(spec.mjcf_path).parent / "meshes"))
    compiler.set("meshdir", str(Path(spec.mjcf_path).parent / "meshes"))
    tree.write(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    return spec, model


def serve(config, offset, output):
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    spec, model = load_live_scene(config, output / "scene.xml")
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=4401 + offset,
        recv_port=4402 + offset,
        send_state_port=4403 + offset,
        send_servo_port=4404 + offset,
        use_remote_computer=False,
        environment={"kind": "default_mujoco"},
        navigation_xy_tolerance=config["live_probe"]["waypoint_xy_tolerance_m"],
        navigation_yaw_tolerance=config["live_probe"]["yaw_tolerance_rad"],
    )
    try:
        server.start(headless=True)
    finally:
        server.stop()


def run(config, config_path, output, offset):
    from probe_rby1_camera import _wait_port

    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.core.parameters import get_parameters
    from emet.utils.process_tree import popen_session, terminate_process_tree

    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, EMET_SIM_NAV_TELEPORT="0", EMET_MOLMOSPACES_NAV_TELEPORT="0", PYTHONUNBUFFERED="1", MUJOCO_GL="egl"
    )
    settings = config["live_probe"]
    report = {
        "robot": config["robot"],
        "base_idle_mode": "production planar pose hold; height/roll/pitch dynamic",
        "navigation_mode": "production holonomic velocity; teleport disabled",
        "settings": settings,
        "stages": [],
    }
    robot = None
    with (output / "sim.log").open("w") as log:
        proc = popen_session(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--serve",
                "--config",
                str(config_path),
                "--port-offset",
                str(offset),
                "--output-dir",
                str(output.resolve()),
            ],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            if not _wait_port(4401 + offset, 120, proc):
                raise RuntimeError("simulation bridge did not start")
            robot = create_robot_client_from_cli(
                config["robot"],
                "127.0.0.1",
                port_offset=offset,
                parameters=get_parameters("dynav_config.yaml"),
                enable_rerun_server=False,
                start_immediately=True,
            )
            robot.head_to(config["head_pan"], config["head_tilt"], blocking=True, timeout=30)
            time.sleep(2)

            def capture(label, expected):
                obs = robot.get_observation()
                session = robot.get_emet_session() or {}
                origin = session.get("navigation_origin_xyt")
                if origin is None:
                    raise RuntimeError("missing navigation world-frame origin")
                from emet.utils.geometry import xyt_base_to_global

                actual = np.asarray(xyt_base_to_global(robot.get_base_pose(), np.asarray(origin)))
                xy_error, yaw_error = pose_error(actual, expected)
                state = getattr(robot, "_state", {}) or {}
                pose = np.asarray(obs.camera_pose)
                forward = pose[:3, 2]
                optical_yaw = float(np.arctan2(forward[1], forward[0]))
                yaw_delta = optical_yaw - actual[2] - config["head_pan"]
                row = {
                    "label": label,
                    "base_xyt": actual.tolist(),
                    "expected_xyt": list(expected),
                    "xy_error_m": xy_error,
                    "yaw_error_rad": yaw_error,
                    "base_up_dot_world_z": state.get("base_up_dot_world_z"),
                    "joint_positions": np.asarray(state.get("joint_positions", [])).tolist(),
                    "actuator_targets": np.asarray(state.get("actuator_targets", [])).tolist(),
                    "camera_pose": pose.tolist(),
                    "camera_yaw_error_rad": float(abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))),
                    "camera_pitch_error_rad": float(abs(np.arcsin(np.clip(forward[2], -1, 1)) - config["head_tilt"])),
                    "receipt": getattr(robot, "_command_receipt", None),
                }
                rgb = np.asarray(obs.rgb).astype(np.uint8)
                Image.fromarray(rgb).save(output / f"{label}.png")
                np.savez_compressed(
                    output / f"{label}.npz",
                    rgb=rgb,
                    depth=np.asarray(obs.depth),
                    camera_pose=pose,
                    camera_K=np.asarray(obs.camera_K),
                )
                with (output / "observations.jsonl").open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                return row

            def settled_capture(label, expected):
                deadline = time.monotonic() + 30
                stable = 0
                index = 0
                while time.monotonic() < deadline:
                    row = capture(f"{label}_settle_{index:02d}", expected)
                    good = (
                        row["xy_error_m"] <= settings["waypoint_xy_tolerance_m"]
                        and row["yaw_error_rad"] <= settings["yaw_tolerance_rad"]
                        and row["camera_yaw_error_rad"] <= np.deg2rad(1)
                        and row["camera_pitch_error_rad"] <= np.deg2rad(1)
                        and row["base_up_dot_world_z"] is not None
                        and row["base_up_dot_world_z"] > 0.98
                    )
                    stable = stable + 1 if good else 0
                    if stable >= 3:
                        return row
                    index += 1
                    time.sleep(0.5)
                raise RuntimeError(f"{label} did not settle within 30 seconds")

            initial = settled_capture("hold_start", config["base_xyt"])
            samples = []
            for i in range(int(settings["hold_seconds"])):
                time.sleep(1)
                samples.append(capture(f"hold_{i:02d}", config["base_xyt"]))
            drift = max(pose_error(s["base_xyt"], initial["base_xyt"])[0] for s in samples)
            hold_pass = drift <= settings["hold_xy_tolerance_m"] and all(
                s["xy_error_m"] <= settings["waypoint_xy_tolerance_m"]
                and s["yaw_error_rad"] <= settings["yaw_tolerance_rad"]
                and s["base_up_dot_world_z"] is not None
                and s["base_up_dot_world_z"] > 0.98
                and s["camera_yaw_error_rad"] <= np.deg2rad(1)
                and s["camera_pitch_error_rad"] <= np.deg2rad(1)
                for s in samples
            )
            report["stages"].append(
                {
                    "stage": "hold",
                    "passed": hold_pass,
                    "max_xy_drift_m": drift,
                    "max_camera_pitch_error_rad": max(s["camera_pitch_error_rad"] for s in samples),
                }
            )
            if not hold_pass:
                raise RuntimeError("hold gate failed; route not attempted")
            for i, goal in enumerate(settings["route"]):
                arrived = robot.move_base_to(
                    goal, world_frame=True, blocking=True, timeout=30, navigation_policy="precision"
                )
                row = settled_capture(f"route_{i:02d}", goal)
                passed = (
                    bool(arrived)
                    and row["xy_error_m"] <= settings["waypoint_xy_tolerance_m"]
                    and row["yaw_error_rad"] <= settings["yaw_tolerance_rad"]
                    and row["base_up_dot_world_z"] is not None
                    and row["base_up_dot_world_z"] > 0.98
                    and row["camera_yaw_error_rad"] <= settings["yaw_tolerance_rad"]
                    and row["camera_pitch_error_rad"] <= settings["yaw_tolerance_rad"]
                )
                report["stages"].append(
                    {
                        "stage": f"route_{i}",
                        "passed": passed,
                        "xy_error_m": row["xy_error_m"],
                        "yaw_error_rad": row["yaw_error_rad"],
                    }
                )
                if not passed:
                    raise RuntimeError(f"route waypoint {i} failed; remaining route not attempted")
        except Exception as exc:
            report["error"] = str(exc)
        finally:
            try:
                if robot is not None:
                    robot.stop()
            finally:
                terminate_process_tree(proc)
    report["passed"] = "error" not in report
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ovmm/stationary_rby1.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port-offset", type=int, default=620)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.serve:
        if args.output_dir is None:
            parser.error("--output-dir is required for the server fixture")
        serve(config, args.port_offset, args.output_dir)
    elif args.output_dir is None:
        parser.error("--output-dir is required for the client probe")
    else:
        raise SystemExit(run(config, args.config.resolve(), args.output_dir, args.port_offset))
