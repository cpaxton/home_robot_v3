#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
# Simulation-only posture/optics probe across registered robots and scene configs.
# Historical filename retained for existing callers; no perception models needed.

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def image_up_world_z(camera_pose: np.ndarray, intrinsics: np.ndarray) -> float:
    """World-up component of decreasing image rows, including signed/rotated K."""
    image_up_camera = np.linalg.solve(intrinsics, np.array([0.0, -1.0, 0.0]))
    image_up_world = camera_pose[:3, :3] @ image_up_camera
    return float(image_up_world[2] / np.linalg.norm(image_up_world))


def _wait_port(port: int, timeout: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def main() -> int:
    def terminate(_signum, _frame):
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="configs/sim/robocasa_pick_place_rby1.yaml")
    parser.add_argument("--port-offset", type=int, default=98)
    parser.add_argument("--poses", type=int, default=6)
    parser.add_argument("--robot", help="Override robot in the scene config (simulation only).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stationary", action="store_true", help="Idle-only probe for fixed-base arms.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
    from emet.utils.process_tree import popen_session, terminate_process_tree

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("EMET_ZMQ_FULL_HZ", "5")
    env.setdefault("EMET_ZMQ_STATE_HZ", "30")
    env.setdefault("EMET_ZMQ_SERVO_HZ", "10")

    sim_cfg = load_sim_launch_config_from_path(args.sim)
    if args.robot:
        sim_cfg.robot = args.robot
    sim_cfg.port_offset = args.port_offset
    sim_cfg.headless = True
    server_argv = prepare_mujoco_server_argv(sim_cfg)
    server_cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *server_argv]
    recv_port = 4401 + args.port_offset

    print("launching sim:", " ".join(server_cmd), file=sys.stderr)
    server_log = args.output_dir / "sim.log"
    fh = server_log.open("w", encoding="utf-8")
    server = popen_session(server_cmd, env=env, stdout=subprocess.DEVNULL, stderr=fh)
    robot = None
    reports = []
    failures = []
    try:
        if not _wait_port(recv_port, timeout=120.0, proc=server):
            raise RuntimeError("sim server did not bind")
        time.sleep(15.0)

        from emet.app.robot_cli import create_robot_client_from_cli
        from emet.core.parameters import get_parameters

        params = get_parameters("dynav_config.yaml")
        robot_kind = str(getattr(sim_cfg, "robot", "stretch"))
        robot = create_robot_client_from_cli(
            robot_kind,
            "127.0.0.1",
            port_offset=args.port_offset,
            parameters=params,
            enable_rerun_server=False,
            start_immediately=True,
        )
        time.sleep(3.0)

        model = getattr(robot, "_model", None)
        data = getattr(robot, "_data", None)
        if model is None or data is None:
            print("no model/data on client; falling back to obs poses", file=sys.stderr)

        from PIL import Image

        from emet.robots import get_robot_spec

        spec = get_robot_spec(robot_kind)
        if not args.stationary:
            robot.move_to_nav_posture()
            robot.look_front(blocking=True)
        time.sleep(2.0)
        for i in range(max(1, int(args.poses)) + 1):
            try:
                if i and not args.stationary:
                    arrived = robot.move_base_to(
                        [0.0, 0.0, 2.0 * np.pi / max(1, int(args.poses))], relative=True, blocking=True, timeout=30.0
                    )
                    if arrived is not True:
                        raise RuntimeError("navigation did not report command-specific success")
            except Exception as e:
                print(f"pose {i}: move failed: {e}", file=sys.stderr)
                failures.append(f"pose {i}: move failed: {e}")
            time.sleep(1.0)
            obs = robot.get_observation()
            rgb = np.asarray(getattr(obs, "rgb", None))
            depth = np.asarray(getattr(obs, "depth", None), dtype=np.float32)
            cam_pose = np.asarray(getattr(obs, "camera_pose", None), dtype=np.float64).reshape(-1)

            report: dict[str, Any] = {"pose": i}
            report["navigation_receipt"] = getattr(robot, "_command_receipt", None)
            q, _, _ = robot.get_joint_state()
            if q is not None:
                report["joint_positions_named"] = dict(zip(spec.joint_names, np.asarray(q).tolist(), strict=False))
            state = getattr(robot, "_state", None) or {}
            report["base_up_dot_world_z"] = state.get("base_up_dot_world_z")
            report["base_xyz"] = state.get("base_xyz")
            targets = state.get("actuator_targets")
            if targets is not None:
                report["joint_targets_named"] = dict(
                    zip(spec.actuator_names, np.asarray(targets).tolist(), strict=True)
                )
            gps = np.asarray(getattr(obs, "gps", None), dtype=np.float64).reshape(-1)
            compass = np.asarray(getattr(obs, "compass", None), dtype=np.float64).reshape(-1)
            if gps.size >= 2:
                report["gps"] = [round(float(x), 3) for x in gps]
            if compass.size:
                report["compass_yaw"] = round(float(compass[0]), 3)
            for fld in ("base_pose", "base_xyt", "base_quat", "joint_positions", "qpos", "head_pose"):
                if hasattr(obs, fld):
                    v = np.asarray(getattr(obs, fld), dtype=np.float64)
                    if v.size:
                        report[fld] = [round(float(x), 3) for x in v.reshape(-1)]
            sess = getattr(obs, "emet_session", None) or {}
            if isinstance(sess, dict) and sess.get("navigation_origin_xyt") is not None:
                report["nav_origin"] = [
                    round(float(x), 3) for x in np.asarray(sess["navigation_origin_xyt"]).reshape(-1)
                ]
            if rgb is not None and rgb.size:
                gray = rgb.mean(axis=-1).astype(np.float32)
                report["rgb_mean"] = round(float(gray.mean()), 1)
                report["rgb_std"] = round(float(gray.std()), 1)
            if depth is not None and depth.size:
                v = np.isfinite(depth) & (depth > 1e-6)
                near = v & (depth <= 5.0)
                report["depth_max"] = round(float(depth[v].max()), 2) if v.any() else None
                report["depth_valid_frac"] = round(float(v.mean()), 4)
                report["depth_le5m_frac"] = round(float(near.mean()), 4)
            if cam_pose is not None and cam_pose.size >= 12:
                try:
                    cp4 = np.asarray(cam_pose, dtype=np.float64).reshape(4, 4)
                    report["cam_origin"] = [round(float(x), 3) for x in cp4[:3, 3]]
                    report["raw_camera_up_dot_world_z"] = float(-cp4[2, 1])
                    intrinsics = getattr(obs, "camera_K", None)
                    if intrinsics is not None:
                        report["camera_up_dot_world_z"] = image_up_world_z(cp4, np.asarray(intrinsics))
                    report["camera_forward_z"] = float(cp4[2, 2])
                except Exception:
                    report["cam_origin"] = [round(float(x), 3) for x in cam_pose[:3]]
            if model is not None and data is not None:
                base = mujoco_name_to_id(model, "body", "base_link")
                report["base_z"] = round(float(data.xpos[base, 2]), 3) if base >= 0 else None
                for jn in ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"):
                    j = mujoco_name_to_id(model, "joint", jn)
                    if j >= 0:
                        report[jn] = round(float(data.qpos[model.jnt_qposadr[j]]), 3)
            if rgb.ndim == 3:
                Image.fromarray(rgb.astype(np.uint8)).save(args.output_dir / f"view_{i:02d}.png")
            reports.append(report)
            with (args.output_dir / "observations.jsonl").open("a") as out:
                out.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
            if not args.stationary and report.get("camera_up_dot_world_z", 1) < 0:
                failures.append(f"pose {i}: camera inverted")
            if report.get("base_up_dot_world_z") is not None and report["base_up_dot_world_z"] < 0.57:
                failures.append(f"pose {i}: base tipped")
            if i and "cam_origin" in report and "cam_origin" in reports[0]:
                if report["cam_origin"][2] < reports[0]["cam_origin"][2] - 0.3:
                    failures.append(f"pose {i}: camera dropped more than 0.3 m")
            if failures:
                break

        summary = {"robot": robot_kind, "scene": args.sim, "failures": failures, "frames": len(reports)}
        missing = any(r.get("base_up_dot_world_z") is None or "joint_targets_named" not in r for r in reports)
        summary["status"] = "failed" if failures else ("incomplete_telemetry" if missing else "completed_probe")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 1 if failures else 0
    finally:
        try:
            if robot is not None:
                robot.stop()
        finally:
            terminate_process_tree(server)
            fh.close()


def mujoco_name_to_id(model, obj: str, name: str) -> int:
    import mujoco

    kind = {"body": mujoco.mjtObj.mjOBJ_BODY, "joint": mujoco.mjtObj.mjOBJ_JOINT}[obj]
    return mujoco.mj_name2id(model, kind, name)


if __name__ == "__main__":
    sys.exit(main())
