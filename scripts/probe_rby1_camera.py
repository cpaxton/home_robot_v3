#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
# Probe: why does rby1/robocasa capture ~no depth? Boot the sim, read one full
# observation, and dump base-z / camera-z / torso joint angles / depth stats.

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="configs/sim/robocasa_pick_place_rby1.yaml")
    parser.add_argument("--port-offset", type=int, default=98)
    parser.add_argument("--poses", type=int, default=6)
    args = parser.parse_args()

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
    sim_cfg.port_offset = args.port_offset
    sim_cfg.headless = True
    server_argv = prepare_mujoco_server_argv(sim_cfg)
    server_cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *server_argv]
    recv_port = 4401 + args.port_offset

    print("launching sim:", " ".join(server_cmd), file=sys.stderr)
    server_log = REPO / "probe_rby1_camera_sim.log"
    fh = server_log.open("w", encoding="utf-8")
    server = popen_session(server_cmd, env=env, stdout=subprocess.DEVNULL, stderr=fh)
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

        for i in range(max(1, int(args.poses))):
            try:
                robot.move_base_to(
                    [0.0, 0.0, 2.0 * np.pi / max(1, int(args.poses))], relative=True, blocking=True, timeout=30.0
                )
            except Exception as e:
                print(f"pose {i}: move failed: {e}", file=sys.stderr)
            time.sleep(1.0)
            obs = robot.get_observation()
            rgb = np.asarray(getattr(obs, "rgb", None))
            depth = np.asarray(getattr(obs, "depth", None), dtype=np.float32)
            cam_pose = np.asarray(getattr(obs, "camera_pose", None), dtype=np.float64).reshape(-1)

            report: dict[str, object] = {"pose": i}
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
                except Exception:
                    report["cam_origin"] = [round(float(x), 3) for x in cam_pose[:3]]
            if model is not None and data is not None:
                base = mujoco_name_to_id(model, "body", "base_link")
                report["base_z"] = round(float(data.xpos[base, 2]), 3) if base >= 0 else None
                for jn in ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"):
                    j = mujoco_name_to_id(model, "joint", jn)
                    if j >= 0:
                        report[jn] = round(float(data.qpos[model.jnt_qposadr[j]]), 3)
            print(report, flush=True)

        return 0
    finally:
        terminate_process_tree(server)


def mujoco_name_to_id(model, obj: str, name: str) -> int:
    import mujoco

    kind = {"body": mujoco.mjtObj.mjOBJ_BODY, "joint": mujoco.mjtObj.mjOBJ_JOINT}[obj]
    return mujoco.mj_name2id(model, kind, name)


if __name__ == "__main__":
    sys.exit(main())
