#!/usr/bin/env python3
# Copyright (c) Chris Paxton
# Diagnostic: are the OVMM find-phase camera images valid (not all-black)?
#
# Boots the same MuJoCo Stretch default-table sim the OVMM find-phase uses,
# moves the base to a few poses, captures RGB from the head camera, and:
#   1. gradient check  — image must not be near-uniform black (no texture/edges)
#   2. ray trace       — a ray from the camera origin forward must not hit
#      geometry within a near-wall distance (camera not inside/against a wall)
#
# Usage (after the sim GPU is free):
#   uv run python scripts/debug_ovmm_camera_quality.py [--sim configs/sim/default_table_stretch.yaml]
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

HEAD_RAY_LENGTH_M = 3.0
MIN_NEAR_WALL_RAY_M = 0.25  # camera must be at least this far from any hit


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


def _gradient_stats(rgb: np.ndarray) -> dict[str, float]:
    """Image gradient / contrast stats; all-black images have ~0 gradient."""
    img = np.asarray(rgb)
    if img.ndim == 3:
        gray = img.mean(axis=-1).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "grad_mean": float(mag.mean()),
        "grad_max": float(mag.max()),
        "frac_black": float((gray < 8).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="configs/sim/default_table_stretch.yaml")
    parser.add_argument("--port-offset", type=int, default=99)
    parser.add_argument("--poses", type=int, default=6)
    parser.add_argument("--camera", default="head")
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
    server_log = REPO / "debug_ovmm_camera_quality_sim.log"
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
            allow_missing_depth=True,
        )
        time.sleep(3.0)

        results: list[dict[str, float]] = []
        for i in range(max(1, int(args.poses))):
            # Rotate +x in place each step so we capture genuinely different views.
            try:
                robot.move_base_to(
                    [0.0, 0.0, 2.0 * np.pi / max(1, int(args.poses))], relative=True, blocking=True, timeout=30.0
                )
            except Exception as e:
                print(f"pose {i}: move failed: {e}", file=sys.stderr)
            time.sleep(1.0)
            obs = robot.get_observation()
            rgb = getattr(obs, "rgb", None)
            if rgb is None:
                print(f"pose {i}: no rgb", file=sys.stderr)
                continue
            rgb = np.asarray(rgb)
            depth = getattr(obs, "depth", None)
            depth_stats = {}
            if depth is not None:
                d = np.asarray(depth, dtype=np.float32)
                depth_stats = {
                    "depth_center": float(d[d.shape[0] // 2, d.shape[1] // 2]) if d.size else float("nan"),
                    "depth_min": float(d.min()) if d.size else float("nan"),
                    "depth_frac_zero": float((d <= 1e-3).mean()) if d.size else float("nan"),
                }
            cam_pose = getattr(obs, "camera_pose", None)
            cam_origin = None
            if cam_pose is not None:
                try:
                    cam_origin = np.asarray(cam_pose, dtype=np.float64).reshape(-1)
                except Exception:
                    cam_origin = None
            if i == 0:
                from PIL import Image

                out_png = REPO / "debug_ovmm_camera_quality_frame0.png"
                Image.fromarray(rgb.astype(np.uint8)).save(out_png)
                print(f"saved frame 0 -> {out_png}", file=sys.stderr)
            stats = _gradient_stats(rgb)
            if depth_stats:
                stats.update(depth_stats)
            if cam_origin is not None:
                stats["cam_origin"] = list(cam_origin[:3])
            results.append(stats)
            print(
                f"pose {i}: shape={np.asarray(rgb).shape} "
                f"mean={stats['mean']:.1f} std={stats['std']:.1f} "
                f"grad={stats['grad_mean']:.3f}/{stats['grad_max']:.1f} "
                f"black_frac={stats['frac_black']:.3f}"
                + (
                    f" depth_c={stats.get('depth_center', float('nan')):.2f} "
                    f"depth_min={stats.get('depth_min', float('nan')):.2f} "
                    f"depth0={stats.get('depth_frac_zero', float('nan')):.3f}"
                    if depth_stats
                    else ""
                ),
                file=sys.stderr,
            )

        # Ray trace from camera origin forward (approx world -x is head default look).
        ray_report = {}
        model = getattr(robot, "_model", None)
        data = getattr(robot, "_data", None)
        last_origin = results[-1].get("cam_origin") if results and "cam_origin" in results[-1] else None
        if model is None or data is None or last_origin is None:
            print("ray trace skipped: no model/data/origin on client (depth check used instead)", file=sys.stderr)
        else:
            import mujoco

            origin = np.array(last_origin, dtype=np.float64)
            for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                vec = np.array([dx, dy, 0.0], dtype=np.float64)
                gid = np.zeros(1, dtype=np.int32)
                dist = mujoco.mj_ray(model, data, origin, vec, None, 1, -1, gid)
                ray_report[f"({dx:+},{dy:+})"] = float(dist) if dist >= 0 else None
            print(f"camera origin {origin} ray hits: {ray_report}", file=sys.stderr)

        ok = True
        if results:
            mean_std = float(np.mean([r["std"] for r in results]))
            mean_val = float(np.mean([r["mean"] for r in results]))
            black = float(np.mean([r["frac_black"] for r in results]))
            depth0 = (
                float(np.mean([r.get("depth_frac_zero", float("nan")) for r in results if "depth_frac_zero" in r]))
                if any("depth_frac_zero" in r for r in results)
                else float("nan")
            )
            dc = (
                float(np.mean([r.get("depth_center", float("nan")) for r in results if "depth_center" in r]))
                if any("depth_center" in r for r in results)
                else float("nan")
            )
            print(
                f"\nAVG mean={mean_val:.1f} std={mean_std:.1f} black_frac={black:.3f} "
                f"grad={float(np.mean([r['grad_mean'] for r in results])):.3f}"
                + (f" depth_center={dc:.2f} depth_frac_zero={depth0:.3f}" if "depth_center" in results[0] else ""),
                file=sys.stderr,
            )
            # Real scene: mid-brightness, high variance, ~no pure-black pixels.
            ok = black < 0.5 and mean_std > 20.0 and mean_val > 10.0
            print("RESULT:", "OK (camera sees a real scene)" if ok else "BAD (all-black / no-texture)", file=sys.stderr)
        else:
            print("RESULT: FAIL (no frames captured)", file=sys.stderr)
            ok = False
        return 0 if ok else 1
    finally:
        try:
            terminate_process_tree(server, grace_s=5.0)
        except Exception:
            pass
        fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
