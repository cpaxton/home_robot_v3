#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Compare Innate Mars ZMQ camera_pose vs local MJCF FK (sim or hardware).

Examples:
  # Live hardware (bridge on herman):
  uv run python scripts/debug_innate_mars_head_align.py --ip herman

  # Local MuJoCo sim (start ``emet serve mujoco --robot innate_mars`` first):
  uv run python scripts/debug_innate_mars_head_align.py --ip 127.0.0.1

Sim should report ~0 mm / ~0° (same model publishes ``camera_pose``).
Hardware uses TF-calibrated stereo mounts (not the sim table-forward hack); expect sub-mm
position in **base_link** frame when ``joint_head`` matches TF (see ``--infer-head``).
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import zmq

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.robots.innate_mars.head_kinematics import (
    compare_mjcf_camera_to_zmq,
    infer_joint_head_from_camera_pose,
    is_hardware_innate_mars_obs,
    ros_head_deg_to_mjcf_rad,
)


def _recv_obs(host: str, port: int = 4401, timeout_ms: int = 8000) -> dict:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    raw = sock.recv()
    return pickle.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", "--robot-ip", default="127.0.0.1", help="ZMQ host")
    parser.add_argument("--port", type=int, default=4401)
    parser.add_argument(
        "--infer-head",
        action="store_true",
        help="Print MJCF joint_head inferred from camera_pose (workstation MuJoCo)",
    )
    args = parser.parse_args()

    try:
        obs = _recv_obs(args.ip, port=args.port)
    except Exception as exc:
        print(f"Failed to receive ZMQ observation from {args.ip}:{args.port}: {exc}", file=sys.stderr)
        return 1

    session = obs.get(EMET_ZMQ_SESSION_KEY) or {}
    runtime = session.get("runtime_kind", "unknown")
    is_sim = not is_hardware_innate_mars_obs(obs)
    hw = not is_sim
    print(f"Host: {args.ip}:{args.port}")
    print(f"Runtime: {runtime}  is_simulation={is_sim}")
    print(f"ZMQ joint_head: {obs.get('joint_head')!r} rad")

    metrics = compare_mjcf_camera_to_zmq(obs)
    print(
        f"MJCF FK vs ZMQ camera_pose (base frame): "
        f"pos_err={metrics['pos_err_m'] * 1000:.1f} mm  "
        f"gaze_err={metrics['gaze_err_deg']:.2f}°  "
        f"rot_err={metrics['rot_err_deg']:.2f}°"
    )

    if args.infer_head:
        from emet.core.zmq_protocol import read_emet_session

        inferred = infer_joint_head_from_camera_pose(
            np.asarray(obs["joint"]),
            np.asarray(obs["camera_pose"]),
            gps=np.asarray(obs.get("gps", np.zeros(2))),
            compass=np.asarray(obs.get("compass", np.zeros(1))),
            session=read_emet_session(obs),
            use_hardware_cameras=hw,
        )
        m_inf = compare_mjcf_camera_to_zmq(obs, joint_head=inferred)
        print(f"Inferred joint_head: {inferred:.4f} rad ({np.degrees(inferred):.2f}°)")
        print(f"  after infer: pos_err={m_inf['pos_err_m'] * 1000:.1f} mm  gaze_err={m_inf['gaze_err_deg']:.2f}°")

    if obs.get("joint_head") is not None and hw:
        deg = float(np.degrees(-float(obs["joint_head"])))
        print(f"ros_head_deg_to_mjcf sign check: topic≈{deg:.1f}° → mjcf {ros_head_deg_to_mjcf_rad(deg):.4f} rad")

    if is_sim:
        if metrics["pos_err_m"] > 0.001 or metrics["gaze_err_deg"] > 0.5:
            print("WARN: sim alignment worse than expected — check MJCF / server OpenCV pose.", file=sys.stderr)
            return 2
        print("OK: sim MJCF matches ZMQ camera_pose.")
    else:
        if metrics["pos_err_m"] > 0.003 or metrics["gaze_err_deg"] > 1.0:
            print(
                "NOTE: hardware MJCF vs TF still misaligned — check arm FK or remount constants in "
                "head_kinematics.HARDWARE_HEAD_CAMERA_MOUNTS. Mapping uses ZMQ camera_pose from TF.",
            )
        elif metrics["pos_err_m"] <= 0.001 and metrics["gaze_err_deg"] <= 1.0:
            print("OK: hardware MJCF FK matches ZMQ camera_pose within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
