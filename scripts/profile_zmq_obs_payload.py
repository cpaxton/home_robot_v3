#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Profile ZMQ full-observation message sizes (find duplicate JPEG streams).

Usage:
  uv run python scripts/profile_zmq_obs_payload.py --robot-ip 192.168.1.43
  uv run python scripts/profile_zmq_obs_payload.py --connection herman --samples 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

_IMAGE_KEYS = frozenset(
    {
        "rgb",
        "rgb_right",
        "rgb_tertiary",
        "depth",
        "head_cam_left/image",
        "head_cam_right/image",
        "ee_cam/image",
    }
)


def _sizeof_value(val: object) -> int:
    if val is None:
        return 0
    if isinstance(val, (bytes, bytearray)):
        return len(val)
    if isinstance(val, dict):
        return sum(_sizeof_value(k) + _sizeof_value(v) for k, v in val.items())
    if isinstance(val, (list, tuple, set, frozenset)):
        return sum(_sizeof_value(v) for v in val)
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return int(val.nbytes)
    except ImportError:
        pass
    return len(str(val).encode("utf-8", errors="replace"))


def _resolve_host(connection: str | None, robot_ip: str | None) -> str:
    if robot_ip:
        return robot_ip.strip()
    if connection:
        from emet.utils.connection import get_connection

        conn = get_connection(connection)
        if conn and conn.get("host"):
            return str(conn["host"]).strip()
    from emet.utils.connection import get_active_connection

    conn = get_active_connection()
    if conn and conn.get("host"):
        return str(conn["host"]).strip()
    return "127.0.0.1"


def profile_obs(host: str, *, port: int, port_offset: int, samples: int, timeout_ms: int) -> int:
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    recv_port = port + port_offset
    sock.connect(f"tcp://{host}:{recv_port}")

    totals: dict[str, int] = {}
    image_totals: dict[str, int] = {}
    n_ok = 0
    for _ in range(samples):
        try:
            msg = sock.recv_pyobj()
        except zmq.Again:
            print(f"TIMEOUT waiting for obs on tcp://{host}:{recv_port}", file=sys.stderr)
            return 1
        if not isinstance(msg, dict):
            continue
        n_ok += 1
        step_total = 0
        for key, val in msg.items():
            sz = _sizeof_value(val)
            totals[key] = totals.get(key, 0) + sz
            step_total += sz
            if key in _IMAGE_KEYS or (isinstance(val, (bytes, bytearray)) and sz > 10_000):
                image_totals[key] = image_totals.get(key, 0) + sz
        totals["__step_total__"] = totals.get("__step_total__", 0) + step_total

    sock.close()
    ctx.term()

    if n_ok == 0:
        print("No observation messages received.", file=sys.stderr)
        return 1

    print(f"ZMQ obs profile  host={host}  port={recv_port}  samples={n_ok}")
    print(f"{'key':<28} {'avg_bytes':>12} {'share':>8}")
    grand = totals.get("__step_total__", 1) or 1
    for key in sorted(totals):
        if key == "__step_total__":
            continue
        avg = totals[key] / n_ok
        share = 100.0 * totals[key] / grand
        flag = "  ← image/depth" if key in _IMAGE_KEYS else ""
        print(f"{key:<28} {avg:12.0f} {share:7.1f}%{flag}")
    print(f"{'TOTAL (avg/step)':<28} {grand / n_ok:12.0f}")
    if image_totals:
        print("\nImage-like keys (check for duplicate head/ee JPEGs):")
        for key in sorted(image_totals, key=lambda k: -image_totals[k]):
            print(f"  {key}: {image_totals[key] / n_ok:.0f} bytes avg")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=None, help="Robot ZMQ host")
    parser.add_argument("--connection", "-c", default=None, help="Saved connection profile")
    parser.add_argument("--port", type=int, default=4401, help="Base recv port")
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3, help="Observation frames to average")
    parser.add_argument("--timeout-ms", type=int, default=15_000)
    args = parser.parse_args()
    host = _resolve_host(args.connection, args.robot_ip)
    return profile_obs(
        host,
        port=args.port,
        port_offset=args.port_offset,
        samples=max(1, args.samples),
        timeout_ms=args.timeout_ms,
    )


if __name__ == "__main__":
    raise SystemExit(main())
