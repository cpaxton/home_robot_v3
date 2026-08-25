#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Benchmark sustained ZMQ full-observation streaming (CPU only, no models/GPU).

Connects a bare SUB socket to the robot/sim observation port and measures frame rate,
wire/payload bytes, jitter, and optional client-side JPEG/JP2 decode cost — without
instantiating any perception or VLM model.

Usage:
  uv run python scripts/benchmark_zmq_obs_stream.py --connection herman --seconds 30
  uv run python scripts/benchmark_zmq_obs_stream.py --robot-ip 192.168.1.43 --seconds 10 --decode
  uv run python scripts/benchmark_zmq_obs_stream.py --connection mars --frames 100 --json-out /tmp/zmq.json

Remote default robot: use `--connection <name>` from `emet connect save …`, or pass
`--robot-ip`. Sim on localhost needs no args (obs port 4401).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from emet.app.zmq_stream_benchmark import BenchmarkOptions, run_benchmark, summarize


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", default=None, help="Robot ZMQ host (overrides --connection)")
    parser.add_argument("--connection", "-c", default=None, help="Saved connection profile (host + robot)")
    parser.add_argument("--port", type=int, default=4401, help="Observation SUB port (default 4401)")
    parser.add_argument("--port-offset", type=int, default=0, help="Add to --port (multi-robot sim)")
    parser.add_argument("--seconds", type=float, default=10.0, help="Stream duration (default 10s)")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (overrides --seconds)")
    parser.add_argument("--timeout-ms", type=int, default=15_000, help="ZMQ receive timeout for first frame")
    parser.add_argument("--decode", action="store_true", help="Measure client JPEG/JP2 decode cost (CPU only)")
    parser.add_argument("--no-conflate", action="store_true", help="Disable CONFLATE=1 (expose drops via step gaps)")
    parser.add_argument("--warmup-frames", type=int, default=2, help="Frames to discard before timing starts")
    parser.add_argument("--json-out", default=None, help="Write machine-readable stats to this path")
    parser.add_argument("--verbose", action="store_true", help="Per-frame wire/payload lines")
    args = parser.parse_args()

    host = _resolve_host(args.connection, args.robot_ip)
    port = args.port + args.port_offset
    opts = BenchmarkOptions(
        host=host,
        port=port,
        seconds=args.seconds,
        frames=args.frames,
        timeout_ms=args.timeout_ms,
        conflate=not args.no_conflate,
        decode=args.decode,
        warmup_frames=args.warmup_frames,
        verbose=args.verbose,
    )
    print(f"Benchmarking ZMQ obs at tcp://{host}:{port} …")
    try:
        result = run_benchmark(opts)
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    text = summarize(result, json_out=args.json_out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())