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

from emet.comm import run_comm_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", default="127.0.0.1", help="Robot ZMQ host (overrides --connection)")
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

    return run_comm_benchmark(
        robot_ip=args.robot_ip or "127.0.0.1",
        connection_name=args.connection,
        port=args.port,
        port_offset=args.port_offset,
        seconds=args.seconds,
        frames=args.frames,
        timeout_ms=args.timeout_ms,
        decode=args.decode,
        no_conflate=args.no_conflate,
        warmup_frames=args.warmup_frames,
        json_out=args.json_out,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())