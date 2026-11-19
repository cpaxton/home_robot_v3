#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Serial full-bridge clock qualification, without a model or motion commands.

Use under the managed GPU lease: the ordinary RGB-D renderer stays enabled.
Published privileged clock/upright values are diagnostics, never agent input.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import zmq
from run_shared_room_pairs import assert_frozen_source, git

from emet.utils.job_registry import validated_gpu_lock_fd
from emet.utils.process_tree import popen_session, terminate_process_tree


def run(args):
    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Use emet jobs --gpu-exclusive")
    if not 3 <= args.seconds <= 60:
        raise ValueError("Sample for 3..60 seconds after startup")
    roots = []
    for entry in args.source:
        name, directory = entry.split("=", 1)
        root = Path(directory).resolve()
        revision = git(root, "rev-parse", "HEAD")
        assert_frozen_source(root, revision)
        roots.append((name, root, revision))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (name, root, revision) in enumerate(roots):
        offset = args.port_offset + 10 * index
        row = {"name": name, "source_commit": revision, "cpus": args.cpus, "complete": False}
        env = {**os.environ, "PYTHONPATH": str(root / "src"), "MUJOCO_GL": "egl", "EMET_EVAL_RERUN": "0"}
        env.pop("EMET_CONFIG", None)
        command = [
            "taskset",
            "-c",
            args.cpus,
            sys.executable,
            "-m",
            "emet.simulation.mujoco_server",
            "--headless",
            "--seed",
            "0",
            "--port-offset",
            str(offset),
        ]
        row["command"] = command
        proc = None
        ctx = zmq.Context()
        state = ctx.socket(zmq.SUB)
        state.setsockopt(zmq.SUBSCRIBE, b"")
        state.setsockopt(zmq.CONFLATE, 1)
        state.connect(f"tcp://127.0.0.1:{4403 + offset}")
        try:
            with (args.output / f"{name}.log").open("w") as log:
                proc = popen_session(command, cwd=root, env=env, stdout=log, stderr=log)
                startup_deadline = time.monotonic() + 120
                samples = []
                until = None
                while until is None or time.monotonic() < until:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Bridge exited: {proc.returncode}")
                    if not state.poll(1000):
                        if until is None and time.monotonic() > startup_deadline:
                            raise TimeoutError("No published physics clock within 120s")
                        continue
                    message = state.recv_pyobj()
                    snapshot = message.get("native_tamp_snapshot") or {}
                    sim_time = snapshot.get("sim_time")
                    if sim_time is None:
                        if time.monotonic() > startup_deadline:
                            raise ValueError("Bridge omitted its physics clock")
                        continue
                    if until is None:
                        until = time.monotonic() + args.seconds
                    samples.append(
                        {
                            "wall_time": time.monotonic(),
                            "sim_time": sim_time,
                            "captured_at": snapshot.get("captured_at"),
                            "reported_rtf": message.get("sim_to_real_ratio"),
                            "upright": message.get("base_up_dot_world_z"),
                        }
                    )
                if len(samples) < 2:
                    raise ValueError("Insufficient clock samples")
                elapsed = samples[-1]["wall_time"] - samples[0]["wall_time"]
                row.update(
                    complete=True,
                    samples=samples,
                    measured_rtf=(samples[-1]["sim_time"] - samples[0]["sim_time"]) / elapsed,
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            terminate_process_tree(proc, grace_s=10)
            state.close(0)
            ctx.term()
        rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    return 0 if all(row["complete"] for row in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="name=/path/to/clean/source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", default="12")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--port-offset", type=int, default=900)
    raise SystemExit(run(parser.parse_args()))
