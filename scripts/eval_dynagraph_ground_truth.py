#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Batch metrics for Dynagraph ground-truth episodes (saved export or live sim run)."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _wait_port(port: int, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _load_placements_from_episode(episode_dir: Path) -> dict[str, dict[str, Any]] | None:
    from emet.memory.format import SIM_GT_PLACEMENTS_FILENAME
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    p = episode_dir / SIM_GT_PLACEMENTS_FILENAME
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        return read_sim_object_placements({"sim_object_placements": raw})
    return None


def _graph_memory_from_episode(episode_dir: Path):
    from emet.memory.backend import get_memory_backend
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    mem = GraphEQAMemory(defer_llm_clients=True)
    backend = get_memory_backend("graph_eqa", graph_memory=mem, voxel_map=None)
    backend.load(str(episode_dir))
    return mem


def compute_episode_metrics(episode_dir: str | Path) -> dict[str, Any]:
    """Compute GT localization / association metrics from a MemoryState export directory."""
    from emet.memory.format import load_memory
    from emet.memory.graph_eqa.sim_ground_truth_graph import (
        gt_graph_completeness,
        gt_localization_errors,
        instance_gt_association_recall,
        projected_association_recall,
    )

    episode_dir = Path(episode_dir)
    state = load_memory(str(episode_dir))
    placements = _load_placements_from_episode(episode_dir)
    mem = _graph_memory_from_episode(episode_dir)

    loc_errors = gt_localization_errors(mem, placements)
    mean_xy = float(np.mean([v["err_xy_m"] for v in loc_errors.values()])) if loc_errors else None
    mean_z = float(np.mean([v["err_z_m"] for v in loc_errors.values()])) if loc_errors else None

    metrics: dict[str, Any] = {
        "episode_dir": str(episode_dir),
        "ground_truth_mode": bool(getattr(state.manifest, "ground_truth_mode", False)),
        "n_frames": len(state.frames),
        "n_graph_nodes": len(state.graph.nodes) if state.graph else 0,
        "n_placements": len(placements) if placements else 0,
        "gt_graph_completeness": gt_graph_completeness(mem, placements),
        "instance_gt_association_recall": instance_gt_association_recall(mem, placements),
        "projected_association_recall": projected_association_recall(state.frames, placements),
        "localization_mean_err_xy_m": mean_xy,
        "localization_mean_err_z_m": mean_z,
        "localization_errors": loc_errors,
    }
    return metrics


def _run_live_export(
    export_dir: Path,
    *,
    port_offset: int,
    not_rotate: bool,
    cpu_only: bool,
) -> int:
    recv_port = 4401 + port_offset
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "emet.simulation.mujoco_server",
            "--headless",
            "--port-offset",
            str(port_offset),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_port(recv_port):
            return 1
        time.sleep(15)
        client_cmd = [
            sys.executable,
            "-m",
            "emet.app.run_dynagraph",
            "--no-rerun",
            "--ground-truth",
            "--port-offset",
            str(port_offset),
            "--export",
            str(export_dir),
        ]
        if not_rotate:
            client_cmd.append("--not_rotate_in_place")
        if cpu_only:
            client_cmd.append("--cpu-only")
        res = subprocess.run(client_cmd, env=env, timeout=180)
        return int(res.returncode)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Dynagraph GT episode metrics.")
    parser.add_argument("--episode", type=str, help="Existing MemoryState export directory.")
    parser.add_argument("--run-live", action="store_true", help="Spawn sim + export before eval.")
    parser.add_argument("--port-offset", type=int, default=int(os.getpid() % 400 + 120))
    parser.add_argument("--not-rotate", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", type=str, help="Write metrics JSON to this path.")
    args = parser.parse_args()

    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    episode_dir: Path | None = Path(args.episode) if args.episode else None
    if args.run_live:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="dynagraph_gt_eval_")
        episode_dir = Path(tmp_ctx.name)
        rc = _run_live_export(
            episode_dir,
            port_offset=args.port_offset,
            not_rotate=args.not_rotate,
            cpu_only=args.cpu_only,
        )
        if rc != 0:
            print(f"Live export failed (exit {rc})", file=sys.stderr)
            return rc
    if episode_dir is None:
        parser.error("Provide --episode or --run-live")

    metrics = compute_episode_metrics(episode_dir)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
