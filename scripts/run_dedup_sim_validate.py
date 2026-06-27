#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""One-shot Robocasa dynagraph run with --compare-to-gt for graph dedup validation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = Path("/tmp/dedup_sim_validate")
DEFAULT_SIM = REPO / "configs/sim/robocasa_pick_place_stretch.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start Robocasa sim, run dynagraph with explore-loop + --compare-to-gt, "
            "and print dedup/GT metrics. Output dir is created before logging starts."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT,
        help=f"Dynagraph export root (default: {DEFAULT_EXPORT})",
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=DEFAULT_SIM,
        help="Sim launch YAML (default: robocasa_pick_place_stretch)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Robocasa layout seed")
    parser.add_argument("--port-offset", type=int, default=220, help="ZMQ port offset")
    parser.add_argument(
        "--explore-max-iters",
        type=int,
        default=8,
        help="Frontier excursions when --explore-loop is enabled",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for perception (default: cpu-only)",
    )
    parser.add_argument(
        "--no-sensor-perception",
        action="store_true",
        help="Use voxel labels only (benchmark-style fusion path)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing export dir contents (default: wipe export dir first)",
    )
    return parser.parse_args()


def _prepare_export_dir(export_dir: Path, *, clean: bool) -> None:
    if clean and export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)


def _build_dynagraph_cmd(
    *,
    export_dir: Path,
    port_offset: int,
    explore_max_iters: int,
    cpu_only: bool,
    no_sensor_perception: bool,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "emet",
        "run",
        "dynagraph",
        "--robot",
        "stretch",
        "--robot-ip",
        "127.0.0.1",
        "--port-offset",
        str(port_offset),
        "--no-rerun",
        "--compare-to-gt",
        "--export",
        str(export_dir),
        "--explore-loop",
        "--explore-max-iters",
        str(explore_max_iters),
        "--explore-max-failures",
        "5",
    ]
    if cpu_only:
        cmd.append("--cpu-only")
    if no_sensor_perception:
        cmd.append("--no-sensor-perception")
    return cmd


def _export_ok(export_dir: Path) -> bool:
    manifest = export_dir / "manifest.json"
    graph = export_dir / "graph.json"
    memory = export_dir / "memory"
    return manifest.is_file() or graph.is_file() or memory.is_dir()


def _print_eval_summary(export_dir: Path) -> None:
    from emet.memory.graph_eqa.dynagraph_eval import compute_dynagraph_eval

    metrics = compute_dynagraph_eval(export_dir)
    print("\n=== eval metrics ===", flush=True)
    for section in ("explore", "graph", "fusion", "gt"):
        block = metrics.get(section)
        if isinstance(block, dict):
            print(f"[{section}]", flush=True)
            for key, val in sorted(block.items()):
                print(f"  {key}: {val}", flush=True)
    report = export_dir / "gt_alignment_report.txt"
    if report.is_file():
        print(f"\nGT alignment report: {report}", flush=True)


def main() -> int:
    args = _parse_args()
    export_dir = args.export_dir.resolve()
    cpu_only = not args.gpu

    _prepare_export_dir(export_dir, clean=not args.resume)
    runner_log = export_dir / "runner.log"
    dyn_log = export_dir / "dynagraph.log"

    from emet.config.sim_launch_config import SimLaunchRobocasa, load_sim_launch_config_from_path
    from emet.eval.dynamic_exploration_runner import _dynagraph_subprocess_timeout_s
    from emet.eval.sim_eval_session import benchmark_sim_server

    sim_cfg = load_sim_launch_config_from_path(args.sim_config)
    if not isinstance(sim_cfg, SimLaunchRobocasa):
        print(f"Expected SimLaunchRobocasa config, got {type(sim_cfg)}", file=sys.stderr)
        return 2
    sim_cfg = replace(
        sim_cfg,
        seed=int(args.seed),
        robot="stretch",
        port_offset=int(args.port_offset),
        headless=True,
    )

    dyn_cmd = _build_dynagraph_cmd(
        export_dir=export_dir,
        port_offset=int(args.port_offset),
        explore_max_iters=int(args.explore_max_iters),
        cpu_only=cpu_only,
        no_sensor_perception=bool(args.no_sensor_perception),
    )

    with runner_log.open("w", encoding="utf-8") as runner_f:
        runner_f.write(f"Export dir: {export_dir}\n")
        runner_f.write(f"Dynagraph: {' '.join(dyn_cmd)}\n")
        runner_f.flush()
        print(f"Export dir: {export_dir}", flush=True)
        print(f"Logs: {runner_log}, {dyn_log}", flush=True)

        try:
            with benchmark_sim_server(sim_cfg, repo=REPO, cpu_only=cpu_only, cwd=REPO) as sim:
                dyn_timeout = _dynagraph_subprocess_timeout_s(
                    explore_max_iters=int(args.explore_max_iters),
                    sim_kind="robocasa",
                    cpu_only=cpu_only,
                )
                with dyn_log.open("w", encoding="utf-8") as log_f:
                    proc = subprocess.run(
                        dyn_cmd,
                        cwd=str(REPO),
                        env=sim.env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=dyn_timeout,
                        check=False,
                    )
        except subprocess.TimeoutExpired:
            msg = f"dynagraph timed out after {dyn_timeout:.0f}s"
            runner_f.write(msg + "\n")
            print(msg, file=sys.stderr)
            return 1
        except Exception as exc:
            runner_f.write(f"benchmark failed: {exc}\n")
            print(f"benchmark failed: {exc}", file=sys.stderr)
            return 1

        combined = dyn_log.read_text(encoding="utf-8", errors="replace")
        tail = combined[-8000:]
        runner_f.write(tail)
        print(tail, flush=True)

        if proc.returncode != 0:
            msg = f"dynagraph exited {proc.returncode}"
            runner_f.write(msg + "\n")
            print(msg, file=sys.stderr)
            return proc.returncode

        if not _export_ok(export_dir):
            msg = f"export missing under {export_dir}"
            runner_f.write(msg + "\n")
            print(msg, file=sys.stderr)
            return 1

    try:
        _print_eval_summary(export_dir)
    except Exception as exc:
        print(f"eval summary failed (export ok): {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
