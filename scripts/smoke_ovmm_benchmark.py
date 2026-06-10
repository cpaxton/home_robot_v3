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
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Smoke OVMM find-phase benchmarks (unit tests + one sim + one Habitat episode)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HABITAT_BIN = REPO / ".venv-habitat" / "bin" / "emet-habitat"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OVMM benchmark smoke tests.")
    parser.add_argument("--benchmark", default="configs/ovmm/benchmark.yaml")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--skip-habitat", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", default=True)
    return parser.parse_args()


def _run(cmd: list[str], *, timeout: float | None = None) -> int:
    print("$", " ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd, cwd=REPO, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("TIMEOUT", file=sys.stderr)
        return 124


def main() -> int:
    from emet.eval.ovmm_benchmark_config import load_ovmm_benchmark_config
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, load_find_phase_episodes, run_episode_find_phase

    args = _parse_args()
    cfg = load_ovmm_benchmark_config(args.benchmark)
    rc = 0

    if not args.skip_unit:
        rc = max(
            rc,
            _run(
                [
                    "uv",
                    "run",
                    "emet",
                    "test",
                    "src/test/memory/test_ovmm_find_phase_metrics.py",
                    "src/test/memory/test_habitat_ovmm_find_loader.py",
                    "-q",
                ],
                timeout=120.0,
            ),
        )

    out_sim = cfg.paths.output_dir_sim / "smoke"
    out_sim.mkdir(parents=True, exist_ok=True)
    if not args.skip_sim:
        episodes = load_find_phase_episodes(cfg.sim_episodes_yaml)
        ep = next(e for e in episodes if e.id == cfg.smoke_sim_episode_id)
        run_cfg = FindPhaseRunConfig(
            backend=cfg.smoke_sim_backend,  # type: ignore[arg-type]
            cpu_only=args.cpu_only,
            not_rotate=True,
            port_offset=int(__import__("os").getpid() % 400 + 220),
        )
        print(f"sim smoke: {ep.id} backend={cfg.smoke_sim_backend}", flush=True)
        try:
            metrics = run_episode_find_phase(ep, run_cfg, repo_root=REPO)
        except Exception as exc:
            print(f"sim smoke FAIL: {exc}", file=sys.stderr)
            rc = 1
        else:
            out_json = out_sim / f"{ep.id}_{cfg.smoke_sim_backend}.json"
            out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            ok = metrics.get("find_partial_success", 0) >= 1.0
            print(f"sim smoke partial={metrics.get('find_partial_success')} -> {out_json}")
            if not ok:
                rc = 1

    out_hab = cfg.paths.output_dir_habitat / "smoke"
    out_hab.mkdir(parents=True, exist_ok=True)
    if not args.skip_habitat:
        if not HABITAT_BIN.is_file():
            print("habitat smoke SKIP: run ./scripts/install_habitat.sh", file=sys.stderr)
            rc = max(rc, 1)
        else:
            cmd = [
                str(HABITAT_BIN),
                "run-ovmm-find-episode",
                "--episodes",
                str(cfg.habitat_episodes_yaml),
                "--episode-id",
                cfg.smoke_habitat_episode_id,
                "--backend",
                cfg.smoke_habitat_backend,
                "--not-rotate",
                "--output",
                str(out_hab / f"{cfg.smoke_habitat_episode_id}_{cfg.smoke_habitat_backend}.json"),
            ]
            if args.cpu_only:
                cmd.append("--cpu-only")
            hab_rc = _run(cmd, timeout=300.0)
            rc = max(rc, hab_rc)

    print(f"smoke done rc={rc} (outputs under {cfg.paths.output_dir_sim} and {cfg.paths.output_dir_habitat})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
