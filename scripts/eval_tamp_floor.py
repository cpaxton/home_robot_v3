#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""TAMP floor pick/place experiment suite (RoboCasa floor objects).

Runs the RoboCasa floor episodes in ``configs/ovmm/full_episodes.yaml``
(``floor_object: true``) using each episode's configured manipulation mode:
MCTS / distance-heuristic pick-place, teleport reference, or find-only.
Pass ``--manip-mode`` to override the mode for every selected episode.

Each episode: launch a RoboCasa MuJoCo server, drop the GT object to the floor,
run find-phase localization, then Pick+Place. MCTS mode drives the real arm
(kinematic) via ``plan_pick_place_mcts``; sim/oracle teleport the object.

Prefer running this as an ``emet jobs`` job (never block an agent turn on sim)::

  NEED_MIB=8000 uv run emet jobs run --name tamp-floor-suite --need-mib 8000 -- \\
    uv run python scripts/eval_tamp_floor.py

Results: JSON under --output-dir (default ~/runs/emet/tamp_floor).

Smoke (fast agent integration check, ~5–8 min on rby1 — no Stretch head sweeps)::

  uv run python scripts/eval_tamp_floor.py --smoke

Full matrix (overnight / paper; includes slow Stretch + find-only explore episodes)::

  uv run python scripts/eval_tamp_floor.py --backend dynagraph
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = REPO / "configs" / "ovmm" / "full_episodes.yaml"
# Galaxea rby1 kinematic MCTS — fast enough for routine gates (~5 min). Avoid Stretch
# in agentic OVMM smokes: head sweeps are 15–45s each vs ~1–2s on rby1/sourccey.
SMOKE_FLOOR_EPISODE_ID = "robocasa_rby1_floor_to_counter_mcts"

from emet.eval.ovmm_batch import MANIP_MODES, OvmmBatchOptions, run_ovmm_batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=str,
        default=str(DEFAULT_EPISODES),
        help="YAML episode registry (default: configs/ovmm/full_episodes.yaml)",
    )
    parser.add_argument(
        "--manip-mode",
        choices=MANIP_MODES,
        default=None,
        help="Override per-episode mode (default: use manip_mode from the episode YAML)",
    )
    parser.add_argument("--backend", action="append", dest="backends", default=None)
    parser.add_argument("--tier", action="append", default=None)
    parser.add_argument("--episode-id", action="append", dest="episode_ids", default=None)
    parser.add_argument("--all-episodes", action="store_true", help="Run all episodes (default: floor-only)")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--not-rotate", action="store_true")
    parser.add_argument("--sensor-perception", action="store_true")
    parser.add_argument("--port-offset", type=int, default=int(os.getpid() % 400 + 240))
    parser.add_argument("--port-stride", type=int, default=2)
    parser.add_argument("--benchmark", type=str, default="configs/ovmm/benchmark.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Fast gate: single dynagraph episode {SMOKE_FLOOR_EPISODE_ID!r} "
            "(rby1 MCTS floor pick; no Stretch agentic sweeps)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    episode_ids = list(args.episode_ids) if args.episode_ids else None
    backends = list(args.backends) if args.backends else ["dynagraph"]
    if args.smoke:
        if episode_ids and episode_ids != [SMOKE_FLOOR_EPISODE_ID]:
            raise SystemExit(
                f"--smoke fixes episode to {SMOKE_FLOOR_EPISODE_ID!r}; do not also pass --episode-id {episode_ids!r}"
            )
        episode_ids = [SMOKE_FLOOR_EPISODE_ID]
        backends = ["dynagraph"]
    opts = OvmmBatchOptions(
        episodes=args.episodes,
        backends=backends,
        tiers=args.tier,
        episode_ids=episode_ids,
        floor_only=not args.all_episodes,
        merge_xy_m=None,
        staleness_horizon=None,
        compare_to_gt=False,
        cpu_only=args.cpu_only,
        sensor_perception=args.sensor_perception,
        graph_query=False,
        not_rotate=args.not_rotate,
        no_perfect_depth=False,
        port_offset=args.port_offset,
        port_stride=args.port_stride,
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        manip_mode=args.manip_mode,
        full=True,
    )
    return run_ovmm_batch(opts, repo_root=REPO)


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
