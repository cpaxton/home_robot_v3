#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
#
# Prefer: ``uv run emet ovmm full --episodes … --manip-mode sim`` (thin argparse wrapper).

"""Batch full OVMM benchmark (FindObj + Pick + FindRec + Place) in emet sim.

Prefer the first-class CLI::

    uv run emet ovmm full --episodes OUT/full_episodes.yaml --backend dynagraph --manip-mode sim
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = REPO / "configs" / "ovmm" / "full_episodes.yaml"

from emet.eval.memory_backends import OVMM_MEMORY_BACKENDS
from emet.eval.ovmm_batch import MANIP_MODES, OvmmBatchOptions, run_ovmm_batch

BACKENDS = OVMM_MEMORY_BACKENDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full OVMM (find + pick/place) in emet sim. Prefer: emet ovmm full …",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=str(DEFAULT_EPISODES),
        help="YAML episode registry (default: configs/ovmm/full_episodes.yaml)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        action="append",
        dest="backends",
        help="Memory backend (repeat; default: dynagraph)",
    )
    parser.add_argument("--tier", action="append", help="Run only episodes with this tier (S0, S1, S2)")
    parser.add_argument("--episode-id", action="append", dest="episode_ids", help="Run only these episode ids")
    parser.add_argument(
        "--manip-mode",
        choices=MANIP_MODES,
        default=None,
        help="Override per-episode mode; default uses episode YAML, otherwise oracle",
    )
    parser.add_argument("--merge-xy-m", type=float, default=None)
    parser.add_argument("--staleness-horizon", type=int, default=None)
    parser.add_argument("--compare-to-gt", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--sensor-perception", action="store_true")
    parser.add_argument("--graph-query", action="store_true")
    parser.add_argument("--not-rotate", action="store_true")
    parser.add_argument("--no-perfect-depth", action="store_true")
    parser.add_argument(
        "--port-offset",
        type=int,
        default=int(os.getpid() % 400 + 140),
        help="Base ZMQ port offset; each episode uses base + index * --port-stride",
    )
    parser.add_argument(
        "--port-stride",
        type=int,
        default=2,
        help="Add index * stride to --port-offset per episode (avoids bind clashes)",
    )
    parser.add_argument("--benchmark", type=str, default="configs/ovmm/benchmark.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    opts = OvmmBatchOptions(
        episodes=args.episodes,
        backends=args.backends,
        tiers=args.tier,
        episode_ids=args.episode_ids,
        merge_xy_m=args.merge_xy_m,
        staleness_horizon=args.staleness_horizon,
        compare_to_gt=args.compare_to_gt,
        cpu_only=args.cpu_only,
        sensor_perception=args.sensor_perception,
        graph_query=args.graph_query,
        not_rotate=args.not_rotate,
        no_perfect_depth=args.no_perfect_depth,
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
