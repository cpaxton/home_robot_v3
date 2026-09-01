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
# Prefer: ``uv run emet ovmm find --episodes …`` (this script remains a thin argparse wrapper).

"""Batch OVMM find-phase benchmark (FindObj / FindRec) across memory backends and scene tiers.

Prefer the first-class CLI::

    uv run emet ovmm find --episodes OUT/find_episodes.yaml --backend dynagraph
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = REPO / "configs" / "ovmm" / "find_phase_episodes.yaml"

from emet.eval.memory_backends import OVMM_MEMORY_BACKENDS
from emet.eval.ovmm_batch import OvmmBatchOptions, run_ovmm_batch

BACKENDS = OVMM_MEMORY_BACKENDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OVMM-inspired find phases (FindObj / FindRec) in emet sim. Prefer: emet ovmm find …",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=str(DEFAULT_EPISODES),
        help="YAML episode registry (default: configs/ovmm/find_phase_episodes.yaml)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        action="append",
        dest="backends",
        help="Memory backend (repeat for multiple; default: dynagraph)",
    )
    parser.add_argument("--tier", action="append", help="Run only episodes with this tier (S0, S1, S2)")
    parser.add_argument("--episode-id", action="append", dest="episode_ids", help="Run only these episode ids")
    parser.add_argument("--merge-xy-m", type=float, default=None, help="Dynagraph merge radius override (m)")
    parser.add_argument(
        "--staleness-horizon",
        type=int,
        default=None,
        help="Dynagraph staleness horizon override (steps; 0 disables)",
    )
    parser.add_argument("--compare-to-gt", action="store_true", help="Dynagraph: overlay sim GT in Rerun")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU for perception models")
    parser.add_argument(
        "--sensor-perception",
        action="store_true",
        help="Enable per-frame VLM graph labeling (full GraphEQA; slow)",
    )
    parser.add_argument(
        "--graph-query",
        action="store_true",
        help="Query graph memory first (prefer_voxel=False); default is voxel-first",
    )
    parser.add_argument("--not-rotate", action="store_true", help="Skip rotate-in-place mapping")
    parser.add_argument(
        "--mapping-max-nav-steps",
        type=int,
        default=None,
        help="Override episode mapping coverage budget (agentic explore max_nav_steps)",
    )
    parser.add_argument(
        "--explore-steps",
        type=int,
        default=None,
        help="Deprecated alias of --mapping-max-nav-steps (mapping coverage, not FindObj)",
    )
    parser.add_argument(
        "--no-scene-cache",
        action="store_true",
        help="Ignore prebuilt scene map cache (always live rotate/explore)",
    )
    parser.add_argument(
        "--no-perfect-depth",
        action="store_true",
        help="Disable sim perfect sensor depth (default: use ZMQ depth for mapping)",
    )
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
    parser.add_argument(
        "--benchmark",
        type=str,
        default="configs/ovmm/benchmark.yaml",
        help="Benchmark YAML for default output path (~/runs/emet/...)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for per-run JSON and aggregate CSV (default from benchmark.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List episodes and exit")
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
        mapping_max_nav_steps=args.mapping_max_nav_steps,
        explore_steps=args.explore_steps,
        no_scene_cache=args.no_scene_cache,
        full=False,
    )
    return run_ovmm_batch(opts, repo_root=REPO)


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
