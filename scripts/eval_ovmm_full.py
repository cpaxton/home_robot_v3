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

"""Batch full OVMM benchmark (FindObj + Pick + FindRec + Place) in emet sim."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = REPO / "configs" / "ovmm" / "full_episodes.yaml"

from emet.eval.memory_backends import OVMM_MEMORY_BACKENDS

BACKENDS = OVMM_MEMORY_BACKENDS
MANIP_MODES = ("skip", "oracle", "sim", "attempt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full OVMM (find + pick/place) in emet sim.",
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
        default="oracle",
        help="Pick/place: skip | oracle (GT from find) | sim (MuJoCo teleport) | attempt (AnyGrasp or sim fallback)",
    )
    parser.add_argument("--merge-xy-m", type=float, default=None)
    parser.add_argument("--staleness-horizon", type=int, default=None)
    parser.add_argument("--compare-to-gt", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--sensor-perception", action="store_true")
    parser.add_argument("--graph-query", action="store_true")
    parser.add_argument("--not-rotate", action="store_true")
    parser.add_argument("--no-perfect-depth", action="store_true")
    parser.add_argument("--port-offset", type=int, default=int(os.getpid() % 400 + 140),
                        help="Base ZMQ port offset; each episode uses base + index * --port-stride")
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


def _filter_episodes(episodes, *, tiers, episode_ids):
    out = episodes
    if tiers:
        tier_set = {t.strip() for t in tiers}
        out = [e for e in out if e.tier in tier_set]
    if episode_ids:
        id_set = {i.strip() for i in episode_ids}
        out = [e for e in out if e.id in id_set]
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main() -> int:
    from emet.eval.ovmm_benchmark_config import load_ovmm_benchmark_config
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, load_find_phase_episodes, run_episode_find_phase

    args = _parse_args()
    bench = load_ovmm_benchmark_config(args.benchmark)
    backends = args.backends or ["dynagraph"]
    episodes_path = args.episodes
    if episodes_path == str(DEFAULT_EPISODES):
        episodes_path = str(bench.full_episodes_yaml)
    episodes = load_find_phase_episodes(episodes_path)
    episodes = _filter_episodes(episodes, tiers=args.tier, episode_ids=args.episode_ids)

    if args.dry_run:
        for ep in episodes:
            print(f"{ep.id}\t{ep.tier}\t{ep.sim}\tmanip={args.manip_mode}\tbackends={backends}")
        return 0

    output_dir = Path(args.output_dir or bench.paths.output_dir_full)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    stride = max(1, int(args.port_stride))
    for backend in backends:
        for ep_i, ep in enumerate(episodes):
            port_offset = int(args.port_offset) + ep_i * stride
            run_cfg = FindPhaseRunConfig(
                backend=backend,
                merge_xy_m=args.merge_xy_m,
                staleness_horizon=args.staleness_horizon,
                compare_to_gt=args.compare_to_gt,
                cpu_only=args.cpu_only,
                port_offset=port_offset,
                not_rotate=args.not_rotate,
                perfect_depth=not args.no_perfect_depth,
                use_sensor_perception=args.sensor_perception,
                prefer_voxel=not args.graph_query,
                manip_mode=args.manip_mode,
            )
            tag = f"{ep.id}_{backend}"
            print(f"Running {tag} manip_mode={args.manip_mode} …", file=sys.stderr)
            try:
                metrics = run_episode_find_phase(ep, run_cfg, repo_root=REPO)
            except Exception as exc:
                print(f"FAIL {tag}: {exc}", file=sys.stderr)
                metrics = {
                    "episode_id": ep.id,
                    "tier": ep.tier,
                    "backend": backend,
                    "manip_mode": args.manip_mode,
                    "error": str(exc),
                    "find_object_success": False,
                    "find_recep_success": False,
                    "find_partial_success": 0.0,
                    "pick_success": False,
                    "place_success": False,
                    "ovmm_full_success": False,
                    "ovmm_full_partial": 0.0,
                }
            out_json = output_dir / f"{tag}.json"
            out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            all_rows.append(metrics)
            print(json.dumps({k: metrics[k] for k in sorted(metrics) if k != "error"}, indent=2))

    stamp = "-".join(backends)
    csv_path = output_dir / f"aggregate_{stamp}.csv"
    _write_csv(all_rows, csv_path)
    print(f"Wrote {len(all_rows)} runs to {output_dir} (CSV: {csv_path})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
