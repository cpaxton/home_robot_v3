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

"""Multi-seed replication for OVMM find-phase sim benchmarks (audit variability)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = REPO / "configs" / "ovmm" / "find_phase_episodes.yaml"

BACKENDS = ("dynamem", "graph_eqa", "dynagraph", "ground_truth")

SUMMARY_NUMERIC_KEYS = (
    "find_object_success",
    "find_recep_success",
    "find_partial_success",
    "localization_err_obj_m",
    "localization_err_recep_m",
    "n_graph_nodes",
    "n_voxel_explored_cells",
    "n_voxel_explored_area_m2",
    "episode_wall_s",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate OVMM find-phase runs across RNG seeds (writes pred_xyz per run).",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=str(DEFAULT_EPISODES),
        help="YAML episode registry",
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
    parser.add_argument("--replicates", type=int, default=5, help="Number of seeds to run (default: 5)")
    parser.add_argument("--seed-base", type=int, default=0, help="First seed value (default: 0)")
    parser.add_argument("--merge-xy-m", type=float, default=None)
    parser.add_argument("--staleness-horizon", type=int, default=None)
    parser.add_argument("--compare-to-gt", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--not-rotate", action="store_true")
    parser.add_argument(
        "--no-perfect-depth",
        action="store_true",
        help="Disable sim perfect sensor depth",
    )
    parser.add_argument(
        "--port-offset-base",
        type=int,
        default=int(os.getpid() % 400 + 140),
        help="Base ZMQ port offset; each replicate uses base + seed * 2",
    )
    parser.add_argument("--benchmark", type=str, default="configs/ovmm/benchmark.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Parent dir for seed_* subdirs")
    parser.add_argument("--dry-run", action="store_true", help="List planned runs and exit")
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


def _seed_values(seed_base: int, replicates: int) -> list[int]:
    return [seed_base + i for i in range(replicates)]


def _summarize_runs(runs: list[dict]) -> dict[str, dict[str, float | int | None]]:
    """Mean/std/min/max for numeric metrics across replicate JSON rows."""
    summary: dict[str, dict[str, float | int | None]] = {}
    for key in SUMMARY_NUMERIC_KEYS:
        values = [float(row[key]) for row in runs if row.get(key) is not None and row.get("error") is None]
        if not values:
            continue
        summary[key] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def main() -> int:
    from emet.eval.ovmm_benchmark_config import load_ovmm_benchmark_config
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, load_find_phase_episodes, run_episode_find_phase

    args = _parse_args()
    bench = load_ovmm_benchmark_config(args.benchmark)
    backends = args.backends or ["dynagraph"]
    episodes_path = args.episodes
    if episodes_path == str(DEFAULT_EPISODES):
        episodes_path = str(bench.sim_episodes_yaml)
    episodes = load_find_phase_episodes(episodes_path)
    episodes = _filter_episodes(episodes, tiers=args.tier, episode_ids=args.episode_ids)
    seeds = _seed_values(args.seed_base, args.replicates)

    if args.dry_run:
        for seed in seeds:
            port = args.port_offset_base + seed * 2
            for backend in backends:
                for ep in episodes:
                    print(f"seed={seed}\tport={port}\t{ep.id}\t{backend}")
        return 0

    output_root = Path(args.output_dir or bench.paths.output_dir_sim) / "replicates"
    output_root.mkdir(parents=True, exist_ok=True)
    per_run_rows: list[dict] = []

    for seed in seeds:
        port_offset = int(args.port_offset_base + seed * 2)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_runs: list[dict] = []

        for backend in backends:
            for ep in episodes:
                run_cfg = FindPhaseRunConfig(
                    backend=backend,
                    merge_xy_m=args.merge_xy_m,
                    staleness_horizon=args.staleness_horizon,
                    compare_to_gt=args.compare_to_gt,
                    cpu_only=args.cpu_only,
                    port_offset=port_offset,
                    not_rotate=args.not_rotate,
                    perfect_depth=not args.no_perfect_depth,
                    seed=seed,
                )
                tag = f"{ep.id}_{backend}"
                print(f"seed={seed} port={port_offset} Running {tag} …", file=sys.stderr)
                try:
                    metrics = run_episode_find_phase(ep, run_cfg, repo_root=REPO)
                except Exception as exc:
                    print(f"FAIL seed={seed} {tag}: {exc}", file=sys.stderr)
                    metrics = {
                        "episode_id": ep.id,
                        "tier": ep.tier,
                        "backend": backend,
                        "seed": seed,
                        "port_offset": port_offset,
                        "error": str(exc),
                        "find_object_success": False,
                        "find_recep_success": False,
                        "find_partial_success": 0.0,
                        "pred_obj_xyz": None,
                        "pred_recep_xyz": None,
                    }
                else:
                    metrics["port_offset"] = port_offset

                out_json = seed_dir / f"{tag}.json"
                out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                seed_runs.append(metrics)
                print(json.dumps({k: metrics[k] for k in sorted(metrics) if k != "error"}, indent=2))

        summary_path = seed_dir / "summary.json"
        summary_path.write_text(
            json.dumps({"seed": seed, "runs": seed_runs, "stats": _summarize_runs(seed_runs)}, indent=2),
            encoding="utf-8",
        )
        per_run_rows.extend(seed_runs)

    by_group: dict[tuple[str, str], list[dict]] = {}
    for row in per_run_rows:
        key = (str(row.get("episode_id")), str(row.get("backend")))
        by_group.setdefault(key, []).append(row)

    cross_seed: list[dict] = []
    for (episode_id, backend), runs in sorted(by_group.items()):
        cross_seed.append(
            {
                "episode_id": episode_id,
                "backend": backend,
                "tier": runs[0].get("tier"),
                "seeds": [r.get("seed") for r in runs],
                "n_replicates": len(runs),
                "stats": _summarize_runs(runs),
            }
        )

    aggregate_path = output_root / "aggregate_replicates.json"
    aggregate_path.write_text(json.dumps(cross_seed, indent=2), encoding="utf-8")
    print(f"Wrote replicates under {output_root} (aggregate: {aggregate_path})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
