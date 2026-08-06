# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared OVMM find/full batch runners (used by ``emet ovmm`` and scripts/)."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emet.eval.memory_backends import OVMM_MEMORY_BACKENDS

BACKENDS = OVMM_MEMORY_BACKENDS
MANIP_MODES = ("skip", "oracle", "sim", "attempt")


@dataclass
class OvmmBatchOptions:
    """Common knobs for find-phase / full OVMM batch runs."""

    episodes: str
    backends: Sequence[str] | None = None
    tiers: Sequence[str] | None = None
    episode_ids: Sequence[str] | None = None
    merge_xy_m: float | None = None
    staleness_horizon: int | None = None
    compare_to_gt: bool = False
    cpu_only: bool = False
    sensor_perception: bool = False
    graph_query: bool = False
    not_rotate: bool = False
    no_perfect_depth: bool = False
    port_offset: int = 140
    port_stride: int = 2
    benchmark: str = "configs/ovmm/benchmark.yaml"
    output_dir: str | Path | None = None
    dry_run: bool = False
    explore_steps: int | None = None
    no_scene_cache: bool = False
    # None → dynagraph/static_graph use shared AgenticEQA find; True/False override.
    agentic_find: bool | None = None
    # Ablation: force one-shot localize (sets agentic_find=False).
    oneshot_localize: bool = False
    agentic_max_rounds: int | None = None
    agentic_max_nav_steps: int | None = None
    manip_mode: str = "skip"
    full: bool = False


def filter_episodes(episodes: list[Any], *, tiers: Sequence[str] | None, episode_ids: Sequence[str] | None) -> list[Any]:
    out = episodes
    if tiers:
        tier_set = {t.strip() for t in tiers}
        out = [e for e in out if e.tier in tier_set]
    if episode_ids:
        id_set = {i.strip() for i in episode_ids}
        out = [e for e in out if e.id in id_set]
    return out


def write_metrics_csv(rows: list[dict], path: Path) -> None:
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


def run_ovmm_batch(opts: OvmmBatchOptions, *, repo_root: Path | None = None) -> int:
    """Run find or full OVMM episodes; write per-run JSON + aggregate CSV."""
    from emet.eval.ovmm_benchmark_config import load_ovmm_benchmark_config
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, load_find_phase_episodes, run_episode_find_phase

    root = repo_root or Path(__file__).resolve().parents[3]
    bench = load_ovmm_benchmark_config(opts.benchmark)
    backends = list(opts.backends) if opts.backends else ["dynagraph"]
    episodes_path = str(opts.episodes)
    default_find = str(root / "configs" / "ovmm" / "find_phase_episodes.yaml")
    default_full = str(root / "configs" / "ovmm" / "full_episodes.yaml")
    if episodes_path in (default_find, "configs/ovmm/find_phase_episodes.yaml") and not opts.full:
        episodes_path = str(bench.sim_episodes_yaml)
    if episodes_path in (default_full, "configs/ovmm/full_episodes.yaml") and opts.full:
        episodes_path = str(bench.full_episodes_yaml)

    episodes = load_find_phase_episodes(episodes_path)
    episodes = filter_episodes(episodes, tiers=opts.tiers, episode_ids=opts.episode_ids)

    if opts.dry_run:
        for ep in episodes:
            extra = f"\tmanip={opts.manip_mode}" if opts.full else ""
            print(f"{ep.id}\t{ep.tier}\t{ep.sim}\tbackends={backends}{extra}")
        return 0

    if opts.output_dir is not None:
        output_dir = Path(opts.output_dir).expanduser().resolve()
    else:
        output_dir = bench.paths.output_dir_full if opts.full else bench.paths.output_dir_sim
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    stride = max(1, int(opts.port_stride))
    manip = str(opts.manip_mode) if opts.full else "skip"
    for backend in backends:
        for ep_i, ep in enumerate(episodes):
            port_offset = int(opts.port_offset) + ep_i * stride
            agentic = False if opts.oneshot_localize else opts.agentic_find
            run_cfg = FindPhaseRunConfig(
                backend=backend,
                merge_xy_m=opts.merge_xy_m,
                staleness_horizon=opts.staleness_horizon,
                compare_to_gt=opts.compare_to_gt,
                cpu_only=opts.cpu_only,
                port_offset=port_offset,
                not_rotate=opts.not_rotate,
                perfect_depth=not opts.no_perfect_depth,
                use_sensor_perception=opts.sensor_perception,
                prefer_voxel=not opts.graph_query,
                agentic_find=agentic,
                agentic_max_rounds=opts.agentic_max_rounds,
                agentic_max_nav_steps=opts.agentic_max_nav_steps,
                explore_steps_override=opts.explore_steps,
                use_scene_cache=not opts.no_scene_cache,
                manip_mode=manip,
            )
            tag = f"{ep.id}_{backend}"
            label = f"Running {tag}" + (f" manip_mode={manip}" if opts.full else "") + " …"
            print(label, file=sys.stderr)
            try:
                metrics = run_episode_find_phase(ep, run_cfg, repo_root=root)
            except Exception as exc:
                print(f"FAIL {tag}: {exc}", file=sys.stderr)
                metrics = {
                    "episode_id": ep.id,
                    "tier": ep.tier,
                    "backend": backend,
                    "error": str(exc),
                    "find_object_success": False,
                    "find_recep_success": False,
                    "find_partial_success": 0.0,
                }
                if opts.full:
                    metrics.update(
                        {
                            "manip_mode": manip,
                            "pick_success": False,
                            "place_success": False,
                            "ovmm_full_success": False,
                            "ovmm_full_partial": 0.0,
                        }
                    )
            out_json = output_dir / f"{tag}.json"
            out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            all_rows.append(metrics)
            print(json.dumps({k: metrics[k] for k in sorted(metrics) if k != "error"}, indent=2))

    stamp = "-".join(backends)
    csv_path = output_dir / f"aggregate_{stamp}.csv"
    write_metrics_csv(all_rows, csv_path)
    print(f"Wrote {len(all_rows)} runs to {output_dir} (CSV: {csv_path})", file=sys.stderr)
    return 0
