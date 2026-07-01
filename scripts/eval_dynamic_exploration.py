#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Batch dynamic exploration benchmark (Robocasa + MolmoSpaces, Stretch)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO / "configs" / "benchmarks" / "dynamic_exploration.yaml"


def _parse_args() -> argparse.Namespace:
    from emet.eval.benchmark_dynagraph import DYNAMIC_EXPLORE_BACKENDS

    parser = argparse.ArgumentParser(
        description="Dynamic exploration eval: active explore-loop (Phase 1) or world-change (Phase 2).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK),
        help="Benchmark YAML (default: configs/benchmarks/dynamic_exploration.yaml)",
    )
    parser.add_argument(
        "--phase",
        choices=("explore", "world-change", "lifelong"),
        default="explore",
        help=(
            "explore: Phase 1 matrix; world-change: Phase 2 single relocation; "
            "lifelong: K-cycle checkpoint/fuzz/reload episodes"
        ),
    )
    parser.add_argument(
        "--env",
        choices=("robocasa", "molmospaces", "all"),
        default="all",
        help="Filter episodes by environment",
    )
    parser.add_argument("--seed", type=int, default=None, help="Filter Robocasa seed / Molmo index")
    parser.add_argument(
        "--episode-id",
        action="append",
        dest="episode_ids",
        help="Run only these episode ids (repeatable)",
    )
    parser.add_argument(
        "--backend",
        choices=DYNAMIC_EXPLORE_BACKENDS,
        action="append",
        dest="backends",
        help="Memory backend row (default: dynagraph)",
    )
    parser.add_argument(
        "--explore-max-iters",
        type=int,
        action="append",
        dest="explore_max_iters_list",
        help="Explore budget K (repeatable; default from benchmark YAML)",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=("explore", "rotate_only", "both"),
        default="both",
        help="Phase 1 mapping: explore-loop, rotate-only baseline, or both",
    )
    parser.add_argument("--resume", action="store_true", help="Skip runs whose JSON already exists")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU for perception models")
    parser.add_argument(
        "--skip-eqa",
        action="store_true",
        help="Phase 1 only: run explore-loop + export without question-bank VLM EQA",
    )
    parser.add_argument(
        "--port-offset-base",
        type=int,
        default=int(os.getpid() % 400 + 160),
        help="Base ZMQ port offset (incremented per run)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default from benchmark YAML)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List planned runs and exit")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Single short run from benchmark YAML ``smoke:`` block (Phase 1 explore by default)",
    )
    return parser.parse_args()


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


def _rows_have_errors(rows: list[dict]) -> bool:
    return any(str(row.get("error") or "").strip() for row in rows)


def main() -> int:
    from emet.eval.dynamic_exploration_config import (
        build_explore_run_matrix,
        filter_episodes,
        load_dynamic_exploration_config,
        resolve_smoke_run_plan,
    )
    from emet.eval.dynamic_exploration_runner import (
        DynamicExploreRunConfig,
        run_explore_episode_subprocess,
        run_lifelong_episode,
        run_world_change_episode,
    )

    args = _parse_args()
    cfg = load_dynamic_exploration_config(args.benchmark)

    if args.smoke:
        smoke = resolve_smoke_run_plan(cfg)
        args.phase = smoke.phase  # type: ignore[assignment]
        args.episode_ids = [smoke.episode_id]
        args.backends = [smoke.backend]
        args.explore_max_iters_list = [smoke.explore_max_iters]
        args.mapping_mode = smoke.mapping_mode  # type: ignore[assignment]
        if args.env == "all":
            ep_match = next((e for e in cfg.episodes if e.id == smoke.episode_id), None)
            if ep_match is not None:
                args.env = ep_match.env  # type: ignore[assignment]

    backends = args.backends or ["dynagraph"]
    env_filter = None if args.env == "all" else args.env
    episodes = filter_episodes(
        cfg.episodes,
        env=env_filter,
        episode_ids=args.episode_ids,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir or cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "explore":
        include_rotate = args.mapping_mode in ("rotate_only", "both")
        mapping_modes: list[str] = []
        if args.mapping_mode in ("explore", "both"):
            mapping_modes.append("explore")
        if include_rotate:
            mapping_modes.append("rotate_only")

        runs = build_explore_run_matrix(
            cfg,
            episodes,
            backends=backends,
            explore_max_iters=args.explore_max_iters_list,
            mapping_modes=mapping_modes,  # type: ignore[arg-type]
            include_rotate_only=False,
        )
        if args.dry_run:
            for run in runs:
                print(f"{run.run_id}\t{run.episode.env}\tbackend={run.backend}")
            return 0

        all_rows: list[dict] = []
        for i, run in enumerate(runs):
            run_cfg = DynamicExploreRunConfig(
                backend=run.backend,
                cpu_only=args.cpu_only,
                port_offset=args.port_offset_base + i,
                resume=args.resume,
                skip_eqa=args.skip_eqa,
            )
            print(f"Running {run.run_id} …", file=sys.stderr)
            payload = run_explore_episode_subprocess(run, run_cfg, cfg, output_dir=output_dir, repo_root=REPO)
            row = dict(payload.get("summary") or {})
            if payload.get("metrics", {}).get("error"):
                row["error"] = payload["metrics"]["error"]
            all_rows.append(row)

        csv_path = output_dir / "aggregate_dynamic_exploration.csv"
        _write_csv(all_rows, csv_path)
        print(f"Wrote {len(all_rows)} runs to {output_dir} (CSV: {csv_path})", file=sys.stderr)
        if _rows_have_errors(all_rows):
            print("One or more explore runs failed (see CSV error column).", file=sys.stderr)
            return 1
        return 0

    if args.phase == "lifelong":
        ll_episodes = cfg.lifelong_episodes
        if args.episode_ids:
            id_set = {x.strip() for x in args.episode_ids}
            ll_episodes = tuple(e for e in ll_episodes if e.id in id_set or e.episode_id in id_set)
        if env_filter:
            base_envs = {e.id: e.env for e in cfg.episodes}
            ll_episodes = tuple(e for e in ll_episodes if base_envs.get(e.episode_id) == env_filter)

        ep_by_id = {e.id: e for e in cfg.episodes}
        if args.dry_run:
            for le in ll_episodes:
                for backend in backends:
                    print(f"{le.id}_{backend}\tlifelong\tbase={le.episode_id}\tcycles={le.cycles}")
            return 0

        all_rows: list[dict] = []
        for i, le in enumerate(ll_episodes):
            base = ep_by_id.get(le.episode_id)
            if base is None:
                print(f"SKIP {le.id}: unknown base episode {le.episode_id}", file=sys.stderr)
                continue
            for j, backend in enumerate(backends):
                run_cfg = DynamicExploreRunConfig(
                    backend=backend,
                    cpu_only=args.cpu_only,
                    port_offset=args.port_offset_base + i * len(backends) + j,
                    resume=args.resume,
                )
                tag = f"{le.id}_{backend}"
                print(f"Running {tag} (lifelong, {le.cycles} cycles) …", file=sys.stderr)
                payload = run_lifelong_episode(
                    le,
                    base,
                    run_cfg,
                    cfg,
                    output_dir=output_dir,
                    repo_root=REPO,
                )
                for cyc in payload.get("cycle_results") or []:
                    churn = cyc.get("moved_body_churn") or []
                    all_rows.append(
                        {
                            "run_id": payload.get("run_id"),
                            "episode_id": payload.get("episode_id"),
                            "backend": payload.get("backend"),
                            "cycle": cyc.get("cycle"),
                            "eqa_accuracy": cyc.get("eqa_accuracy"),
                            "object_node_count": cyc.get("object_node_count"),
                            "total_node_count": cyc.get("total_node_count"),
                            "n_moves_adapted": sum(1 for m in churn if m.get("adapted")),
                            "n_moves_stale": sum(1 for m in churn if m.get("stale")),
                            "cycle_wall_s": cyc.get("cycle_wall_s"),
                            "error": payload.get("error"),
                        }
                    )

        csv_path = output_dir / "aggregate_dynamic_exploration_lifelong.csv"
        _write_csv(all_rows, csv_path)
        print(f"Wrote {len(all_rows)} lifelong cycle rows to {output_dir} (CSV: {csv_path})", file=sys.stderr)
        if _rows_have_errors(all_rows):
            print("One or more lifelong runs failed (see CSV error column).", file=sys.stderr)
            return 1
        return 0

    # Phase 2 world-change
    wc_episodes = cfg.world_change_episodes
    if args.episode_ids:
        id_set = {x.strip() for x in args.episode_ids}
        wc_episodes = tuple(w for w in wc_episodes if w.id in id_set or w.episode_id in id_set)

    ep_by_id = {e.id: e for e in cfg.episodes}
    if args.dry_run:
        for wc in wc_episodes:
            for backend in backends:
                print(f"{wc.id}_{backend}\tworld-change\tbase={wc.episode_id}")
        return 0

    all_rows: list[dict] = []
    explore_k = (args.explore_max_iters_list or [15])[0]
    for i, wc in enumerate(wc_episodes):
        base = ep_by_id.get(wc.episode_id)
        if base is None:
            print(f"SKIP {wc.id}: unknown base episode {wc.episode_id}", file=sys.stderr)
            continue
        for j, backend in enumerate(backends):
            run_cfg = DynamicExploreRunConfig(
                backend=backend,
                cpu_only=args.cpu_only,
                port_offset=args.port_offset_base + i * len(backends) + j,
                resume=args.resume,
            )
            tag = f"{wc.id}_{backend}"
            print(f"Running {tag} (world-change) …", file=sys.stderr)
            payload = run_world_change_episode(
                wc,
                base,
                run_cfg,
                cfg,
                explore_max_iters=int(explore_k),
                output_dir=output_dir,
                repo_root=REPO,
            )
            all_rows.append(payload)

    csv_path = output_dir / "aggregate_dynamic_exploration_world_change.csv"
    _write_csv(all_rows, csv_path)
    print(f"Wrote {len(all_rows)} world-change runs to {output_dir} (CSV: {csv_path})", file=sys.stderr)
    if _rows_have_errors(all_rows):
        print("One or more world-change runs failed (see CSV error column).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
