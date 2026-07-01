# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Aggregate SQA3D episode JSONL sweeps to CSV (paper tables)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from emet.benchmarks.sqa3d.analysis import classify_episodes, summarize_outcomes
from emet.benchmarks.sqa3d.metrics import load_episode_metrics_jsonl


def _split_coverage_fields(
    episodes: list[dict[str, Any]],
    outcomes: dict[str, Any],
    *,
    split: str | None,
    data_dir: Path | None,
    scannet_root: Path | None,
    replay_mode: str,
) -> dict[str, Any]:
    if not split:
        return {}
    from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions
    from emet.benchmarks.sqa3d.scannet.config import (
        default_scannet_root,
        filter_questions_with_scannet,
    )

    all_q = load_sqa3d_questions(split, data_dir=data_dir)
    n_split_total = len(all_q)
    root = scannet_root or default_scannet_root()
    n_runnable = len(filter_questions_with_scannet(all_q, root, replay_mode=replay_mode))
    n_scored = int(outcomes.get("n_scored", 0))
    n_episodes = int(outcomes.get("n_episodes", 0))
    n_exported = sum(1 for e in episodes if str(e.get("export_dir", "")).strip())
    return {
        "split": split,
        "n_split_total": n_split_total,
        "n_runnable": n_runnable,
        "n_attempted": n_episodes,
        "n_exported": n_exported,
        "coverage_frac": (n_scored / n_split_total) if n_split_total else 0.0,
        "runnable_coverage_frac": (n_episodes / n_runnable) if n_runnable else "",
    }


def aggregate_sqa3d_episodes(
    episodes: list[dict[str, Any]],
    *,
    source_jsonl: str | Path | None = None,
    dedupe: bool = True,
    split: str | None = None,
    data_dir: Path | None = None,
    scannet_root: Path | None = None,
    replay_mode: str = "auto",
) -> dict[str, Any]:
    """Summarize one JSONL episode list into a flat row for CSV export."""
    if not episodes:
        return {
            "source_jsonl": str(source_jsonl or ""),
            "n_episodes": 0,
            "n_scored": 0,
            "em@1": 0.0,
            "em@1_refined": 0.0,
        }

    classified = classify_episodes(episodes, dedupe=dedupe)
    outcomes = summarize_outcomes(episodes, dedupe=dedupe)
    methods = {str(e.get("method", "")) for e in episodes if e.get("method")}
    backends = {str(e.get("replay_backend", "")) for e in episodes if e.get("replay_backend")}
    sens_xy = [
        float(e["sens_match_xy_m"])
        for e in episodes
        if e.get("sens_match_xy_m") is not None and str(e.get("sens_match_xy_m", "")).strip() != ""
    ]
    n_sens = sum(1 for e in episodes if str(e.get("replay_backend", "")) == "sens")
    planning = [int(e["planning_steps"]) for e in episodes if e.get("planning_steps") is not None]

    n_classified = max(1, len(classified))
    em_refined = sum(1 for r in classified if r.em_refined) / n_classified
    row: dict[str, Any] = {
        "source_jsonl": str(source_jsonl or ""),
        "method": sorted(methods)[0] if len(methods) == 1 else ",".join(sorted(methods)),
        "replay_backend": sorted(backends)[0] if len(backends) == 1 else ",".join(sorted(backends)),
        "n_episodes": outcomes["n_episodes"],
        "n_scored": outcomes["n_scored"],
        "n_infra": outcomes["n_infra"],
        "em@1": outcomes["em@1"],
        "em@1_refined": em_refined,
        "tp": outcomes["tp"],
        "fp": outcomes["fp"],
        "fn": outcomes["fn"],
        "precision": outcomes["precision"],
        "recall": outcomes["recall"],
        "fp_confident": outcomes["fp_confident"],
        "n_sens_anchor": n_sens,
        "sens_anchor_frac": n_sens / len(episodes),
        "mean_sens_match_xy_m": sum(sens_xy) / len(sens_xy) if sens_xy else "",
        "mean_planning_steps": sum(planning) / len(planning) if planning else "",
    }
    row.update(
        _split_coverage_fields(
            episodes,
            outcomes,
            split=split,
            data_dir=data_dir,
            scannet_root=scannet_root,
            replay_mode=replay_mode,
        )
    )
    return row


def aggregate_sqa3d_jsonl(
    path: Path,
    *,
    dedupe: bool = True,
    split: str | None = None,
    data_dir: Path | None = None,
    scannet_root: Path | None = None,
    replay_mode: str = "auto",
) -> dict[str, Any]:
    """Load one JSONL file and return an aggregate summary row."""
    episodes = load_episode_metrics_jsonl(path)
    if split is None and episodes:
        splits = {str(e.get("split", "")).strip() for e in episodes if e.get("split")}
        if len(splits) == 1:
            split = next(iter(splits))
    if replay_mode == "auto" and episodes:
        modes = {str(e.get("replay_mode", "")).strip() for e in episodes if e.get("replay_mode")}
        if len(modes) == 1:
            replay_mode = next(iter(modes))
    return aggregate_sqa3d_episodes(
        episodes,
        source_jsonl=path,
        dedupe=dedupe,
        split=split,
        data_dir=data_dir,
        scannet_root=scannet_root,
        replay_mode=replay_mode,
    )


def write_aggregate_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write aggregate rows to CSV; returns ``path``."""
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
    return path


def aggregate_sqa3d_jsonl_paths(
    paths: list[Path],
    *,
    dedupe: bool = True,
    split: str | None = None,
    data_dir: Path | None = None,
    scannet_root: Path | None = None,
    replay_mode: str = "auto",
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> list[dict[str, Any]]:
    """Aggregate multiple JSONL sweeps; optionally write CSV/JSON summaries."""
    rows = [
        aggregate_sqa3d_jsonl(
            path,
            dedupe=dedupe,
            split=split,
            data_dir=data_dir,
            scannet_root=scannet_root,
            replay_mode=replay_mode,
        )
        for path in paths
    ]
    if output_csv is not None:
        write_aggregate_csv(rows, output_csv)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
