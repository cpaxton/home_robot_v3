#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Inspect seven-track simulation smoke battery results (metrics + visual artifact paths).

Usage:
  uv run python scripts/inspect_simulation_smoke_battery.py --run-id sim_smoke_agent_20260628
  uv run python scripts/inspect_simulation_smoke_battery.py --run-id sim_smoke_agent_20260628 --write-report
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()


@dataclass
class TrackCheck:
    track: int
    name: str
    harness: str = "unknown"  # PASS | FAIL | SKIP | unknown
    semantic: str = "unknown"  # PASS | FAIL | WARN | SKIP | unknown
    headline: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_csv_row(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            return dict(row)
    return None


def _parse_summary(summary_path: Path) -> dict[int, str]:
    harness: dict[int, str] = {}
    if not summary_path.is_file():
        return harness
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"=== (PASS|FAIL|SKIP) track(\d+)_", line)
        if m:
            harness[int(m.group(2))] = m.group(1)
        m = re.search(r"SKIP track(\d)\b", line)
        if m:
            harness[int(m.group(1))] = "SKIP"
    return harness


def _status_icon(status: str) -> str:
    return {
        "PASS": "✓",
        "FAIL": "✗",
        "WARN": "!",
        "SKIP": "-",
        "unknown": "?",
    }.get(status, "?")


def inspect_track1(run_id: str, harness: str) -> TrackCheck:
    out = TrackCheck(1, "Habitat EQA", harness=harness)
    jsonl = HOME / ".cache/habitat_eqa/results" / f"{run_id}_hmeqa_q17.jsonl"
    rows = _load_jsonl(jsonl)
    if not rows:
        out.semantic = "FAIL" if harness == "PASS" else "unknown"
        out.headline = "missing HM-EQA jsonl"
        out.notes.append(f"expected: {jsonl}")
        return out
    row = rows[0]
    correct = bool(row.get("correct"))
    out.metrics = {
        "correct": correct,
        "predicted": row.get("predicted_answer"),
        "gold": row.get("gold_answer_letter"),
        "graph_nodes": row.get("graph_nodes"),
        "planning_steps": row.get("planning_steps"),
    }
    bundle = row.get("debug_bundle_dir") or ""
    topdown = row.get("topdown_map_path") or ""
    if bundle:
        out.artifacts.append(str(bundle))
    if topdown and Path(str(topdown)).is_file():
        out.artifacts.append(str(topdown))
    for name in ("topdown_map.png", "episode_rgb.mp4", "scene_graph_report.txt"):
        p = Path(bundle) / name
        if p.is_file():
            out.artifacts.append(str(p))
    out.semantic = "PASS" if correct else "FAIL"
    out.headline = f"MCQ {'correct' if correct else 'WRONG'} ({row.get('predicted_answer')} vs gold {row.get('gold_answer_letter')})"
    return out


def inspect_track2(run_id: str, harness: str) -> TrackCheck:
    out = TrackCheck(2, "Habitat OVMM GT", harness=harness)
    path = HOME / "runs/emet/ovmm_habitat" / run_id / "hm3d_lamp_bed_00006_gt.json"
    row = _load_json(path)
    if row is None:
        out.semantic = "FAIL" if harness == "PASS" else "unknown"
        out.headline = "missing OVMM json"
        out.notes.append(f"expected: {path}")
        return out
    fps = float(row.get("find_partial_success") or 0.0)
    out.metrics = {
        "find_partial_success": fps,
        "find_object_success": row.get("find_object_success"),
        "find_recep_success": row.get("find_recep_success"),
        "n_graph_nodes": row.get("n_graph_nodes"),
    }
    out.semantic = "PASS" if fps > 0 else "FAIL"
    out.headline = f"find_partial_success={fps}"
    out.artifacts.append(str(path))
    return out


def _inspect_ovmm_find(run_id: str, harness: str, *, suffix: str, episode_id: str, expect_gt_pass: bool) -> TrackCheck:
    track_num = 3 if suffix == "robocasa" else 4
    label = "Robocasa search" if suffix == "robocasa" else "Molmo search"
    out = TrackCheck(track_num, label, harness=harness)
    root = HOME / "runs/emet/ovmm_find_phase" / f"{run_id}_{suffix}"
    path = root / f"{episode_id}_ground_truth.json"
    row = _load_json(path)
    if row is None:
        out.semantic = "FAIL" if harness == "PASS" else "unknown"
        out.headline = "missing find-phase json"
        out.notes.append(f"expected: {path}")
        return out
    fps = float(row.get("find_partial_success") or 0.0)
    out.metrics = {
        "episode_id": row.get("episode_id"),
        "find_partial_success": fps,
        "find_object_success": row.get("find_object_success"),
        "find_recep_success": row.get("find_recep_success"),
        "n_graph_nodes": row.get("n_graph_nodes"),
        "gt_object_body": row.get("gt_object_body"),
        "obj_localize_source": row.get("obj_localize_source"),
    }
    out.artifacts.append(str(path))
    csv_path = root / "aggregate_ground_truth.csv"
    if csv_path.is_file():
        out.artifacts.append(str(csv_path))
    if expect_gt_pass and fps <= 0:
        out.semantic = "FAIL"
        out.headline = f"find_partial_success={fps} (GT smoke expects >0)"
        out.notes.append(
            "Harness exit 0 alone is not enough — GT oracle should localize object + recep."
        )
        if row.get("gt_object_body") is None:
            out.notes.append(
                "gt_object_body is null: sim placements may not match episode queries "
                f"({row.get('object_query')!r} / {row.get('goal_recep')!r})."
            )
    elif fps > 0:
        out.semantic = "PASS"
        out.headline = f"find_partial_success={fps}"
    else:
        out.semantic = "WARN"
        out.headline = f"find_partial_success={fps}"
    return out


def inspect_track5(harness: str) -> TrackCheck:
    out = TrackCheck(5, "SQA3D mock", harness=harness)
    # SQA3D CLI prints JSON to stdout; track log is the artifact path from battery.
    out.semantic = "PASS" if harness == "PASS" else ("SKIP" if harness == "SKIP" else "unknown")
    out.headline = "mock-LLM EM@1 (see track5 log for JSON block)"
    out.notes.append("Re-run with MOCK_LLM=0 for real VLM + ScanNet replay smoke.")
    return out


def inspect_track6(run_id: str, harness: str) -> TrackCheck:
    out = TrackCheck(6, "Robocasa world-change", harness=harness)
    root = HOME / "runs/emet/dynamic_exploration" / f"{run_id}_world_change"
    csv_row = _load_csv_row(root / "aggregate_dynamic_exploration_world_change.csv")
    json_path = root / "robocasa_seed0_world_change_dynagraph.json"
    payload = _load_json(json_path)
    if payload is None and csv_row is None:
        out.semantic = "FAIL" if harness != "SKIP" else "SKIP"
        out.headline = "no aggregate output (timeout or crash before write)"
        out.notes.append(f"check log + partial exports under {root}")
        return out
    err = (payload or csv_row or {}).get("error", "")
    if err:
        out.semantic = "FAIL"
        out.headline = "error in payload"
        out.metrics["error"] = str(err)[:200]
    else:
        pre = (payload or {}).get("answer_correct_pre")
        post = (payload or {}).get("answer_correct_post")
        out.metrics = {
            "answer_correct_pre": pre,
            "answer_correct_post": post,
            "recovery_steps": (payload or {}).get("recovery_steps"),
            "n_stale_nodes_after_move": (payload or {}).get("n_stale_nodes_after_move"),
            "episode_wall_s": (payload or {}).get("episode_wall_s"),
        }
        out.semantic = "PASS" if pre is not False and post is not False else "WARN"
        out.headline = f"pre={pre} post={post}"
    export = root / "exports" / "robocasa_seed0_world_change_dynagraph"
    for name in ("scene_graph_report.txt", "manifest.json", "floor_metrics.json"):
        p = export / name
        if p.is_file():
            out.artifacts.append(str(p))
    if json_path.is_file():
        out.artifacts.append(str(json_path))
    return out


def inspect_track7(run_id: str, harness: str) -> TrackCheck:
    out = TrackCheck(7, "Molmo dynamic explore", harness=harness)
    root = HOME / "runs/emet/dynamic_exploration" / f"{run_id}_molmo_explore"
    json_path = root / "molmo_ithor0_dynagraph_explore_3.json"
    payload = _load_json(json_path)
    csv_row = _load_csv_row(root / "aggregate_dynamic_exploration.csv")
    err = ""
    if payload:
        err = str((payload.get("metrics") or {}).get("error") or payload.get("error") or "")
    if not err and csv_row:
        err = str(csv_row.get("error") or "")
    export = root / "exports" / "molmo_ithor0_dynagraph_explore_3"
    log_path = export / "dynagraph.log"
    if log_path.is_file():
        out.artifacts.append(str(log_path))
        text = log_path.read_text(encoding="utf-8", errors="replace")
        ok_steps = len(re.findall(r"explore-loop: step \d+/\d+ ok", text))
        out.metrics["explore_steps_ok"] = ok_steps
        if "Exported graph memory to" in text:
            out.metrics["export_completed"] = True
        if "reached max_iterations (3)" in text:
            out.metrics["explore_k3_finished"] = True
    for name in ("scene_graph_report.txt", "manifest.json", "floor_metrics.json"):
        p = export / name
        if p.is_file():
            out.artifacts.append(str(p))
    if err:
        out.semantic = "FAIL"
        out.headline = "subprocess error (see JSON/CSV)"
        out.metrics["error"] = err[:240]
        if out.metrics.get("explore_k3_finished") and not out.metrics.get("export_completed"):
            out.notes.append(
                "Explore K=3 finished but export/EQA did not — use --skip-eqa + GPU for smoke."
            )
    elif out.metrics.get("explore_k3_finished") and not out.metrics.get("export_completed"):
        out.semantic = "FAIL" if harness == "PASS" else "WARN"
        out.headline = "explore finished; export missing (likely EQA timeout)"
        out.notes.append("Re-run with --skip-eqa and DYNAMIC_GPU=1.")
    elif payload and payload.get("summary"):
        summary = payload["summary"]
        frac = summary.get("explored_fraction")
        nodes = summary.get("node_count")
        out.metrics.update(
            {
                "explored_fraction": frac,
                "node_count": nodes,
                "spatial_recall": summary.get("spatial_recall"),
                "eqa_accuracy": summary.get("eqa_accuracy"),
            }
        )
        ok = (frac is None or float(frac or 0) > 0) and (nodes is None or float(nodes or 0) > 0)
        out.semantic = "PASS" if ok else "WARN"
        out.headline = f"explored_fraction={frac} nodes={nodes}"
    else:
        out.semantic = "unknown"
        out.headline = "no JSON summary"
    if json_path.is_file():
        out.artifacts.append(str(json_path))
    return out


def _rerun_replay_commands(run_id: str) -> list[str]:
    return [
        "# Habitat Q17 bundle (topdown map, trajectory, scene graph):",
        f"xdg-open ~/.cache/habitat_eqa/episodes/cli_episode_q0017/q0017_dynagraph/topdown_map.png",
        "",
        "# Molmo dynamic explore — live Rerun replay (K=3, no EQA, GPU):",
        "EMET_SIM_NAV_TELEPORT=1 uv run emet run dynagraph \\",
        "  --robot stretch --robot-ip 127.0.0.1 --port-offset 0 \\",
        "  --molmospaces-scene ithor --molmospaces-split train --molmospaces-index 0 \\",
        "  --explore-loop --explore-max-iters 3 --export /tmp/smoke_molmo_explore_rerun",
        "",
        "# Robocasa world-change visual (short K=3; needs sim server running):",
        "uv run python scripts/eval_dynamic_exploration.py \\",
        f"  --phase world-change --episode-id robocasa_seed0_world_change \\",
        "  --backend dynagraph --explore-max-iters 3 \\",
        f"  --output-dir ~/runs/emet/dynamic_exploration/{run_id}_world_change_replay",
    ]


def format_report(run_id: str, checks: list[TrackCheck], summary_path: Path) -> str:
    lines = [
        f"# Simulation smoke inspection — `{run_id}`",
        "",
        f"Summary log: `{summary_path}`",
        "",
        "| Track | Harness | Semantic | Headline |",
        "|------|---------|----------|----------|",
    ]
    for c in checks:
        lines.append(
            f"| {c.track} {c.name} | {_status_icon(c.harness)} {c.harness} "
            f"| {_status_icon(c.semantic)} {c.semantic} | {c.headline} |"
        )
    lines.append("")
    for c in checks:
        lines.extend([f"## Track {c.track} — {c.name}", ""])
        if c.metrics:
            lines.append("**Metrics:**")
            for k, v in c.metrics.items():
                if v is not None and v != "":
                    lines.append(f"- `{k}`: {v}")
            lines.append("")
        if c.artifacts:
            lines.append("**Artifacts (open locally):**")
            for a in c.artifacts:
                lines.append(f"- `{a}`")
            lines.append("")
        if c.notes:
            lines.append("**Notes:**")
            for n in c.notes:
                lines.append(f"- {n}")
            lines.append("")
    lines.extend(["## Visual replay commands", ""] + _rerun_replay_commands(run_id) + [""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect seven-track simulation smoke battery results.")
    parser.add_argument("--run-id", required=True, help="RUN_ID used by run_simulation_smoke_battery.sh")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write inspection_report.md next to summary.txt",
    )
    args = parser.parse_args()
    run_id = args.run_id.strip()
    log_dir = HOME / "runs/emet/simulation_smoke" / run_id
    summary_path = log_dir / "summary.txt"
    harness = _parse_summary(summary_path)

    def h(track_num: int) -> str:
        return harness.get(track_num, "unknown")

    checks = [
        inspect_track1(run_id, h(1)),
        inspect_track2(run_id, h(2)),
        _inspect_ovmm_find(run_id, h(3), suffix="robocasa", episode_id="robocasa_pp_s1", expect_gt_pass=True),
        _inspect_ovmm_find(
            run_id, h(4), suffix="molmo", episode_id="molmo_ithor_s2_idx0", expect_gt_pass=True
        ),
        inspect_track5(h(5)),
        inspect_track6(run_id, h(6)),
        inspect_track7(run_id, h(7)),
    ]

    report = format_report(run_id, checks, summary_path)
    print(report)

    if args.write_report:
        out_path = log_dir / "inspection_report.md"
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", file=sys.stderr)

    semantic_fail = any(c.semantic == "FAIL" for c in checks if c.harness != "SKIP")
    return 1 if semantic_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
