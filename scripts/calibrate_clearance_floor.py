#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Calibrate the nav clearance floor (``min_clearance``) for graded conservatism.

Sweeps the frozen nav_goal clutter tasks over the ``floor_values`` grid, running each
task at each clearance via ``eval_tamp_clutter.py --clearance-m <floor>``. Aggregates
success / refusal / margin per floor and recommends the knee: the smallest floor whose
success rate matches the best across the grid (ties broken toward the more conservative
higher floor).

This is a **sim-only** calibration (RoboCasa/MolmoSpaces is not physical). RBY1 arm
collision geometry is incomplete, so "zero collisions" is a soft signal — verify the
recommended floor visually before locking it.

Run under ``emet jobs`` (never block an agent turn on sim)::

  uv run emet jobs run --name clearance-sweep --need-mib 8000 -- \\
    uv run python scripts/calibrate_clearance_floor.py

Dry-run (validate the manifest and print the matrix without launching sim)::

  uv run python scripts/calibrate_clearance_floor.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "configs" / "benchmarks" / "clearance_calibration.yaml"
CLUTTER_RUNNER = REPO / "scripts" / "eval_tamp_clutter.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    p.add_argument("--output-dir", type=str, default=None, help="default: ~/runs/emet/clearance_sweep")
    p.add_argument("--dry-run", action="store_true", help="validate manifest + print the (floor x task) matrix")
    p.add_argument("--run-smoke", action="store_true", help="also run the pick-place smoke once (after the sweep)")
    p.add_argument("--cpu-only", action="store_true")
    return p.parse_args()


def load_manifest(path: str) -> dict[str, Any]:
    full = Path(path).expanduser().resolve()
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid manifest (expected a mapping): {full}")
    return raw


def validate_tasks(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (nav_goal_task_ids, pick_place_smoke) after validating against the registry."""
    from emet.eval.tamp_clutter import load_clutter_episodes

    episodes_file = str(manifest.get("episodes_file") or "configs/ovmm/clutter_episodes_signal.yaml")
    eps = load_clutter_episodes(REPO / episodes_file)
    by_id = {e.id: e for e in eps}
    tasks = [str(t) for t in manifest.get("nav_goal_tasks") or []]
    if not tasks:
        raise ValueError("manifest has no nav_goal_tasks")
    missing = [t for t in tasks if t not in by_id]
    if missing:
        raise ValueError(f"nav_goal_tasks not in {episodes_file}: {missing}")
    non_nav = [t for t in tasks if by_id[t].mode != "nav_goal"]
    if non_nav:
        raise ValueError(f"tasks are not nav_goal (clearance-insensitive): {non_nav}")
    smoke = str(manifest.get("pick_place_smoke") or "").strip()
    return tasks, ([smoke] if smoke else [])


def _runner_cmd(episodes_file: str, task_ids: list[str], clearance_m: float, output_dir: Path, *, cpu_only: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        str(CLUTTER_RUNNER),
        "--episodes",
        episodes_file,
        "--clearance-m",
        f"{clearance_m:.2f}",
        "--output-dir",
        str(output_dir),
    ]
    if cpu_only:
        cmd.append("--cpu-only")
    for t in task_ids:
        cmd += ["--episode-id", t]
    return cmd


def run_floor(episodes_file: str, task_ids: list[str], clearance_m: float, output_dir: Path, *, cpu_only: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _runner_cmd(episodes_file, task_ids, clearance_m, output_dir, cpu_only=cpu_only)
    print(f"[floor {clearance_m:.2f}] {task_ids}", file=sys.stderr)
    subprocess.run(cmd, cwd=str(REPO), check=True)
    return output_dir


def collect(output_dir: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in task_ids:
        f = output_dir / f"{t}.json"
        if not f.exists():
            rows.append({"episode_id": t, "status": "missing_output"})
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        refusal = data.get("nav_refusal")
        rows.append(
            {
                "episode_id": t,
                "task_success": bool(data.get("task_success", False)),
                "nav_path_open": data.get("nav_path_open"),
                "refused": bool(refusal) if refusal is not None else (not bool(data.get("nav_path_open", True))),
                "margin_m": (refusal or {}).get("margin_m") if isinstance(refusal, dict) else None,
                "status": data.get("status"),
            }
        )
    return rows


def aggregate(floor_summary: dict[float, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for floor in sorted(floor_summary):
        rows = floor_summary[floor]
        n = len(rows)
        n_success = sum(1 for r in rows if r.get("task_success"))
        n_refused = sum(1 for r in rows if r.get("refused"))
        margins = [r["margin_m"] for r in rows if r.get("margin_m") is not None]
        out.append(
            {
                "floor_m": floor,
                "n_tasks": n,
                "n_success": n_success,
                "n_refused": n_refused,
                "success_rate": round(n_success / n, 3) if n else 0.0,
                "refusal_rate": round(n_refused / n, 3) if n else 0.0,
                "min_margin_m": round(min(margins), 3) if margins else None,
            }
        )
    return out


def recommend_floor(agg: list[dict[str, Any]]) -> float | None:
    """Smallest floor whose success rate matches the best across the grid (the knee)."""
    if not agg:
        return None
    best = max(a["success_rate"] for a in agg)
    candidates = [a for a in agg if a["success_rate"] == best]
    return min(c["floor_m"] for c in candidates)


def main() -> int:
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    floors = [float(f) for f in manifest.get("floor_values") or []]
    if not floors:
        raise SystemExit("manifest has no floor_values")
    episodes_file = str(manifest.get("episodes_file") or "configs/ovmm/clutter_episodes_signal.yaml")
    task_ids, smoke = validate_tasks(manifest)
    seed = manifest.get("seed")

    if args.dry_run:
        print(f"manifest: {args.manifest}")
        print(f"seed: {seed}  episodes_file: {episodes_file}")
        print(f"floors: {[f'{f:.2f}' for f in floors]}")
        print(f"nav_goal tasks ({len(task_ids)}): {task_ids}")
        print(f"pick_place smoke: {smoke or '(none)'}")
        print(f"matrix: {len(floors)} floors x {len(task_ids)} tasks = {len(floors) * len(task_ids)} runs")
        return 0

    out_root = Path(args.output_dir or "~/runs/emet/clearance_sweep").expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    floor_summary: dict[float, list[dict[str, Any]]] = {}
    for floor in floors:
        odir = out_root / f"floor_{floor:.2f}".replace(".", "_")
        run_floor(episodes_file, task_ids, floor, odir, cpu_only=args.cpu_only)
        floor_summary[floor] = collect(odir, task_ids)

    agg = aggregate(floor_summary)
    recommendation = recommend_floor(agg)

    print("\n=== clearance sweep summary ===", file=sys.stderr)
    for a in agg:
        print(
            f"floor={a['floor_m']:.2f}  success={a['success_rate']:.2f}  "
            f"refused={a['refusal_rate']:.2f}  min_margin={a['min_margin_m']}",
            file=sys.stderr,
        )
    print(f"recommended min_clearance = {recommendation}", file=sys.stderr)
    print(
        "NOTE: the probe and the scorer share the same clearance, so success at a tight "
        "floor is not by itself proof of safety — verify no penetration visually before "
        "locking the floor (RBY1 arm collision geometry is incomplete).",
        file=sys.stderr,
    )

    if args.run_smoke and smoke:
        smoke_dir = out_root / "pick_place_smoke"
        smoke_cmd = [
            sys.executable,
            str(REPO / "scripts" / "eval_tamp_floor.py"),
            "--episode-id",
            smoke[0],
        ]
        if args.cpu_only:
            smoke_cmd.append("--cpu-only")
        print(f"[smoke] running pick-place {smoke[0]}", file=sys.stderr)
        subprocess.run(smoke_cmd, cwd=str(REPO), check=True)

    (out_root / "sweep_summary.json").write_text(
        json.dumps({"floors": agg, "recommended_min_clearance": recommendation, "seed": seed}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_root / 'sweep_summary.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
