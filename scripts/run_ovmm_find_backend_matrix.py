#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run OVMM find-phase backend matrix for Robocasa S1 + Molmo S2 (paper Phase 1 find).

Non-GT backends **rotate in place** by default so the voxel/graph maps populate.
GT oracle keeps ``--not-rotate`` (placements only). Perception backends use explore-budget
episode aliases (``*_exploreN``) so mapping covers the target object.

Examples::

    # CPU oracle smoke (fast wiring check)
    uv run python scripts/run_ovmm_find_backend_matrix.py --cpu-only --backends ground_truth

    # Full backend matrix (GPU recommended; one job at a time)
    NEED_MIB=12000 ./scripts/gpu_preflight.sh --wait
    uv run python scripts/run_ovmm_find_backend_matrix.py \\
      --backends ground_truth,dynamem,graph_eqa,dynagraph \\
      --output-dir ~/runs/emet/ovmm_find_phase/backend_matrix
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_EPISODES = ("robocasa_pp_s1", "molmo_ithor_s2_idx0")
# Perception backends need mapping; use explore-budget episode variants when available.
_PERCEPTION_EPISODE_ALIASES = {
    "robocasa_pp_s1": "robocasa_pp_s1_explore5",
    "molmo_ithor_s2_idx0": "molmo_ithor_s2_idx0_explore15",
}
# Floor on explore_steps for perception backends (belt-and-suspenders if alias missing).
_PERCEPTION_MIN_EXPLORE_STEPS = {
    "robocasa_pp_s1": 8,
    "molmo_ithor_s2_idx0": 15,
}
_DEFAULT_BACKENDS = ("ground_truth", "dynamem", "graph_eqa", "dynagraph")
_METRIC_KEYS = (
    "find_partial_success",
    "find_object_success",
    "find_recep_success",
    "localization_err_obj_m",
    "localization_err_recep_m",
    "n_graph_nodes",
    "n_voxel_explored_cells",
    "obj_localize_source",
    "recep_localize_source",
    "object_query",
    "explore_steps",
)


def _ingest_metrics(run_dir: Path, row: dict) -> None:
    """Pull find metrics from the episode result JSON (not only summary.json)."""
    candidates: list[Path] = []
    for cand in sorted(run_dir.rglob("*.json")):
        name = cand.name
        if name == "matrix_summary.json":
            continue
        if name in ("summary.json",) or name.endswith("_metrics.json"):
            candidates.insert(0, cand)
        elif cand.parent == run_dir or cand.parent.parent == run_dir:
            candidates.append(cand)
    for cand in candidates:
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "find_partial_success" not in data and "episode_id" not in data:
            continue
        row["metrics_path"] = str(cand)
        for k in _METRIC_KEYS:
            if k in data:
                row[k] = data[k]
        return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", default=",".join(_DEFAULT_EPISODES))
    ap.add_argument("--backends", default=",".join(_DEFAULT_BACKENDS))
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--not-rotate",
        action="store_true",
        help="Skip rotate-in-place for all backends (GT-only smokes). Default: rotate for non-GT.",
    )
    ap.add_argument(
        "--rotate",
        action="store_true",
        help="Force rotate-in-place even for ground_truth.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "runs/emet/ovmm_find_phase/backend_matrix",
    )
    ap.add_argument("--port-offset-base", type=int, default=50)
    args = ap.parse_args()

    episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    offset = int(args.port_offset_base)

    for ep in episodes:
        for backend in backends:
            episode_id = ep
            explore_override: int | None = None
            if backend != "ground_truth" and not args.not_rotate:
                episode_id = _PERCEPTION_EPISODE_ALIASES.get(ep, ep)
                explore_override = _PERCEPTION_MIN_EXPLORE_STEPS.get(ep)
            run_dir = out_root / f"{ep}_{backend}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(_REPO / "scripts/eval_ovmm_find_phases.py"),
                "--episode-id",
                episode_id,
                "--backend",
                backend,
                "--output-dir",
                str(run_dir),
                "--port-offset",
                str(offset),
            ]
            if args.cpu_only:
                cmd.append("--cpu-only")
            if explore_override is not None:
                cmd.extend(["--explore-steps", str(explore_override)])
            # GT oracle uses placements; skip rotate unless forced.
            # Perception backends use explore-budget episodes + rotate.
            skip_rotate = args.not_rotate or (backend == "ground_truth" and not args.rotate)
            if skip_rotate:
                cmd.append("--not-rotate")
            t0 = time.monotonic()
            print(f"[matrix] {' '.join(cmd)}", flush=True)
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            proc = subprocess.run(cmd, cwd=str(_REPO), env=env)
            wall = time.monotonic() - t0
            row = {
                "episode_id": ep,
                "eval_episode_id": episode_id,
                "backend": backend,
                "returncode": proc.returncode,
                "wall_s": wall,
                "output_dir": str(run_dir),
                "not_rotate": bool(skip_rotate),
                "explore_steps_override": explore_override,
            }
            _ingest_metrics(run_dir, row)
            summary.append(row)
            offset += 2
            partial = row.get("find_partial_success")
            nodes = row.get("n_graph_nodes")
            cells = row.get("n_voxel_explored_cells")
            print(
                f"[matrix] done {ep} {backend} rc={proc.returncode} "
                f"partial={partial} nodes={nodes} cells={cells} "
                f"query={row.get('object_query')!r} explore={row.get('explore_steps')} "
                f"wall_s={wall:.0f}",
                flush=True,
            )
            if proc.returncode != 0 or (
                isinstance(partial, (int, float)) and float(partial) <= 0.0 and backend == "ground_truth"
            ):
                print(f"[matrix] FAIL {ep} {backend} rc={proc.returncode} partial={partial}", flush=True)

    summary_path = out_root / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    failed = [
        r
        for r in summary
        if int(r.get("returncode", 1)) != 0
        or (
            r.get("backend") == "ground_truth"
            and isinstance(r.get("find_partial_success"), (int, float))
            and float(r["find_partial_success"]) <= 0.0
        )
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
