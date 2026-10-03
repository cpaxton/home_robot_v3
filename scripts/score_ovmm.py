#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Score a directory of OVMM find-phase result JSON files.

Each ``<episode_id>.json`` holds the metrics dict from
``emet.eval.ovmm_find_phase.compute_find_phase_metrics`` (find-object / find-recep
success, localization error, partial success). Prints per-episode rows and the
aggregate find-object / find-recep success rate + mean localization error.

This is the fast OVMM dev-hill score (RoboCasa S1 rby1); Habitat HM3D remains the
occasional generalization gate.

Usage:
    python scripts/score_ovmm.py OUT_DIR
    python scripts/score_ovmm.py OUT_DIR --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _is_metrics(obj: dict) -> bool:
    return "find_object_success" in obj or "localization_err_obj_m" in obj


def load_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(out_dir.glob("*.json")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and _is_metrics(item):
                    rows.append(item)
        elif isinstance(data, dict) and _is_metrics(data):
            rows.append(data)
    return rows


def _rate(n_ok: int, n: int) -> str:
    return f"{n_ok}/{n} ({100.0 * n_ok / n:.0f}%)" if n else "n/a"


def _mean(values: list[float]) -> str:
    finite = [v for v in values if v is not None]
    return f"{sum(finite) / len(finite):.3f} m" if finite else "n/a"


def summarize(rows: list[dict], *, out_dir: Path | None = None) -> dict:
    obj_ok = sum(1 for r in rows if r.get("find_object_success"))
    rec_ok = sum(1 for r in rows if r.get("find_recep_success"))
    errored = sum(1 for r in rows if bool(r.get("error")))
    # Errored rows stay in the headline success rates (flagged, not hidden);
    # this is the supplementary view with them excluded entirely.
    clean_rows = [r for r in rows if not r.get("error")]
    clean = {
        "n": len(clean_rows),
        "find_object_success": sum(1 for r in clean_rows if r.get("find_object_success")),
        "find_recep_success": sum(1 for r in clean_rows if r.get("find_recep_success")),
    }
    obj_err = [r.get("localization_err_obj_m") for r in rows]
    rec_err = [r.get("localization_err_recep_m") for r in rows]
    partial = [r.get("find_partial_success") for r in rows if r.get("find_partial_success") is not None]
    per_episode = [
        {
            "episode_id": str(r.get("episode_id") or r.get("id") or "?"),
            "find_object_success": bool(r.get("find_object_success")),
            "find_recep_success": bool(r.get("find_recep_success")),
            "localization_err_obj_m": r.get("localization_err_obj_m"),
            "localization_err_recep_m": r.get("localization_err_recep_m"),
            "find_partial_success": r.get("find_partial_success"),
            "error": str(r.get("error") or ""),
        }
        for r in rows
    ]
    return {
        "n": len(rows),
        "n_errored": errored,
        "clean": clean,
        "find_object_success": obj_ok,
        "find_recep_success": rec_ok,
        "mean_localization_err_obj_m": (
            sum(v for v in obj_err if v is not None) / len([v for v in obj_err if v is not None])
        )
        if any(v is not None for v in obj_err)
        else None,
        "mean_localization_err_recep_m": (
            sum(v for v in rec_err if v is not None) / len([v for v in rec_err if v is not None])
        )
        if any(v is not None for v in rec_err)
        else None,
        "mean_partial_success": (sum(partial) / len(partial)) if partial else None,
        "per_episode": per_episode,
        "out_dir": str(out_dir) if out_dir else None,
    }


def print_summary(summary: dict) -> None:
    print(f"source: {summary['out_dir']}")
    n = summary["n"]
    errored = summary["n_errored"]
    print(f"episodes: {n}")
    if errored:
        print(f"ERRORED (crashed/exception, not a clean miss): {errored}")
        clean = summary["clean"]
        print(
            f"clean success, errored excluded: find-object {_rate(clean['find_object_success'], clean['n'])}, find-recep {_rate(clean['find_recep_success'], clean['n'])}"
        )
    print(f"find-object success: {_rate(summary['find_object_success'], n)}")
    print(f"find-recep  success: {_rate(summary['find_recep_success'], n)}")
    obj = summary["mean_localization_err_obj_m"]
    rec = summary["mean_localization_err_recep_m"]
    print(f"mean localization err  obj={obj:.3f} m" if obj is not None else "mean localization err  obj=n/a")
    print(f"mean localization err recep={rec:.3f} m" if rec is not None else "mean localization err recep=n/a")
    if summary["mean_partial_success"] is not None:
        print(f"mean partial success: {summary['mean_partial_success']:.2f}")
    print()
    print("per-episode:")
    for e in summary["per_episode"]:
        obj = e["localization_err_obj_m"]
        rec = e["localization_err_recep_m"]
        obj_s = f"{obj:.3f}" if obj is not None else "n/a"
        rec_s = f"{rec:.3f}" if rec is not None else "n/a"
        tag = f"  ERROR: {e['error']}" if e["error"] else ""
        print(
            f"  {e['episode_id']:32s} obj={'OK' if e['find_object_success'] else '--'} "
            f"recep={'OK' if e['find_recep_success'] else '--'}  err_obj={obj_s:>8s} err_recep={rec_s:>8s}{tag}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Directory of per-episode find-phase *.json results")
    parser.add_argument("--json", action="store_true", help="Emit a single JSON summary object")
    args = parser.parse_args()

    rows = load_rows(args.out_dir)
    if not rows:
        print(f"no find-phase *.json results found under {args.out_dir}", file=sys.stderr)
        return 2
    summary = summarize(rows, out_dir=args.out_dir)
    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
