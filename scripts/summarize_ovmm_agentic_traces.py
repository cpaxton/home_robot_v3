#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Summarize OVMM find-phase JSON outputs (FindObj / FindRec rates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"aggregate.json", "summary.json"}:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("episode_id"):
            rows.append(data)
    agg = out_dir / "aggregate.json"
    if agg.is_file():
        try:
            blob = json.loads(agg.read_text())
            if isinstance(blob, list):
                rows.extend([r for r in blob if isinstance(r, dict)])
        except (OSError, json.JSONDecodeError):
            pass
    return rows


def _summarize(rows: list[dict]) -> dict[str, int | float]:
    n = len(rows)
    obj = sum(1 for r in rows if r.get("find_object_success"))
    recep = sum(1 for r in rows if r.get("find_recep_success"))
    partial = sum(1 for r in rows if r.get("find_partial_success"))
    return {
        "n_episodes": n,
        "find_object_success": obj,
        "find_recep_success": recep,
        "find_partial_success": partial,
        "find_object_rate": (obj / n) if n else 0.0,
        "find_recep_rate": (recep / n) if n else 0.0,
        "find_partial_rate": (partial / n) if n else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    if not out_dir.is_dir():
        raise SystemExit(f"not a directory: {out_dir}")
    rows = _load_rows(out_dir)
    summary = _summarize(rows)
    print(f"OUT {out_dir}")
    print(
        f"episodes={summary['n_episodes']} "
        f"FindObj={summary['find_object_success']}/{summary['n_episodes']} "
        f"FindRec={summary['find_recep_success']}/{summary['n_episodes']} "
        f"partial={summary['find_partial_success']}/{summary['n_episodes']}"
    )
    for row in rows:
        eid = row.get("episode_id", "?")
        find_obj = bool(row.get("find_object_success"))
        find_recep = bool(row.get("find_recep_success"))
        meta = row.get("agentic_meta") or {}
        print(
            f"  {eid}: obj={find_obj} recep={find_recep} "
            f"obj_rounds={meta.get('obj_agentic_rounds')} recep_rounds={meta.get('recep_agentic_rounds')}"
        )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
