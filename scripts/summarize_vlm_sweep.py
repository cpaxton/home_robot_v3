#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Print accuracy table for habitat VLM sweep JSONL runs vs paper baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SWEEP_SLUGS = (
    "gemma4_e4b_auto",
    "gemma4_e4b",
    "gemma4_e2b",
    "gemma3_4b",
    "qwen3_vl_4b",
    "qwen25_vl_3b",
)


def load_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[int(row["question_id"])] = row
    return rows


def acc(rows: dict[int, dict], qids: range) -> tuple[int, int, float, int]:
    items = [rows[q] for q in qids if q in rows]
    if not items:
        return 0, 0, 0.0, 0
    n = len(items)
    c = sum(1 for r in items if r.get("correct"))
    oom = sum(1 for r in items if "cuda out of memory" in str(r.get("predicted_answer", "")).lower())
    err = sum(1 for r in items if str(r.get("error", "")).strip())
    return c, n, 100.0 * c / n, oom, err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("~/.cache/habitat_eqa/results"))
    parser.add_argument("--baseline", type=Path, default=None, help="Paper baseline JSONL")
    parser.add_argument("--q-start", type=int, default=0)
    parser.add_argument("--q-end", type=int, default=19)
    args = parser.parse_args()

    out_dir = args.results_dir.expanduser()
    qslice = range(args.q_start, args.q_end + 1)
    baseline_path = args.baseline.expanduser() if args.baseline else out_dir / "graph_eqa_gemma3_paper_q0-112.jsonl"
    base = load_jsonl(baseline_path)
    bc, bn, bpct, _, _ = acc(base, qslice)

    print(f"Baseline ({baseline_path.name}): Q{args.q_start}-{args.q_end} = {bc}/{bn} ({bpct:.1f}%)\n")
    print(f"{'slug':<18} {'model':<32} {'acc':>12} {'oom':>5} {'err':>4} {'n':>4} {'delta':>7}")
    print("-" * 86)

    for slug in SWEEP_SLUGS:
        tag = f"vlm_sweep_{slug}_q{args.q_start}-{args.q_end}"
        path = out_dir / f"{tag}.jsonl"
        rows = load_jsonl(path)
        c, n, pct, oom, err = acc(rows, qslice)
        if not rows:
            print(f"{slug:<18} {'(missing)':<32} {'—':>12} {'—':>5} {'—':>4} {'0':>4} {'—':>7}")
            continue
        sample = next(iter(rows.values()))
        model = str(sample.get("vl_hf_model_id") or sample.get("eqa_hf_model_id") or "?")[:32]
        delta = pct - bpct if bn else 0.0
        print(f"{slug:<18} {model:<32} {c}/{n} ({pct:4.1f}%) {oom:>5} {err:>4} {n:>4} {delta:+6.1f}pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
