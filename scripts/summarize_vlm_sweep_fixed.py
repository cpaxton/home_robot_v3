#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compare fixed VLM sweep runs vs paper baseline and pre-fix sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SLUGS = ("gemma4_e4b", "gemma4_e2b", "gemma3_4b", "qwen3_vl_4b", "qwen25_vl_3b")


def load_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["question_id"])] = row
    return out


def acc(rows: dict[int, dict], qids: range) -> tuple[int, int]:
    items = [rows[q] for q in qids if q in rows]
    if not items:
        return 0, 0
    return sum(1 for r in items if r.get("correct")), len(items)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("~/.cache/habitat_eqa/results"))
    p.add_argument("--q-start", type=int, default=0)
    p.add_argument("--q-end", type=int, default=9)
    p.add_argument("--suffix", default="_fixed")
    args = p.parse_args()
    out_dir = args.results_dir.expanduser()
    qslice = range(args.q_start, args.q_end + 1)
    base = load_jsonl(out_dir / "graph_eqa_gemma3_paper_q0-112.jsonl")
    bc, bn = acc(base, qslice)
    print(f"Paper baseline Q{args.q_start}-{args.q_end}: {bc}/{bn} ({100 * bc / bn:.1f}%)\n")
    print(f"{'slug':<16} {'pre-fix':>10} {'fixed':>10} {'model':<30}")
    print("-" * 72)
    for slug in SLUGS:
        pre = load_jsonl(out_dir / f"vlm_sweep_{slug}_q{args.q_start}-{args.q_end}.jsonl")
        post = load_jsonl(out_dir / f"vlm_sweep_{slug}_q{args.q_start}-{args.q_end}{args.suffix}.jsonl")
        pc, pn = acc(pre, qslice)
        fc, fn = acc(post, qslice)
        model = ""
        if post:
            model = str(next(iter(post.values())).get("vl_hf_model_id", ""))[:30]
        elif pre:
            model = str(next(iter(pre.values())).get("vl_hf_model_id", ""))[:30]
        pre_s = f"{pc}/{pn}" if pn else "—"
        fix_s = f"{fc}/{fn}" if fn else "—"
        print(f"{slug:<16} {pre_s:>10} {fix_s:>10} {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
