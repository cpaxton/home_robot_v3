#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compare HM-EQA JSONL runs (accuracy, Q0-19 slice, per-question deltas)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = int(row["question_id"])
        rows[qid] = row
    return rows


def accuracy(rows: dict[int, dict], qids: range | list[int] | None = None) -> tuple[int, int, float]:
    if qids is None:
        items = list(rows.values())
    else:
        items = [rows[q] for q in qids if q in rows]
    if not items:
        return 0, 0, 0.0
    n = len(items)
    c = sum(1 for r in items if r.get("correct"))
    return c, n, 100.0 * c / n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline JSONL (e.g. paper run)")
    parser.add_argument("candidate", type=Path, nargs="?", help="Candidate JSONL (e.g. frontier v2)")
    parser.add_argument("--q-start", type=int, default=0)
    parser.add_argument("--q-end", type=int, default=19)
    args = parser.parse_args()

    base = load_jsonl(args.baseline.expanduser())
    cand = load_jsonl(args.candidate.expanduser()) if args.candidate else {}

    qslice = range(args.q_start, args.q_end + 1)
    bc, bn, bpct = accuracy(base, qslice)
    print(f"baseline {args.baseline.name}: Q{args.q_start}-{args.q_end} = {bc}/{bn} ({bpct:.1f}%)")
    fc, fn, fpct_all = accuracy(base)
    print(f"baseline full: {fc}/{fn} ({fpct_all:.1f}%)")

    if not cand:
        return 0

    cc, cn, cpct = accuracy(cand, qslice)
    print(f"candidate {args.candidate.name}: Q{args.q_start}-{args.q_end} = {cc}/{cn} ({cpct:.1f}%)")
    delta = cpct - bpct
    print(f"delta Q{args.q_start}-{args.q_end}: {delta:+.1f} pp")

    print("\nper-question (candidate vs baseline):")
    print(f"{'qid':>4} {'gold':>4} {'base':>5} {'cand':>5} {'base_ok':>7} {'cand_ok':>7}")
    gains: list[int] = []
    losses: list[int] = []
    for q in qslice:
        b = base.get(q)
        c = cand.get(q)
        if not b and not c:
            continue
        gold = (b or c).get("gold_answer_letter", "")
        bl = (b or {}).get("parsed_answer_letter") or ""
        cl = (c or {}).get("parsed_answer_letter") or ""
        bok = "Y" if b and b.get("correct") else ("-" if not b else "N")
        cok = "Y" if c and c.get("correct") else ("-" if not c else "N")
        print(f"{q:4d} {gold:>4} {bl:>5} {cl:>5} {bok:>7} {cok:>7}")
        if b and c:
            if c.get("correct") and not b.get("correct"):
                gains.append(q)
            elif b.get("correct") and not c.get("correct"):
                losses.append(q)
    print(f"\ncandidate gains vs baseline: {gains}")
    print(f"candidate losses vs baseline: {losses}")
    if cn and bn:
        print(
            f"\nverdict: {'IMPROVED' if cpct > bpct else 'FLAT/REGRESSED' if cpct < bpct else 'TIE'} on Q{args.q_start}-{args.q_end}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
