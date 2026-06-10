#!/usr/bin/env python3
"""Re-grade Habitat EQA JSONL with fixed MCQ parsing (no blank/spurious-A)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from emet.habitat.metrics import (
    _answer_field_lines,
    extract_mcq_letter_from_raw_eqa,
    grade_mcq_answer,
)


def _answer_field_blank(raw: str) -> bool:
    fields = _answer_field_lines(raw)
    return bool(fields) and not fields[-1].strip()


def rescore_letter(row: dict) -> str:
    """Letter under fixed parser; ignore legacy false A from empty answer: fields."""
    if str(row.get("error") or "").strip():
        return ""
    if str(row.get("predicted_answer") or "").startswith("ERROR:"):
        return ""

    raw = str(row.get("raw_eqa_output") or "")
    choices = row.get("choices") or None
    letter = extract_mcq_letter_from_raw_eqa(raw, choices if choices else None)
    if letter:
        return letter

    blank = _answer_field_blank(raw)
    for key in ("parsed_answer_letter", "predicted_answer"):
        val = str(row.get(key) or "").strip().upper()
        if len(val) == 1 and val in "ABCD":
            if blank:
                return ""
            return val
    return ""


def rescore_file(path: Path, q_start: int | None, q_end: int | None) -> dict:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    old_c = new_c = n = blank = 0
    flips: list[tuple] = []
    for r in rows:
        qid = int(r["question_id"])
        if q_start is not None and qid < q_start:
            continue
        if q_end is not None and qid > q_end:
            continue
        n += 1
        gold = str(r.get("gold_answer_letter") or "").strip().upper()
        choices = r.get("choices") or None
        old_ok = bool(r.get("correct"))
        letter = rescore_letter(r)
        if not letter:
            blank += 1
        new_ok = bool(letter and gold and grade_mcq_answer(letter, gold, choices=choices))
        old_c += int(old_ok)
        new_c += int(new_ok)
        if old_ok != new_ok:
            flips.append(
                (
                    qid,
                    gold,
                    str(r.get("parsed_answer_letter") or "")[:1],
                    letter or "-",
                    old_ok,
                    new_ok,
                )
            )
    return {
        "path": path,
        "n": n,
        "old_correct": old_c,
        "new_correct": new_c,
        "blank": blank,
        "flips": flips,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+")
    parser.add_argument("--q-start", type=int, default=None)
    parser.add_argument("--q-end", type=int, default=None)
    parser.add_argument("--show-flips", action="store_true")
    args = parser.parse_args()

    print(f"{'file':<52} {'n':>4} {'old':>8} {'new':>8} {'Δ':>5} {'blank':>6}")
    print("-" * 90)
    for p in args.jsonl:
        s = rescore_file(p.expanduser(), args.q_start, args.q_end)
        delta = s["new_correct"] - s["old_correct"]
        print(
            f"{p.name:<52} {s['n']:>4} "
            f"{s['old_correct']:>3}/{s['n']:<3} "
            f"{s['new_correct']:>3}/{s['n']:<3} "
            f"{delta:>+5} {s['blank']:>6}"
        )
        if args.show_flips and s["flips"]:
            gains = [f for f in s["flips"] if not f[4] and f[5]]
            losses = [f for f in s["flips"] if f[4] and not f[5]]
            if gains:
                print(f"  gains ({len(gains)}): " + ", ".join(f"Q{q}" for q, *_ in gains[:12]))
            if losses:
                print(f"  losses ({len(losses)}): " + ", ".join(f"Q{q}" for q, *_ in losses[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
