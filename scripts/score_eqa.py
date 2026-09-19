#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Score a directory of per-question HM-EQA result JSONL files.

Each ``q<id>.jsonl`` holds one serialized ``emet.habitat.metrics.EpisodeMetrics``
object. Prints overall accuracy, the holdout-8 / balanced-32 split, and the
commit-rate decomposition by ``answer_provenance``.

The commit-rate decomposition is the hill-climb signal: ``vlm_suggested`` means
the model *committed* (retrieval succeeded well enough to answer); ``mcq_debias``
/ ``uniform_prior`` mean the harness forced a guess (retrieval failed to surface
evidence). We climb commit rate; given-commit accuracy is the accepted ceiling.

Usage:
    python scripts/score_eqa.py OUT_DIR
    python scripts/score_eqa.py OUT_DIR --json   # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from emet.eval.harness import DEFAULT_BAL32_IDS, DEFAULT_HOLDOUT8_IDS
except Exception:  # pragma: no cover - fall back to the frozen paper lists
    DEFAULT_HOLDOUT8_IDS = "15,56,65,68,79,88,104,105"
    DEFAULT_BAL32_IDS = "2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84"

HOLDOUT8 = {int(x) for x in DEFAULT_HOLDOUT8_IDS.split(",")}
BAL32 = {int(x) for x in DEFAULT_BAL32_IDS.split(",")}

# Committed vs forced provenance, used for the commit-rate summary.
COMMITTED = {"vlm_suggested", "eqa_answer"}
FORCED = {"mcq_debias", "uniform_prior"}


def _parse_set(ids: str) -> set[int]:
    return {int(x) for x in (ids or "").replace(",", " ").split() if x}


def load_rows(out_dir: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in sorted(out_dir.glob("q*.jsonl")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Per-question files hold exactly one metrics object, so only the LAST
        # line of a q*.jsonl is scored. If a driver ever appends several rows
        # (e.g. retries in one file), earlier rows are silently ignored — that
        # drops a question from n rather than double-counting it. Empty files
        # are treated as a native crash (missing run), not a scored miss.
        data = json.loads(text.splitlines()[-1])
        rows[int(data["question_id"])] = data
    return rows


def _acc(correct: int, total: int) -> str:
    return f"{correct}/{total} ({100.0 * correct / total:.1f}%)" if total else "n/a"


def _bucket_acc(correct: int, total: int) -> str:
    return f"{100.0 * correct / total:.1f}%" if total else "-"


def summarize(rows: dict[int, dict], *, out_dir: Path | None = None) -> dict:
    by_prov: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    by_set: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    total = correct = 0
    flips: list[dict] = []
    for qid, row in sorted(rows.items()):
        c = bool(row.get("correct"))
        prov = str(row.get("answer_provenance") or "unknown")
        total += 1
        correct += int(c)
        by_prov[prov]["n"] += 1
        by_prov[prov]["correct"] += int(c)
        if qid in HOLDOUT8:
            bucket = "holdout8"
        elif qid in BAL32:
            bucket = "bal32"
        else:
            bucket = "other"
        by_set[bucket]["n"] += 1
        by_set[bucket]["correct"] += int(c)
        flips.append(
            {
                "qid": qid,
                "pred": str(row.get("predicted_answer") or "")[:40],
                "gold": str(row.get("gold_answer_letter") or ""),
                "correct": c,
                "provenance": prov,
                "error": str(row.get("error") or ""),
            }
        )

    committed_n = sum(by_prov.get(p, {"n": 0})["n"] for p in COMMITTED)
    committed_correct = sum(by_prov.get(p, {"correct": 0})["correct"] for p in COMMITTED)
    forced_n = sum(by_prov.get(p, {"n": 0})["n"] for p in FORCED)
    forced_correct = sum(by_prov.get(p, {"correct": 0})["correct"] for p in FORCED)
    errored = sum(1 for f in flips if f["error"])
    # Errored rows stay in the headline accuracy (flagged, not hidden); this is
    # the supplementary view with them excluded entirely.
    clean = {
        "n": total - errored,
        "correct": sum(1 for f in flips if f["correct"] and not f["error"]),
    }

    return {
        "total": total,
        "correct": correct,
        "n_errored": errored,
        "clean": clean,
        "by_prov": {p: dict(v) for p, v in sorted(by_prov.items())},
        "by_set": {s: dict(v) for s, v in sorted(by_set.items())},
        "committed": {"n": committed_n, "correct": committed_correct},
        "forced": {"n": forced_n, "correct": forced_correct},
        "flips": flips,
        "out_dir": str(out_dir) if out_dir else None,
    }


def print_summary(summary: dict) -> None:
    print(f"source: {summary['out_dir']}")
    total = summary["total"]
    correct = summary["correct"]
    print(f"overall: {_acc(correct, total)}")
    if summary.get("n_errored"):
        print(f"ERRORED (crashed/exception, not a clean miss): {summary['n_errored']}")
        clean = summary["clean"]
        print(f"clean accuracy (errored excluded): {_acc(clean['correct'], clean['n'])}")
    print()
    print("split:")
    for name in ("holdout8", "bal32", "other"):
        s = summary["by_set"].get(name)
        if s:
            print(f"  {name:9s} {_acc(s['correct'], s['n'])}")
    print()
    print("commit-rate decomposition (provenance):")
    for prov, v in summary["by_prov"].items():
        tag = "committed" if prov in COMMITTED else ("forced" if prov in FORCED else "other")
        print(f"  {prov:16s} n={v['n']:2d}  correct={_acc(v['correct'], v['n']):22s} [{tag}]")
    committed = summary["committed"]
    forced = summary["forced"]
    print()
    print(f"  committed (vlm_suggested+eqa_answer): {_acc(committed['correct'], committed['n'])}")
    print(f"  forced    (mcq_debias+uniform_prior): {_acc(forced['correct'], forced['n'])}")
    print()
    print("per-question (pred/gold/correct/provenance):")
    for f in summary["flips"]:
        mark = "ERR" if f["error"] else ("OK " if f["correct"] else "XX ")
        err = f"  ERROR={f['error']}" if f["error"] else ""
        print(f"  q{f['qid']:>3d}  {mark} pred={f['pred']!r} gold={f['gold']} [{f['provenance']}]{err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Directory of q*.jsonl episode results")
    parser.add_argument("--json", action="store_true", help="Emit a single JSON summary object")
    args = parser.parse_args()

    rows = load_rows(args.out_dir)
    if not rows:
        print(f"no q*.jsonl results found under {args.out_dir}", file=sys.stderr)
        return 2
    summary = summarize(rows, out_dir=args.out_dir)
    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
