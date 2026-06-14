#!/usr/bin/env python3
"""Summarize HM-EQA frontier ablation JSONL runs (rescored MCQ parsing)."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.rescore_habitat_jsonl import rescore_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.home() / ".cache/habitat_eqa/results",
    )
    parser.add_argument("--q-start", type=int, default=0)
    parser.add_argument("--q-end", type=int, default=19)
    parser.add_argument(
        "--family",
        default="qwen3_vl",
        help="Filename substring filter (e.g. qwen3_vl)",
    )
    args = parser.parse_args()

    pattern = f"ablation_*_{args.family}_q{args.q_start}-{args.q_end}.jsonl"
    files = sorted(args.results_dir.glob(pattern))
    if not files:
        print(f"No files matching {pattern} in {args.results_dir}")
        return 1

    print(f"Frontier ablation Q{args.q_start}-{args.q_end} (rescored)\n")
    print(f"{'arm':<12} {'n':>4} {'stored':>12} {'rescored':>12} {'blank':>6}")
    print("-" * 52)
    for path in files:
        arm = path.stem.split("_")[1] if path.stem.startswith("ablation_") else path.stem
        stats = rescore_file(path, args.q_start, args.q_end)
        stored = f"{stats['old_correct']}/{stats['n']}"
        new = f"{stats['new_correct']}/{stats['n']}"
        print(f"{arm:<12} {stats['n']:>4} {stored:>12} {new:>12} {stats['blank']:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
