#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026

"""Summarize HM-EQA probe/holdout runs and enforce the balanced-32 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.eval.agentic_metrics import balanced32_gate, summarize_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--require-balanced32-gate",
        action="store_true",
        help="Exit nonzero unless the combined ladder has verified answers and no forced submits",
    )
    args = parser.parse_args()
    reports = [summarize_run(path) for path in args.run_dirs]
    combined_episodes = [
        episode for report in reports for episode in report["episodes"]
    ]
    from emet.eval.agentic_metrics import summarize_policy_metrics

    combined = {
        "runs": reports,
        "summary": summarize_policy_metrics(combined_episodes),
    }
    passed, reasons = balanced32_gate(combined)
    combined["balanced32_gate"] = {"passed": passed, "reasons": reasons}
    text = json.dumps(combined, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if args.require_balanced32_gate and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
