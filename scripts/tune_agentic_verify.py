#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline sweep of agentic EQA verify thresholds / budgets from agentic_trace.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.eval.agentic_tuning import load_agentic_traces_dir, tune_from_traces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_root",
        type=Path,
        help="Path to agentic_trace.jsonl or a directory to search recursively",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write full JSON report here (default: stdout summary only)",
    )
    args = parser.parse_args()
    traces = load_agentic_traces_dir(args.trace_root)
    if not traces:
        print(f"No traces found under {args.trace_root}")
        return 1
    report = tune_from_traces(traces)
    best = report.get("best_threshold") or {}
    knee = report.get("budget_knee") or {}
    print(
        f"traces={len(traces)} verify={report['n_verify_rows']} summaries={report['n_summary_rows']}"
    )
    if best:
        print(
            f"best_threshold={best.get('threshold')} "
            f"f1={best.get('f1'):.3f} prec={best.get('precision'):.3f} "
            f"rec={best.get('recall'):.3f} n={int(best.get('n_labeled', 0))}"
        )
    else:
        print("best_threshold=(no gt_present labels — collect with EMET_EQA_TRACE=1 in sim)")
    if knee:
        print(
            f"budget_knee max_rounds={int(knee.get('max_rounds_cap', -1))} "
            f"accuracy={knee.get('accuracy'):.3f} mean_rounds={knee.get('mean_rounds'):.2f}"
        )
    print(f"nav_report={report.get('nav_report')}")
    print(f"router_report={report.get('router_report')}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
