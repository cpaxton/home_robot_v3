#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Summarize the complete declared development cohort, keeping faults separate."""

import argparse
import json
from pathlib import Path


def summarize(suite, output):
    declared = {row["id"]: row for row in suite["cases"]}
    observed = {}
    for path in sorted(output.glob("lane-*.json")):
        manifest = json.loads(path.read_text())
        for row in manifest["cases"]:
            key = row["id"]
            if key not in declared or key in observed or row["case"] != declared[key]:
                raise ValueError("inconsistent_regression_manifest")
            observed[key] = row
    rows = []
    for key, case in declared.items():
        row = observed.get(key, {})
        result = row.get("result") or {}
        raw = output / key / "result.json"
        if not result and raw.exists():
            result = json.loads(raw.read_text())
        receipts = result.get("responses", [])
        last = receipts[-1] if receipts else {}
        detail = last.get("data", {})
        status = row.get("status", "pending")
        if status == "failed" and row.get("exit_code", 0) < 0:
            status = "native_crash"
        rows.append(
            {
                "id": key,
                "kind": "negative_control" if case.get("fault") else "task",
                "status": status,
                "exit_code": row.get("exit_code"),
                "runtime_sha256": row.get("runtime_sha256"),
                "code": last.get("code"),
                "completed_ops": detail.get("completed_ops"),
                "failed_stage": detail.get("failed_stage"),
                "execution_started": detail.get("execution_started"),
                "recovery": last.get("recovery"),
                "error": result.get("error"),
                "last_phase": result.get("phases", [None])[-1],
                "peak_process_rss_bytes": result.get("cumulative_peak_rss_bytes"),
                "map_store": result.get("map_store"),
            }
        )
    return {
        "schema_version": 1,
        "scope": suite["scope"],
        "benchmark_denominator": None,
        "task_passed": sum(r["kind"] == "task" and r["status"] == "passed" for r in rows),
        "task_declared": sum(r["kind"] == "task" for r in rows),
        "negative_controls_verified": sum(r["kind"] == "negative_control" and r["status"] == "passed" for r in rows),
        "negative_controls_declared": sum(r["kind"] == "negative_control" for r in rows),
        "unfinished": sum(r["status"] in {"pending", "running"} for r in rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/eval/mobile_map_regression_20261117.json"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.suite.read_text()), args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
