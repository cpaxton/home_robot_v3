#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Export auditable figure inputs; unattempted and unscored cases stay explicit."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def summarize(run):
    manifest = json.loads((run / "manifest.json").read_text())
    ledger = run / "results.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    indexed = {r["id"]: r for r in records}
    if len(indexed) != len(records) or indexed.keys() - {t["id"] for t in manifest["tasks"]}:
        raise ValueError("Duplicate or unassigned episode in ledger")
    table, groups = (
        [],
        defaultdict(lambda: {"assigned": 0, "pending": 0, "complete": 0, "errors": 0, "unscored": 0, "success": 0}),
    )
    for task in manifest["tasks"]:
        result = indexed.get(task["id"], {})
        if result and result.get("source_sha") != manifest["source_sha"]:
            raise ValueError("Mixed runtime sources in ledger")
        method = next(
            (m for m in ("lazy_graph", "dynamem", "static_graph", "dynagraph") if f"-{m}-" in task["id"]), "native"
        )
        metrics = result.get("metrics") or {}
        status = (
            "pending"
            if not result
            else "complete"
            if result.get("complete") is True
            else "errors"
            if result.get("complete") is False
            else "unscored"
        )
        bucket = groups[f"{task['family']}/{method}"]
        bucket["assigned"] += 1
        bucket[status] += 1
        bucket["success"] += int(result.get("success") is True)
        table.append(
            {
                "id": task["id"],
                "family": task["family"],
                "method": method,
                "status": status,
                "success": result.get("success"),
                "wall_s_including_supervisor": result.get("wall_s"),
                "sim_time_s": metrics.get("sim_time_s"),
                "question": metrics.get("question"),
                "answer": metrics.get("formatted_answer"),
                "question_id": metrics.get("question_id"),
                "graph_nodes": metrics.get("graph_nodes"),
                "observations": metrics.get("observations"),
                "find_object_success": metrics.get("find_object_success"),
                "find_recep_success": metrics.get("find_recep_success"),
                "debug_bundle_dir": metrics.get("debug_bundle_dir"),
                "topdown_map_path": metrics.get("topdown_map_path"),
                "diagnostics_manifest_path": metrics.get("diagnostics_manifest_path"),
                "metric_path": task.get("metric"),
                "source_sha": manifest["source_sha"],
            }
        )
    for bucket in groups.values():
        bucket["accuracy_among_complete"] = bucket["success"] / bucket["complete"] if bucket["complete"] else None
    return table, dict(groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    table, groups = summarize(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "groups": groups,
                "limits": [
                    "Partial/completed denominators shown separately; no aggregate across benchmark families",
                    "One seed/default recorded RNG, development populations, not superiority evidence",
                    "Null simulation time/visibility is unavailable, never inferred from wall time/outcomes",
                    "Current snapshot; pending assignments remain in the frozen population",
                ],
                "manifest_sha256": hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    with (args.output / "figure_inputs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(groups)
    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.35)))
    for index, label in enumerate(labels):
        g = groups[label]
        parts = [
            ("success", g["success"], "#3a9873"),
            ("task failure", g["complete"] - g["success"], "#cf7c48"),
            ("infrastructure", g["errors"], "#ad4c57"),
            ("unscored", g["unscored"], "#a1a0d0"),
            ("pending", g["pending"], "#dedede"),
        ]
        left = 0
        for name, n, color in parts:
            ax.barh(index, n, left=left, color=color, label=name if index == 0 else None)
            left += n
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Assigned cases (raw counts; benchmark families remain separate)")
    ax.set_title("Frozen benchmark refresh: task outcomes and completion status")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "completion_and_outcomes.png", dpi=170)
    plt.close(fig)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
