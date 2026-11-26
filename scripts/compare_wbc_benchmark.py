# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compare a WBC clean-suite rerun against the recorded multi-robot baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGGREGATE_TOL = 0.02
TASK_TOL = 0.05


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--aggregate-tol", type=float, default=AGGREGATE_TOL)
    parser.add_argument("--task-tol", type=float, default=TASK_TOL)
    parser.add_argument("--episodes", type=int, default=None, help="override the baseline episodes-per-subtask")
    parser.add_argument("--robots", default="", help="comma/space-separated robots; default all baseline robots")
    parser.add_argument("--seeds", default="", help="comma/space-separated seeds; default all baseline seeds")
    args = parser.parse_args(argv)

    baseline = _load(args.baseline)
    failures: list[str] = []
    requested = tuple(args.robots.replace(",", " ").split()) if args.robots else ()
    rows = baseline["robots"]
    if requested:
        rows = {robot: rows[robot] for robot in requested if robot in rows}
    print(f"{'robot':<16} {'seed':>6} {'agg':>7} {'base':>7} {'delta':>7}")
    for robot, row in rows.items():
        robot_dir = args.results / robot
        if not robot_dir.is_dir():
            failures.append(f"{robot}: no result directory {robot_dir}")
            continue
        episodes = int(args.episodes or baseline.get("protocol", {}).get("episodes_per_subtask", 50))
        requested_seeds = tuple(args.seeds.replace(",", " ").split()) if args.seeds else ()
        metrics = row["metrics"]
        if requested_seeds:
            metrics = {seed: metrics[seed] for seed in requested_seeds if seed in metrics}
        for seed, expected in metrics.items():
            path = robot_dir / f"heldout_{episodes}_seed{seed}.json"
            if not path.is_file():
                failures.append(f"{robot} seed {seed}: missing {path}")
                continue
            actual = _load(path)
            agg = float(actual["aggregate"]["success_rate"])
            base_agg = float(expected["aggregate"])
            delta = agg - base_agg
            print(f"{robot:<16} {seed:>6} {agg:7.3f} {base_agg:7.3f} {delta:+7.3f}")
            if delta < -args.aggregate_tol:
                failures.append(f"{robot} seed {seed}: aggregate {agg:.3f} < {base_agg:.3f} - {args.aggregate_tol:.2f}")
            for task, base_task in expected["tasks"].items():
                actual_task = actual.get(task)
                if not isinstance(actual_task, dict):
                    failures.append(f"{robot} seed {seed}: missing task {task!r}")
                    continue
                task_rate = float(actual_task["success_rate"])
                if task_rate < float(base_task) - args.task_tol:
                    failures.append(
                        f"{robot} seed {seed} {task}: {task_rate:.3f} < {float(base_task):.3f} - {args.task_tol:.2f}"
                    )
    if failures:
        print("\nFAILURES", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("\nPASS: all replayed seeds stay within the regression tolerances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
