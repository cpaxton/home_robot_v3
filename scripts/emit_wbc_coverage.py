# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Emit strict coverage-gate results from the shipped WBC evaluators.

``eval_wbc_policy.py`` writes per-seed subtask results and a manifest.
``eval_wbc_hard_cases.py`` writes per-case hard-posture results. The coverage
gate expects one row per coverage case with integer episodes, success rate,
collisions, mean tracking error, and duration. This script maps registry cases
to those reports and refuses to guess: missing or non-scalar metrics raise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.controller.manipulation.wbc_robots import get_wbc_robot
from emet.rl.manipulation.wbc_coverage import (
    hard_case_metrics,
    load_coverage,
    task_result_metrics,
)
from emet.rl.manipulation.wbc_dagger import default_subtasks


def _task_for_source(source: str, robot: str) -> str:
    text = source.lower()
    known = {task.name for task in default_subtasks(robot)}
    for task in (
        "combined_left",
        "combined_right",
        "arm_left",
        "arm_right",
        "combined",
        "arm",
        "base",
    ):
        if task in text:
            if task not in known:
                raise ValueError(
                    f"coverage source {source!r} maps to subtask {task!r}, "
                    f"which is not in the {robot} suite {sorted(known)}"
                )
            return task
    raise ValueError(f"cannot map coverage source to a WBC subtask: {source!r}")


def _load_policy_report(path: Path) -> dict[str, dict]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("seed_results"), dict):
        raise ValueError(f"{manifest_path} has no seed_results mapping")
    return manifest["seed_results"]


def _load_policy_reports(specs: list[str]) -> dict[str, dict[str, dict]]:
    """Load ``PATH`` or ``ROBOT=PATH`` policy reports.

    A plain path applies to every case; a robot-qualified path lets one gate
    consume the per-robot held-out reports without merging their JSON by hand.
    """
    reports: dict[str, dict[str, dict]] = {}
    for spec in specs:
        if "=" in spec:
            robot_raw, path_raw = spec.split("=", 1)
            robot = get_wbc_robot(robot_raw).name
            key = robot
        else:
            path_raw = spec
            key = ""
        if key in reports:
            raise ValueError(f"duplicate policy report for {key or 'all robots'}")
        reports[key] = _load_policy_report(Path(path_raw))
    return reports


def _hard_case_result(report: dict, source: str) -> dict:
    suffix = source.rsplit("#", 1)[-1].strip()
    results = report.get("results") if isinstance(report, dict) else None
    if not isinstance(results, dict):
        raise ValueError("hard-case report has no results mapping")
    matches = [value for name, value in results.items() if suffix and suffix in name]
    if len(matches) != 1:
        raise ValueError(f"hard-case report has {len(matches)} matches for {suffix!r}")
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert WBC evaluator reports to coverage-gate rows")
    parser.add_argument("--cases", type=Path, required=True, help="coverage registry JSON")
    parser.add_argument(
        "--policy-report",
        action="append",
        default=[],
        metavar="[ROBOT=]PATH",
        help="directory with manifest.json or the manifest path; repeat with ROBOT= for a multi-robot gate",
    )
    parser.add_argument("--hard-case-report", type=Path, help="eval_wbc_hard_cases.py JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = load_coverage(args.cases)
    policy_reports = _load_policy_reports(args.policy_report)
    hard_report = (
        json.loads(args.hard_case_report.expanduser().read_text(encoding="utf-8")) if args.hard_case_report else None
    )
    rows: dict[str, dict] = {}
    for case in cases:
        if case.status != "active":
            continue
        if case.family == "hard_posture":
            if hard_report is None:
                raise ValueError(f"{case.id} requires --hard-case-report")
            rows[case.id] = hard_case_metrics(_hard_case_result(hard_report, case.source))
        elif case.family in {"free_space_reach", "base_arm"}:
            seed_results = policy_reports.get(case.robot) or policy_reports.get("")
            if seed_results is None:
                raise ValueError(f"{case.id} requires --policy-report for {case.robot!r}")
            rows[case.id] = task_result_metrics(seed_results, _task_for_source(case.source, case.robot))
        else:
            raise ValueError(f"{case.id}: no result emitter for active family {case.family!r}")
    if not rows:
        raise ValueError("no active coverage cases were emitted")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} coverage rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
