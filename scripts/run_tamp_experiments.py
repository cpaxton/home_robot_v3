#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Serial, resumable TAMP experiment accounting; task success is never process success.

Run from a frozen checkout via ``emet jobs run --cpu-safe --gpu-exclusive``.
Dry-run is read-only. Every scheduled case has a durable status, including cases
not run because cleanup could not be confirmed after a timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_cases(suite, python=sys.executable):
    script = str(ROOT / "scripts/eval_tamp_clutter.py")
    if suite == "protocol":
        return [
            {
                "id": f"protocol_{robot}_{scene}",
                "execution_mode": "oracle_teleport",
                "command": [
                    python,
                    script,
                    "--test-battery",
                    "--battery-robots",
                    robot,
                    "--battery-scenes",
                    str(scene),
                ],
                "expected_episodes": 4,
            }
            for robot in ("nori", "innate_mars", "rby1")
            for scene in (0, 1)
        ]
    if suite in ("small", "full"):
        registry = (
            ROOT / "configs/ovmm" / ("clutter_episodes_large.yaml" if suite == "full" else "clutter_episodes.yaml")
        )
        rows = yaml.safe_load(registry.read_text())["episodes"]
        from emet.eval.tamp_clutter import ROBOT_DEFAULT_MANIP_MODE

        cases = []
        for row in rows:
            mode = row.get("manip_mode") or ROBOT_DEFAULT_MANIP_MODE[row["robot"]]
            cases.append(
                {
                    "id": row["id"],
                    "robot": row["robot"],
                    "scene_index": row.get("scene_index", 0),
                    "execution_mode": {
                        "sim": "oracle_teleport",
                        "latch": "kinematic_latch",
                        "attempt": "physical_attempt",
                    }[mode],
                    "expected_episodes": 1,
                    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                    "command": [python, script, "--episodes", str(registry), "--episode-id", row["id"]],
                }
            )
        return cases
    if suite == "floor":
        rows = yaml.safe_load((ROOT / "configs/ovmm/full_episodes.yaml").read_text())["episodes"]
        return [
            {
                "id": row["id"],
                "execution_mode": "gt_floor_control",
                "expected_episodes": 1,
                "command": [python, str(ROOT / "scripts/eval_tamp_floor.py"), "--gt-only", "--episode-id", row["id"]],
            }
            for row in rows
            if row.get("floor_object")
        ]
    raise ValueError(suite)


def summarize_case(case, directory, returncode):
    directory = Path(directory)
    if case["id"].startswith("protocol_"):
        path = directory / "battery_summary.json"
    elif case["execution_mode"] == "gt_floor_control":
        path = directory / (case["id"] + ".json")
    else:
        path = directory / (case["id"] + ".json")
    if not path.exists():
        return {"status": "error", "task_success": False, "reason": "missing_result_artifact", "exit_code": returncode}
    result = json.loads(path.read_text())
    if case["id"].startswith("protocol_"):
        passed = len(result.get("tests", [])) == case["expected_episodes"] and all(
            r.get("pass") for r in result["tests"]
        )
        return {
            "status": "completed" if returncode == 0 else "failed",
            "task_success": bool(passed),
            "evidence": str(path),
            "exit_code": returncode,
            "battery": result,
        }
    success = bool(result.get("task_success", False))
    status = result.get("status")
    if status not in ("unsupported_capability", "deferred_gt_only"):
        status = (
            "invalid_fixture" if result.get("skipped_invalid") else ("error" if result.get("error") else "completed")
        )
    return {"status": status, "task_success": success, "exit_code": returncode, "evidence": str(path), "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["protocol", "small", "full", "floor"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-timeout", type=float, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = build_cases(args.suite)
    if args.dry_run:
        print(
            json.dumps(
                {"suite": args.suite, "cases": cases, "expected_episodes": sum(c["expected_episodes"] for c in cases)},
                indent=2,
            )
        )
        return 0
    if not os.environ.get("EMET_JOB_ID"):
        raise SystemExit("Live experiments require emet jobs --cpu-safe --gpu-exclusive")
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = out / "manifest.json"
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("Use a clean frozen checkout for experiment runs")
    manifest = {
        "schema": 1,
        "suite": args.suite,
        "source_sha": source,
        "cases": cases,
        "episode_timeout_s": args.episode_timeout,
    }
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("Resume manifest differs from source/config/budget")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    ledger_path = out / "ledger.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if args.resume
        else {c["id"]: {"status": "pending", "task_success": False} for c in cases}
    )

    if args.resume:
        for row in ledger.values():
            if row["status"] == "running":
                row.update(status="interrupted", task_success=False)

    def persist():
        temp = ledger_path.with_suffix(".tmp")
        temp.write_text(json.dumps(ledger, indent=2) + "\n")
        temp.replace(ledger_path)

    persist()
    from emet.utils.process_tree import popen_session, terminate_process_tree

    for case in cases:
        if ledger[case["id"]]["status"] != "pending":
            continue  # Reruns belong in a fresh run; never erase an earlier failure.
        directory = out / case["id"]
        directory.mkdir(exist_ok=False)
        command = [*case["command"], "--output-dir", str(directory)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", EMET_UV_RUN="1")
        env.update(EMET_SIM_NAV_TELEPORT="1", EMET_MOLMOSPACES_NAV_TELEPORT="1")
        for key in (
            "EMET_PHYSICAL_EXECUTION",
            "EMET_PHYSICAL_AUDIT",
            "EMET_PHYSICAL_START_MARKER",
            "EMET_SIM_EVAL_CONFIG",
            "EMET_SIM_EVAL_TRACE",
        ):
            env.pop(key, None)
        ledger[case["id"]] = {
            "status": "running",
            "task_success": False,
            "execution_mode": case["execution_mode"],
            "command": command,
        }
        persist()
        start = time.monotonic()
        with (directory / "process.log").open("x") as log:
            proc = popen_session(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = proc.wait(timeout=args.episode_timeout * case["expected_episodes"])
                score = summarize_case(case, directory, returncode)
            except subprocess.TimeoutExpired:
                terminate_process_tree(proc, grace_s=20)
                score = {"status": "timeout", "task_success": False, "reason": "wall_timeout"}
            except Exception as exc:
                terminate_process_tree(proc, grace_s=20)
                score = {"status": "error", "task_success": False, "reason": str(exc)}
            finally:
                terminate_process_tree(proc)
        ledger[case["id"]].update(score, wall_s=time.monotonic() - start)
        persist()
        print(json.dumps({"case": case["id"], **score}), flush=True)
        # A timeout requires explicit simulator/job cleanup inspection before continuing.
        if score["status"] == "timeout":
            break
    summary = {
        "scheduled_cases": len(cases),
        "scheduled_episodes": sum(c["expected_episodes"] for c in cases),
        "terminal_cases": sum(r["status"] not in ("pending", "running") for r in ledger.values()),
        "successful_cases": sum(bool(r.get("task_success")) for r in ledger.values()),
        "status_counts": {
            s: sum(r["status"] == s for r in ledger.values()) for s in sorted({r["status"] for r in ledger.values()})
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return (
        0
        if summary["terminal_cases"] == len(cases)
        and not any(r["status"] in ("error", "timeout") for r in ledger.values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
