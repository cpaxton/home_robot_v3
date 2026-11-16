# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Twelve fixed teacher episodes, comparing pregrasp and unloaded grasp pose."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.primitive_admission import unloaded_stage_readiness
from emet.rl.manipulation.whole_body_training import DEFAULT_SUITE, load_development, primitive_rollout, write_json
from emet.rl.regression_gate import ROOT, source_state

PRIMITIVES = ("pregrasp", "grasp_pose")


def run(output, *, planning_scope="full_task"):
    development, _ = load_development()
    cases = development["train_cases"] + development["audit_cases"]
    if len(cases) != 6:
        raise ValueError("this bounded protocol requires the original six development/audit cases")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = source_state(ROOT)
    write_json(output / "sources.json", before)
    report = {
        "format": "mars-unloaded-primitive-admission-v2",
        "planning_scope": planning_scope,
        "producer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_state_digest": digest(before),
        "source_manifest_sha256": hashlib.sha256((output / "sources.json").read_bytes()).hexdigest(),
        "suite_file_sha256": hashlib.sha256(DEFAULT_SUITE.read_bytes()).hexdigest(),
        "cases": cases,
        "wheel_feedback_current_velocity": True,
        "initial_settle_s": 0.0,
        "arrival_contract": "original_unmodified_stage_contracts",
        "expected_exceptions": {},
        "new_ppo_steps": 0,
        "hardware_used": False,
        "full_pick_place_qualified": False,
        "ready_for_training": False,
        "complete": False,
        "stages": {},
    }
    write_json(output / "results.json", report)
    started = time.monotonic()
    for primitive in PRIMITIVES:
        rows = primitive_rollout(
            cases,
            primitive=primitive,
            planning_scope=planning_scope,
            wheel_feedback_current_velocity=True,
            trace_dir=output / primitive,
        )
        # Commands stay in raw traces; hashes bind the full, unmodified rows.
        write_json(output / (primitive + ".json"), rows)
        report["stages"][primitive] = {
            "attempts": len(rows),
            "successes": sum(r["success"] for r in rows),
            "stops_confirmed": sum(r.get("stop_confirmed") is True for r in rows),
            "stage_teacher_ready": unloaded_stage_readiness(rows, cases, primitive),
            "rows_file_sha256": hashlib.sha256((output / (primitive + ".json")).read_bytes()).hexdigest(),
            "rows": [{k: v for k, v in row.items() if k != "commands"} for row in rows],
        }
        report["elapsed_wall_s"] = time.monotonic() - started
        write_json(output / "results.json", report)
    if before != source_state(ROOT):
        raise RuntimeError("source changed during unloaded primitive admission; results cannot be admitted")
    report["complete"] = True
    write_json(output / "results.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planning-scope", choices=("full_task", "unloaded_prefix"), default="full_task")
    args = parser.parse_args()
    result = run(args.output, planning_scope=args.planning_scope)
    print(
        json.dumps(
            {
                k: {n: v[n] for n in ("attempts", "successes", "stops_confirmed", "stage_teacher_ready")}
                for k, v in result["stages"].items()
            }
        ),
        flush=True,
    )
