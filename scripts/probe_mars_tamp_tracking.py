# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Fixed24-episode follow-up: reset settling and frozen trajectory controllers."""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from summarize_mars_current_feedback import compact_primitive
from validate_mars_current_feedback import LEGACY, REFERENCE, PrimitiveBundle, sha, write

from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.mastery import save_report
from emet.rl.manipulation.policy import load_control_policy
from emet.rl.manipulation.whole_body_training import load_development, primitive_rollout
from emet.rl.regression_gate import verify_regressions

ROOT = Path(__file__).resolve().parents[1]
VELOCITY = Path("logs/manipulation/mars-frozen-base-arm-followthrough-v1/comparison/velocity-seed-90/last")
VELOCITY_SHA256 = "38c291b84c3b898fe40b55b6d257801db1c344c04a2b28bf9446daf607b4b21a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("logs/eval/mars-current-feedback-validation-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline_path = args.baseline_root / "baseline/summary.json"
    baseline = json.loads(baseline_path.read_text())
    if not baseline["complete"] or baseline["source_hashes"] != source_hashes():
        raise ValueError("complete same-runtime frozen baseline required")
    receipt = args.baseline_root / "full.receipt.json"
    verify_regressions(receipt)
    if sha(VELOCITY / "policy.zip") != VELOCITY_SHA256:
        raise ValueError("predeclared frozen velocity actor differs")
    development, _ = load_development()
    cases = development["train_cases"] + development["audit_cases"]
    if len(cases) != 6:
        raise ValueError("the same six existing TAMP cases are required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    summary = {
        "format": "mars-tamp-tracking-followup-v1",
        "producer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": sha(__file__),
        "baseline_summary_sha256": sha(baseline_path),
        "full_regression_receipt_sha256": sha(receipt),
        "source_hashes": sources,
        "groups": {},
        "new_ppo_steps": 0,
        "hardware_used": False,
        "qualified": False,
        "defaults_changed": False,
        "initial_state_hash_convention": "worker initial_state_sha256 precedes optional settling; do not claim equal post-conditioning states across settling settings",
    }
    # Existing zero-settle teacher/legacy outcomes are the comparison anchors;
    # only these four predefined groups add native episodes (4*6=24).
    groups = (
        ("teacher_settle2", None, 2.0),
        ("legacy_settle2", LEGACY, 2.0),
        ("velocity_legacy", VELOCITY, 0.0),
        ("gru_candidate", REFERENCE / "candidate/last/actor", 0.0),
    )
    try:
        for name, path, settling in groups:
            policy = load_control_policy(path) if path else None
            adapter = PrimitiveBundle(policy) if policy else None
            record = {
                "initial_settle_s": settling,
                "weights_sha256": policy.manifest["weights_sha256"] if policy else None,
                "cases": [],
                "complete": False,
            }
            summary["groups"][name] = record
            raw_rows = []
            for case in cases:
                (row,) = primitive_rollout(
                    [case], policy=adapter, initial_settle_s=settling, wheel_feedback_current_velocity=True
                )
                raw_rows.append(row)
                record["cases"].append(compact_primitive(row))
                save_report(args.output / (name + "-progress.json.gz"), {"rows": raw_rows})
                write(args.output / "progress.json", summary)
            archive = args.output / (name + ".json.gz")
            save_report(archive, {"rows": raw_rows})
            record.update(
                complete=True,
                archive_sha256=sha(archive),
                successes=sum(row["success"] for row in raw_rows),
                confirmed_stops=sum(row.get("stop_confirmed") is True for row in raw_rows),
            )
        if source_hashes() != sources:
            raise ValueError("runtime sources changed during follow-up")
        summary["complete"] = True
    except BaseException as exc:
        summary.update(complete=False, error=repr(exc))
        raise
    finally:
        write(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
