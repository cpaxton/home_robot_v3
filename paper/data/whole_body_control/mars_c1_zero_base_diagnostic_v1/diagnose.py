# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Declared all-case C1 command intervention; no PPO, export or qualification."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.basic_learning import load_basic_learning
from emet.rl.manipulation.mastery import evaluate_suite, read_report, save_report, summarize
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_training import load_development
from emet.rl.regression_gate import verify_regressions

OUT = Path(__file__).resolve().parent
PLAN = json.loads((OUT / "plan.json").read_text())
ROOT = Path(PLAN["runtime"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class InitialZeroBase(WholeBodyBundle):
    def __init__(self):
        super().__init__(PLAN["final_actor"])
        self.initial = WholeBodyBundle(PLAN["initial_actor"])
        self.intervention_calls = 0

    def bind_reference(self, reference):
        super().bind_reference(reference)
        self.initial.bind_reference(reference)

    def __call__(self, observation):
        action = super().__call__(observation)
        anchor = self.initial(observation)
        if np.count_nonzero(anchor[:2]):
            raise RuntimeError("initial base function is not zero as declared")
        action[:2] = anchor[:2]
        self.intervention_calls += 1
        return action


def compact(row):
    return {
        "id": row["id"], "stage": row["stage"], "split": row["split"],
        "mastered": row["mastered"], "tracking_and_settling_pass": row["tracking_and_settling_pass"],
        "smoothness_pass": row["smoothness_pass"], "failure_reasons": row["failure_reasons"],
        "code": row["code"], "metrics_measured_at_stop": row["metrics"],
        "reference_complete": row["final_info"].get("reference_complete"),
        "initial_state_sha256": row["admission"]["initial_state_sha256"],
        "reference_sha256": row["admission"]["reference_sha256"],
    }


def main():
    if os.sched_getaffinity(0) != {15} or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("declared CPU15/CUDA-hidden resources required")
    torch.set_num_threads(1)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != PLAN["implementation_revision"]:
        raise RuntimeError("source snapshot changed")
    verify_regressions(PLAN["regression_receipt"])
    if sha(PLAN["basic_cases"]) != PLAN["basic_cases_sha256"]:
        raise RuntimeError("case manifest changed")
    files = [Path(PLAN[k]) / name for k in ("initial_actor", "final_actor") for name in ("manifest.json", "policy.zip")]
    files += [Path(PLAN[k]) for k in ("initial_report", "final_report", "teacher_report")]
    bindings = {str(path): sha(path) for path in files}
    _, general = load_development(ROOT / "configs/benchmarks/mars_whole_body_v2_development.json")
    _, _, suite = load_basic_learning(PLAN["basic_cases"], general)
    teacher = read_report(PLAN["teacher_report"])
    originals = {mode: {row["id"]: row for row in read_report(PLAN[mode + "_report"])["cases"]}
                 for mode in ("initial", "final")}
    result = dict(plan=PLAN, script_sha256=sha(__file__), source_artifacts=bindings, complete=False,
                  ppo_steps=0, actor_exported=False, hardware_used=False, safety_promoted=False,
                  rows=[], replication=[])
    for mode in PLAN["modes"]:
        policy = InitialZeroBase() if mode == "final_arm_initial_zero_base" else WholeBodyBundle(PLAN[mode + "_actor"])
        report = evaluate_suite(suite, stages=("base", "arm"), policy=policy, teacher_report=teacher)
        if len(report["cases"]) != 12:
            raise RuntimeError("declared case population changed")
        report.update(diagnostic_only=True, actor_exported=False, scope=PLAN["scope"],
                      intervention=mode, script_sha256=result["script_sha256"])
        save_report(OUT / f"{mode}.json.gz", report)
        if mode in originals:
            for row in report["cases"]:
                old = originals[mode][row["id"]]
                identity_matches = all(row["admission"][key] == old["admission"][key]
                                       for key in ("initial_state_sha256", "reference_sha256"))
                flags_match = all(row[key] == old[key] for key in
                                  ("mastered", "tracking_and_settling_pass", "smoothness_pass", "failure_reasons", "code"))
                jerk_delta = max(abs(row["metrics"][key] - old["metrics"][key]) for key in
                                 ("base_linear_jerk", "base_yaw_jerk", "joint_jerk"))
                result["replication"].append(dict(mode=mode, id=row["id"], identity_matches=identity_matches,
                                                  flags_match=flags_match, max_jerk_abs_delta=jerk_delta))
                if not identity_matches or not flags_match or jerk_delta != 0:
                    write(OUT / "results.json", result)
                    raise RuntimeError("unchanged replay mismatch; no causal interpretation")
        else:
            result["intervention_calls"] = policy.intervention_calls
            for row in report["cases"]:
                old = originals["final"][row["id"]]
                if any(row["admission"][key] != old["admission"][key] for key in
                       ("initial_state_sha256", "reference_sha256")):
                    raise RuntimeError("intervention case identity changed")
        result["rows"].append(dict(mode=mode, summary=summarize(report), cases=[compact(row) for row in report["cases"]]))
        write(OUT / "results.json", result)
        if job := os.environ.get("EMET_JOB_ID"):
            from emet.utils.job_registry import update_job

            update_job(job, phase=mode, units_done=len(result["rows"]) * 12, units_total=36)
    if any(sha(path) != checksum for path, checksum in bindings.items()):
        raise RuntimeError("original artifacts changed")
    result["complete"] = True
    write(OUT / "results.json", result)
    print("Diagnostic complete: 36 rollouts; no PPO or actor export.", flush=True)


if __name__ == "__main__":
    main()
