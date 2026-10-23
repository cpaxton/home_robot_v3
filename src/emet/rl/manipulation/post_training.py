"""Local post-training audit: fresh paired tests, oracle plans, and risk fitting.

Selection uses validation only. Fresh test results never change the controller.
No checkpoints/reports are uploaded and no hardware promotion occurs here.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from emet.rl.manipulation.env import SCENARIOS
from emet.rl.manipulation.evaluation import evaluate, score
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.risk import fit_risk
from emet.rl.manipulation.tamp_cases import generate


def curriculum_gates(teacher, learned):
    if set(teacher) != set(SCENARIOS) or set(learned) != set(SCENARIOS):
        return ["missing/unknown scenario categories"]
    reasons = []
    for name, row in learned.items():
        base = teacher[name]
        if min(row["episodes"], base["episodes"]) < 100:
            reasons.append(f"{name}: fewer than 100 paired test episodes")
        if row["success_rate"] < max(0.95, base["success_rate"]):
            reasons.append(f"{name}: success below target/teacher")
        for key in ("wheel_lift_rate", "unsafe_contact_rate", "initialization_failure_rate"):
            if row[key] != 0:
                reasons.append(f"{name}: {key} not zero")
        for key in ("joint_error_mean", "peak_contact_force_p95", "peak_measured_jerk_p95"):
            if row[key] is None or base[key] is None or row[key] > base[key]:
                reasons.append(f"{name}: {key} worse than teacher or unsupported")
    return reasons


def component_gates(teacher, learned):
    reasons = []
    for scenario in ("reach", "carry"):
        groups = []
        for report in (teacher, learned):
            groups.append(
                [
                    r
                    for r in report["cases"]
                    if r["scenario"] == scenario
                    and not r.get("initialization_failed")
                    and r.get("waypoints") is not None
                ]
            )
        if not all(groups):
            reasons.append(f"TAMP {scenario}: no executable-path evidence")
            continue
        rates = [sum(bool(r["execution_verified"]) for r in group) / len(group) for group in groups]
        if rates[1] < max(0.95, rates[0]):
            reasons.append(f"TAMP {scenario}: execution success below target/teacher")
        if any(
            r.get("controller_failure") in {"unsafe_contact", "wheel_lift"}
            or r.get("contact_metrics", {}).get("wheel_lift") is not False
            for r in groups[1]
        ):
            reasons.append(f"TAMP {scenario}: unsafe or unsupported execution")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--wait-seconds", type=float, default=86400)
    args = parser.parse_args()
    if args.episodes < 100 or args.wait_seconds <= 0:
        parser.error("at least 100 episodes/category and a positive wait budget required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def status(phase, **kwargs):
        write("status.json", {"phase": phase, "updated_at": time.time(), **kwargs})

    try:
        status("waiting_for_training", seeds=args.seeds)
        deadline = time.monotonic() + args.wait_seconds
        directories = [args.root / f"seed-{seed}" for seed in args.seeds]
        while not all((p / "complete.json").exists() for p in directories):
            if time.monotonic() >= deadline:
                raise TimeoutError("training completion deadline exceeded; no passing result assumed")
            time.sleep(20)
        candidates = [(p / "best", json.loads((p / "best/validation.json").read_text())) for p in directories]
        selected, validation = max(candidates, key=lambda item: score(item[1]["summary"]))
        policy = ManipulationPolicy(selected, robot="stretch")
        write(
            "selection.json",
            {
                "policy": str(selected),
                "selection_split": "validation",
                "validation": validation["summary"],
                "weights_sha256": policy.manifest["weights_sha256"],
            },
        )
        status("fresh_teacher_evaluation")
        teacher = evaluate(episodes=args.episodes, seed=500000)
        write("teacher.json", teacher)
        status("fresh_policy_evaluation")
        learned = evaluate(policy, episodes=args.episodes, seed=500000)
        write("learned.json", learned)
        status("tamp_component_execution")
        tamp_teacher = generate(episodes=10, seed=650000, execute=True)
        tamp_learned = generate(episodes=10, seed=650000, execute=True, policy=policy)
        write("tamp-teacher.json", tamp_teacher)
        write("tamp-learned.json", tamp_learned)
        status("oracle_public_tool_execution")
        script = Path(__file__).resolve().parents[4] / "scripts/eval_manipulation_tamp_oracle.py"
        with (args.output / "oracle.log").open("w") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--native-root",
                    str(args.native_root),
                    "--output",
                    str(args.output / "oracle"),
                    "--controller",
                    "policy",
                    "--policy",
                    str(selected),
                    "--cases",
                    "familiar",
                    "left",
                    "right",
                    "generated",
                    "--repeats",
                    "3",
                ],
                stdout=log,
                stderr=log,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            )
        if completed.returncode not in (0, 1) or not (args.output / "oracle/summary.json").exists():
            raise RuntimeError("oracle harness failed; inspect oracle.log")
        oracle = json.loads((args.output / "oracle/summary.json").read_text())
        reasons = curriculum_gates(teacher["summary"], learned["summary"])
        reasons.extend(component_gates(tamp_teacher, tamp_learned))
        if completed.returncode != 0 or len(oracle["cases"]) != 12 or not all(r["passed"] for r in oracle["cases"]):
            reasons.append("oracle full-task acceptance failed or incomplete")
        write(
            "acceptance.json",
            {
                "simulation_gates_passed": not reasons,
                "reasons": reasons,
                "hardware_promoted": False,
                "selected_policy": str(selected),
            },
        )
        status("collecting_separate_risk_outcomes")
        outcomes = evaluate(policy, episodes=args.episodes, seed=800000)
        write("risk-outcomes.json", outcomes)
        rows = [r for r in outcomes["episodes"] if not r["initialization_failed"]]
        status("risk_calibration_and_audit")
        if len(rows) >= 100:
            manifest = fit_risk(
                rows,
                args.output / "risk",
                robot="stretch",
                fingerprint=policy.manifest["fingerprint"],
                controller_id=policy.manifest["weights_sha256"],
            )
            write("risk-heads.json", manifest["heads"])
        else:
            write("risk-heads.json", {"status": "unknown", "reason": "insufficient executable initializations"})
        status("complete", simulation_gates_passed=not reasons, hardware_promoted=False)
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
