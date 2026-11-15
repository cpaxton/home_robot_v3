# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Predeclared reset/motion/stop checks, then corrected-plant frozen baselines."""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.basic_learning import load_basic_learning
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.jerk_measurements import NativeJerkObserver
from emet.rl.manipulation.mastery import MasteryEnv, evaluate_suite, mastery_metrics, read_report, save_report
from emet.rl.manipulation.mastery_objective import smoothness_budgets, tracking_contract_passes
from emet.rl.manipulation.policy import load_control_policy
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.whole_body_training import load_development, primitive_rollout, tamp_training_readiness

ROOT = Path(__file__).resolve().parents[1]
BASES = ("hold", "forward", "reverse", "yaw_left", "yaw_right")
REFERENCE = Path("logs/manipulation/mars-reference-base-exploration-v1")
LEGACY = Path("logs/manipulation/mars-base-curriculum-legacy-v1-seed90/last/actor")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def loop_suite(general, enabled):
    """Twelve existing conditions per plant; no new reset seeds or geometry."""
    _, _, basic = load_basic_learning(Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json"), general)
    suite = copy.deepcopy(basic)
    suite["config"]["wheel_feedback_current_velocity"] = enabled
    by_id = {c["id"]: c for c in general["cases"]}
    rows = []
    for push in (False, True):
        for name in BASES:
            source = by_id["train:base:" + name]
            rows.append(
                {
                    **source,
                    "source_case_id": source["id"],
                    "id": source["id"] + (":push" if push else ":nominal"),
                    "probe_push": push,
                }
            )
    for case_id in ("train:base:forward", "train:arm:pitch_up@amplitude:0.25"):
        source = next(c for c in basic["cases"] if c["id"] == case_id)
        rows.append(
            {**source, "source_case_id": case_id, "id": case_id + ":cancel", "probe_push": False, "cancel_tick": 20}
        )
    suite["cases"] = rows
    return suite


def measured_stop(env):
    """Bounded simulated stop; no reset, rearming or grip release."""
    env.control.stop()
    # The finite external test push ends before stop verification. Contact and
    # gravity remain live. Do not claim this is a sustained-force stop test.
    env.data.xfrc_applied[:] = 0
    start = float(env.data.time)
    trace = []
    for _ in range(16):
        trace.append(
            {
                "time": float(env.data.time),
                "arm_velocity": env.control.velocities().copy().tolist(),
                "base_qvel": env.data.qvel[env.control.base_dof : env.control.base_dof + 6].copy().tolist(),
            }
        )
        if whole_body_stopped(env.control):
            return {"confirmed": True, "elapsed_sim_s": float(env.data.time) - start, "trace": trace}
        if env._physics_invalid or float(env.data.time) - start >= 0.75 - 1e-9:
            break
        env._tick(np.zeros(env.profile.action_size), score=False)
    return {"confirmed": False, "elapsed_sim_s": float(env.data.time) - start, "trace": trace}


def run_loop(output, general):
    results = {}
    for enabled in (False, True):
        name = "current_com" if enabled else "cached_com"
        suite = loop_suite(general, enabled)
        observer = NativeJerkObserver(record_physics=True)
        env = MasteryEnv(suite, stages=("base", "arm"), split="all")
        rows = []
        try:
            for case in suite["cases"]:
                obs, info = env.reset(options={"case_id": case["id"]})
                row = {
                    "id": case["id"],
                    "admission": copy.deepcopy(env.mastery_admission),
                    "admitted": not info["initialization_failed"],
                }
                rows.append(row)
                if row["admitted"]:
                    if env.control.base_velocity_feedback_current is not enabled:
                        raise RuntimeError("feedback option did not survive actual reset")
                    observer.reset(env, case)
                    if case["probe_push"]:
                        env._scheduled_push = True
                        env.disturbance_start, env.disturbance_steps = 10, 6
                        env.disturbance_wrench = np.array([0.5, 0, 0, 0, 0, 0])
                    for tick in range(round(env.config.horizon / env.profile.dt) + 1):
                        action = task_pose_teacher_action(env.control)
                        observer.before_step(env, action)
                        obs, _, done, truncated, info = env.step(action)
                        observer.after_step(env, action, info, terminated=done, truncated=truncated)
                        if done or truncated or tick + 1 == case.get("cancel_tick"):
                            break
                    else:
                        raise RuntimeError("bounded loop probe exceeded horizon")
                    row.update(
                        code=info["code"],
                        intentional_cancel="cancel_tick" in case,
                        metrics=mastery_metrics(env, info),
                        tracking_success=bool(info["is_success"]),
                        feedback_current=env.control.base_velocity_feedback_current,
                    )
                    row["stop"] = measured_stop(env)
                    row["safety"] = {
                        "wheel_lift": bool(env.monitor.wheel_lift),
                        "invalid_physics": bool(env._physics_invalid),
                        "peak_force_N": float(env.monitor.peak_force),
                        "unsafe_contact": bool(env.monitor.peak_force > env.profile.unsafe_force),
                    }
                print("LOOP", name, row["id"], row.get("code"), row.get("stop", {}).get("confirmed"), flush=True)
        finally:
            save_report(
                output / (name + "-physics.json.gz"), {"cases": rows, "measurements": list(observer.cases.values())}
            )
            env.close()
        results[name] = {
            "suite": suite,
            "suite_sha256": digest(suite),
            "cases": rows,
            "channels": observer.summarize(),
            "archive_sha256": sha(output / (name + "-physics.json.gz")),
        }
    current = results["current_com"]["cases"]
    ready = len(current) == 12 and all(
        r["admitted"]
        and r["stop"]["confirmed"]
        and not any(r["safety"][k] for k in ("wheel_lift", "invalid_physics", "unsafe_contact"))
        for r in current
    )
    nominal = [r for r in current if r["id"].endswith(":nominal")]
    ready &= len(nominal) == 5 and all(r.get("tracking_success") for r in nominal)
    return {"variants": results, "ready_for_broad_validation": ready}


class ProgressObserver:
    """Save completed cases after scoring; no simulator or policy access."""

    def __init__(self, path):
        self.path = path

    def reset(self, *args):
        pass

    def before_step(self, *args):
        pass

    def after_step(self, *args, **kwargs):
        pass

    def case_complete(self, report, row):
        if len(report["cases"]) % 10 == 0:
            temporary = self.path.with_suffix(".partial")
            save_report(temporary, report)
            temporary.replace(self.path)


class PrimitiveBundle:
    """Adapt the existing sensor-only bundle to the tensor rollout interface."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.fingerprint = bundle.manifest["fingerprint"]
        if bundle.robot != "innate_mars" or bundle.action_mode != "direct_velocity":
            raise ValueError("direct Mars bundle required")

    def __call__(self, observation):
        keys = (
            ("history", "preview", "motion")
            if self.bundle.manifest["format"] == "emet-whole-body-v2"
            else tuple(self.bundle.normalization)
        )
        sensor = {key: observation[key][0].detach().cpu().numpy().copy() for key in keys}
        return torch.from_numpy(np.asarray(self.bundle(sensor), dtype=np.float32).copy())[None]


def archived_scores(report, anchor, suite):
    """Comparison only: old budgets do not certify new reset identities."""
    previous = {row["id"]: row for row in anchor["cases"]}
    rows = []
    for case in report["cases"]:
        limits = smoothness_budgets(suite, previous[case["id"]]["metrics"])
        failed = [key for key, limit in limits.items() if case["metrics"][key] > limit]
        rows.append(
            {
                "id": case["id"],
                "smoothness_failures": failed,
                "tracking_and_settling_pass": case["tracking_and_settling_pass"],
                "comparison_pass": bool(
                    case["admitted"]
                    and not failed
                    and tracking_contract_passes(
                        case["final_info"],
                        case["metrics"],
                        suite["gate"],
                        path_failed=case["final_info"]["mastery"]["path_failed"],
                    )
                ),
            }
        )
    return rows


def run_baseline(output, development, general, loop_report):
    loop = json.loads(loop_report.read_text())
    if (
        not loop["complete"]
        or loop["phase"] != "loop"
        or not loop["ready_for_broad_validation"]
        or loop["source_hashes"] != source_hashes()
        or loop["script_sha256"] != sha(__file__)
        or set(loop["variants"]) != {"cached_com", "current_com"}
        or any(
            loop["variants"][name]["suite_sha256"] != digest(loop_suite(general, enabled))
            for name, enabled in (("cached_com", False), ("current_com", True))
        )
    ):
        raise ValueError("passing current-source reset/motion/stop probe required")
    current = copy.deepcopy(general)
    current["config"]["wheel_feedback_current_velocity"] = True
    teacher = evaluate_suite(
        current,
        stages=("base", "arm", "coupled"),
        population="admission",
        observer=ProgressObserver(output / "teacher-progress.json.gz"),
    )
    save_report(output / "general-teacher.json.gz", teacher)
    cases = development["train_cases"] + development["audit_cases"]
    tamp = primitive_rollout(cases, wheel_feedback_current_velocity=True)
    result = {
        "loop_report_sha256": sha(loop_report),
        "teacher": {
            "general_archive_sha256": sha(output / "general-teacher.json.gz"),
            "cases": len(teacher["cases"]),
            "full": sum(r["mastered"] for r in teacher["cases"]),
            "tamp": tamp,
            "ready_for_experimental_training": all(r["mastered"] for r in teacher["cases"])
            and tamp_training_readiness(tamp, development),
        },
        "baselines": {},
    }
    original = read_report(REFERENCE / "preflight/general-teacher.json.gz")
    for name, path in (
        ("retained_legacy", LEGACY),
        ("gru_control", REFERENCE / "control/last/actor"),
        ("gru_candidate", REFERENCE / "candidate/last/actor"),
    ):
        policy = load_control_policy(path)
        if name == "retained_legacy":
            evaluation, population = current, "admission"
        else:
            _, _, evaluation = load_basic_learning(
                Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json"), general
            )
            evaluation["config"]["wheel_feedback_current_velocity"] = True
            population = "canonical"
        report = evaluate_suite(
            evaluation,
            stages=("base", "arm", "coupled"),
            policy=policy,
            teacher_report=teacher,
            population=population,
            observer=ProgressObserver(output / (name + "-progress.json.gz")),
        )
        archive = output / (name + ".json.gz")
        save_report(archive, report)
        record = {
            "actor_weights_sha256": policy.manifest["weights_sha256"],
            "bundle_environment_option": policy.manifest["environment"].get("wheel_feedback_current_velocity", False),
            "evaluation_option_from_reset": True,
            "archive_sha256": sha(archive),
            "cases": len(report["cases"]),
            "fresh_budget_full": sum(r["mastered"] for r in report["cases"]),
            "outcomes": [
                {
                    k: row[k]
                    for k in (
                        "id",
                        "code",
                        "mastered",
                        "tracking_and_settling_pass",
                        "smoothness_pass",
                        "failure_reasons",
                        "metrics",
                    )
                }
                for row in report["cases"]
            ],
            "archived_budget_anchor_sha256": sha(REFERENCE / "preflight/general-teacher.json.gz"),
            "archived_budget_comparison": archived_scores(report, original, evaluation),
        }
        if name == "retained_legacy":
            record["tamp"] = primitive_rollout(
                cases, policy=PrimitiveBundle(policy), wheel_feedback_current_velocity=True
            )
            for before, after in zip(tamp, record["tamp"], strict=True):
                if before.get("initial_state_sha256") and after.get("initial_state_sha256"):
                    if before["initial_state_sha256"] != after["initial_state_sha256"]:
                        raise ValueError("teacher/legacy TAMP reset identities differ")
        result["baselines"][name] = record
        write(output / "progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("loop", "baseline"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loop-report", type=Path)
    args = parser.parse_args()
    if args.phase == "baseline" and args.loop_report is None:
        parser.error("baseline phase requires --loop-report")
    torch.set_num_threads(1)
    development, general = load_development()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = source_hashes()
    summary = {
        "format": "mars-current-feedback-validation-v1",
        "producer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": sha(__file__),
        "phase": args.phase,
        "source_hashes": sources,
        "general_suite_file_sha256": sha(development["general_suite"]),
        "hardware_used": False,
        "new_ppo_steps": 0,
        "qualified": False,
        "defaults_changed": False,
    }
    try:
        summary.update(
            run_loop(args.output, general)
            if args.phase == "loop"
            else run_baseline(args.output, development, general, args.loop_report)
        )
        if sources != source_hashes():
            raise RuntimeError("validation sources changed")
        summary["complete"] = True
    except BaseException as exc:
        summary.update(complete=False, error=repr(exc))
        raise
    finally:
        summary["elapsed_wall_seconds"] = time.monotonic() - started
        write(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
