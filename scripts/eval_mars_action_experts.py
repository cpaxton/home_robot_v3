"""Evaluate a simulator-only Mars expert portfolio on the frozen held-out mix."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from emet.rl.manipulation.arm_spline_benchmark import save, stop
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv
from emet.rl.manipulation.expert_routing import FAMILIES, bind_expert, tracking_family
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.regression_gate import file_digest

ARM_SPEED_LIMIT = 0.25


def episode(case, experts):
    env = CoupledTrackingEnv("innate_mars", [case], suite_path=COARSE_SUITE)
    plant = env.unwrapped
    plant.config = replace(plant.config, wheel_feedback_base_only_gains=(0.0, 0.5))
    family = tracking_family(case["mode"])
    try:
        bundle = experts.get(family)
        if bundle is not None and bundle.manifest["fingerprint"] != plant.fingerprint:
            raise ValueError(f"Expert model mismatch for {family}")
        observation, info = env.reset(options={"case_id": case["id"]})
        initial = digest({"qpos": plant.data.qpos.tolist(), "qvel": plant.data.qvel.tolist()})
        reference = info["reference_sha256"]
        policy = bind_expert(case["mode"], plant.control, experts, arm_speed_limit=ARM_SPEED_LIMIT)
        peak_hand = np.zeros(2)
        peak_drift = peak_speed = 0.0
        for _ in range(int(np.ceil(plant.config.horizon / plant.profile.dt)) + 1):
            act = policy({key: observation[key] for key in ("history", "preview", "motion")})
            observation, _, done, truncated, info = env.step(act)
            peak_hand = np.maximum(peak_hand, [info["ee_position_error"], info["ee_orientation_error"]])
            peak_drift = max(peak_drift, float(info["base_goal_error"]))
            peak_speed = max(peak_speed, float(np.max(np.abs(plant.control.velocities()))))
            if done or truncated:
                break
        confirmed = stop(env)
        post = plant._info()
        flags = {
            name: bool(info.get(name) or post.get(name))
            for name in (
                "wheel_lift",
                "unsafe_contact",
                "retention_lost",
                "invalid_physics",
            )
        }
        path_passed = family == "base_only" or bool(np.all(peak_hand <= [0.003, 0.02]))
        arm_envelope_passed = family != "arm_only" or (peak_drift <= 0.01 and peak_speed <= ARM_SPEED_LIMIT)
        return {
            "case_id": case["id"],
            "family": family,
            "initial_sha256": initial,
            "reference_sha256": reference,
            "native_success": bool((done or truncated) and info.get("primitive_success")),
            "passed": bool(
                (done or truncated)
                and info.get("primitive_success")
                and path_passed
                and arm_envelope_passed
                and confirmed
                and not any(flags.values())
            ),
            "code": info.get("code"),
            "stop_confirmed": confirmed,
            "safety_flags": flags,
            "peak_hand_position_m": float(peak_hand[0]),
            "peak_hand_orientation_rad": float(peak_hand[1]),
            "peak_base_error_m": peak_drift,
            "peak_measured_arm_speed_rad_s": peak_speed,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--expert-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qualification", type=Path, help="execute a previously audited simulator route")
    parser.add_argument("--case-id", help="one audited command to execute with --qualification")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required")
    if bool(args.qualification) != bool(args.case_id):
        parser.error("--qualification and --case-id must be supplied together")
    sources = source_hashes()
    cases = [case for case in json.loads(args.cases.read_text()) if case["split"] == "audit"]
    if {tracking_family(case["mode"]) for case in cases} != FAMILIES:
        raise ValueError("All three families require independent audit cases")
    experts = {family: WholeBodyBundle(args.expert_root / family / "actor", robot="innate_mars") for family in FAMILIES}
    if args.qualification:
        qualification = json.loads(args.qualification.read_text())
        expected_weights = {name: bundle.manifest["weights_sha256"] for name, bundle in experts.items()}
        if (
            qualification.get("complete") is not True
            or qualification.get("sources") != sources
            or qualification.get("cases_sha256") != file_digest(args.cases)
            or qualification.get("expert_weights_sha256") != expected_weights
        ):
            raise ValueError("Route qualification does not match source, cases, or actors")
        case = next((case for case in cases if case["id"] == args.case_id), None)
        if case is None:
            raise ValueError("Only a declared audit command can be replayed")
        family = tracking_family(case["mode"])
        selected = qualification["simulator_selection"][family]
        if selected not in {"learned", "teacher"}:
            raise ValueError(f"No qualified simulator route for {family}")
        audited = next(
            row["expert" if selected == "learned" else "teacher"]
            for row in qualification["results"]
            if row["expert"]["case_id"] == case["id"]
        )
        if not audited["passed"]:
            raise ValueError("Requested command did not pass with the selected route")
        result = episode(case, experts if selected == "learned" else {})
        if sources != source_hashes() or any(
            result[key] != audited[key] for key in ("initial_sha256", "reference_sha256")
        ):
            raise RuntimeError("Selected replay did not match the qualification")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, {"route": selected, "hardware_used": False, **result})
        print(family, selected, result["passed"], flush=True)
        if not result["passed"]:
            raise SystemExit("Selected route failed on replay")
        return
    report = {
        "complete": False,
        "hardware_used": False,
        "safety_promoted": False,
        "sources": sources,
        "runner_sha256": file_digest(__file__),
        "cases_sha256": file_digest(args.cases),
        "expert_weights_sha256": {name: bundle.manifest["weights_sha256"] for name, bundle in experts.items()},
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        teacher = episode(case, {})
        expert = episode(case, experts)
        for key in ("initial_sha256", "reference_sha256"):
            if teacher[key] != expert[key]:
                raise RuntimeError(f"Unpaired {key} for {case['id']}")
        report["results"].append({"teacher": teacher, "expert": expert})
        save(args.output, report)
        print(case["id"], "teacher", teacher["passed"], "expert", expert["passed"], flush=True)
    if source_hashes() != sources:
        raise RuntimeError("Sources changed while evaluating")
    report["simulator_selection"] = {}
    for family in sorted(FAMILIES):
        rows = [row for row in report["results"] if row["expert"]["family"] == family]
        report["simulator_selection"][family] = (
            "learned"
            if all(row["expert"]["passed"] for row in rows)
            else "teacher"
            if all(row["teacher"]["passed"] for row in rows)
            else "unsupported"
        )
    report["complete"] = True
    save(args.output, report)
    print(report["simulator_selection"], flush=True)


if __name__ == "__main__":
    main()
