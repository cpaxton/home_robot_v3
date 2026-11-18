"""Confirm an experimental fitted motion head without exporting or training it."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.arm_spline_benchmark import episode, save, stop
from emet.rl.manipulation.arm_splines import ANCHORS, V2_NAMESPACES, ArmSplineTrackingEnv, SplineLevel, load_manifest
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import TaskPoseTeacherConfig, task_pose_teacher_action
from emet.rl.manipulation.head_comparison import matched_heads
from emet.rl.manipulation.motion_distribution import make_reference_env, preflight_pools
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_training import primitive_rollout
from emet.rl.regression_gate import file_digest


def confirmation_cases(spec):
    """Fixed fresh seeds, balanced across anchors; never select successful cases."""
    cases = [
        {
            "id": f"mars:arm-splines-v2:audit:l{level}:confirmation:{index}",
            "seed": V2_NAMESPACES["audit"] + level * 10000 + 1000 + index,
            "split": "audit",
            "level": level,
            "posture": list(ANCHORS)[index % len(ANCHORS)],
        }
        for level, count in ((0, 16), (1, 8), (2, 8))
        for index in range(count)
    ]
    if {c["seed"] for c in cases} & {c["seed"] for c in spec["cases"]}:
        raise ValueError("confirmation seeds overlap the canonical population")
    return cases


def infer(actor, obs):
    with torch.no_grad():
        tensors = {
            k: torch.from_numpy(np.asarray(obs[k], np.float32).copy())[None] for k in ("history", "preview", "motion")
        }
        return actor(tensors)[0].numpy().copy()


def run(fit_path, initializer, output, *, head="gain"):
    fit_path, output = Path(fit_path), Path(output)
    fit = json.loads((fit_path / "report.json").read_text())
    sources = source_hashes()
    if output.exists() or fit.get("complete") is not True or fit.get("sources") != sources:
        raise ValueError("fresh output and completed source-matched experimental fit required")
    spec = load_manifest("configs/benchmarks/mars_arm_splines_v2.json")
    cases = confirmation_cases(spec)
    env = ArmSplineTrackingEnv(
        cases,
        levels=tuple(SplineLevel(**x) for x in spec["levels"]),
        timing=spec["reference_timing"],
        finish_after_hold=True,
        spread_requirements=spec["spread_requirements"],
    )
    output.mkdir(parents=True)
    report = {
        "format": "experimental-motion-head-confirmation-v1",
        "complete": False,
        "hardware_used": False,
        "safety_promoted": False,
        "scope": "fresh spline seeds plus reused general/TAMP regression pools; no fitting or case pruning",
        "fit_report_sha256": file_digest(fit_path / "report.json"),
        "runner_sha256": file_digest(__file__),
        "sources": sources,
        "fingerprint": env.unwrapped.fingerprint,
        "cases": cases,
        "splines": {"teacher": [], head: []},
        "families": {},
    }
    try:
        source = WholeBodyBundle(initializer, robot="innate_mars", fingerprint=env.unwrapped.fingerprint)
        if source.manifest["weights_sha256"] != fit["initializer_weights_sha256"]:
            raise ValueError("matching experimental backbone initializer required")
        actor = matched_heads(source.actor, backbone_frozen=fit["configuration"]["backbone_frozen"])[head]
        weights = fit_path / (head + ".pt")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if state.get("format") != "experimental-motion-fitting-v1" or state.get("head") != head:
            raise ValueError("matching experimental motion weights required")
        actor.load_state_dict(state["state_dict"], strict=True)
        actor.eval()
        report["weights_sha256"] = file_digest(weights)
        teacher = TaskPoseTeacherConfig(**fit["configuration"]["teacher_configuration"])
        for name, policy in (
            ("teacher", lambda _: task_pose_teacher_action(env.unwrapped.control, config=teacher)),
            (head, lambda obs: infer(actor, obs)),
        ):
            for case in cases:
                try:
                    row = episode(env, case, policy=policy)
                except Exception as exc:
                    row = {
                        "case": case,
                        "success": False,
                        "code": str(exc),
                        "error_type": type(exc).__name__,
                        "stop_confirmed": stop(env),
                    }
                report["splines"][name].append(row)
                save(output / "report.json", report)
                print(name, case["id"], row["success"], row["code"], flush=True)
    finally:
        env.close()
    for family, population in preflight_pools().items():
        if family == "tamp":
            report["families"][family] = primitive_rollout(
                population,
                policy=actor,
                primitive="grasp_pose",
                planning_scope="unloaded_prefix",
                wheel_feedback_current_velocity=True,
                wheel_feedback_base_only_gains=(0.0, 0.5),
            )
        else:
            env = make_reference_env(
                family, population, spec["spread_requirements"], settling_budget_s=spec["general_settling_budget_s"]
            )
            report["families"][family] = []
            try:
                for case in population:
                    row = {"case": case, "success": False}
                    try:
                        obs, initial = env.reset(options={"case_id": case["id"]})
                        row["reference_sha256"] = initial["reference_sha256"]
                        e = env.unwrapped
                        row["initial_state_sha256"] = digest(
                            {"qpos": e.data.qpos.copy().tolist(), "qvel": e.data.qvel.copy().tolist()}
                        )
                        for _ in range(int(np.ceil(e.config.horizon / e.profile.dt)) + 1):
                            obs, _, done, truncated, info = env.step(infer(actor, obs))
                            if done or truncated:
                                break
                        else:
                            raise RuntimeError("bounded family rollout did not terminate")
                        row.update(
                            success=bool(info["primitive_success"]),
                            code=info["code"],
                            metrics={
                                k: info[k]
                                for k in (
                                    "ee_position_error",
                                    "ee_orientation_error",
                                    "unsafe_contact",
                                    "wheel_lift",
                                    "invalid_physics",
                                    "settled_seconds",
                                    "peak_contact_force",
                                )
                            },
                        )
                    except Exception as exc:
                        row.update(code=str(exc), error_type=type(exc).__name__)
                    row["stop_confirmed"] = stop(env)
                    report["families"][family].append(row)
                    save(output / "report.json", report)
                    print(family, case["id"], row["success"], row["code"], flush=True)
            finally:
                env.close()
        save(output / "report.json", report)
    if sources != source_hashes() or report["weights_sha256"] != file_digest(weights):
        raise RuntimeError("experimental sources or weights changed during confirmation")
    report["complete"] = True
    save(output / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--initializer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--head", choices=("gain", "direct"), default="gain")
    a = p.parse_args()
    torch.set_num_threads(1)
    run(a.fit, a.initializer, a.output, head=a.head)
