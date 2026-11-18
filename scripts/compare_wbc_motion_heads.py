"""Controlled motion-head fitting and closed-loop tests; not PPO or promotion."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.arm_spline_benchmark import episode, save, stop
from emet.rl.manipulation.arm_splines import ArmSplineTrackingEnv, SplineLevel, load_manifest
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import TaskPoseTeacherConfig, task_pose_teacher_action
from emet.rl.manipulation.head_comparison import fit_head, label_nearby_goals, matched_heads, recovery_reference
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.regression_gate import file_digest


def infer(actor, observations):
    tensors = {k: torch.from_numpy(np.asarray(v, np.float32).copy())[None] for k, v in observations.items()}
    with torch.no_grad():
        return actor(tensors)[0].numpy().copy()


def run(
    output,
    initializer,
    *,
    updates=1000,
    count=4,
    backbone_trainable=False,
    recovery_offsets=False,
    coupled_teacher=False,
    heads=("gain", "direct"),
    goal_queries=0,
):
    output = Path(output)
    if output.exists() or type(count) is not int or not 1 <= count <= 16 or type(updates) is not int or updates <= 0:
        raise ValueError("fresh output and one to sixteen cases per split required")
    if not heads or len(set(heads)) != len(heads) or not set(heads) <= {"gain", "direct"}:
        raise ValueError("explicit unique diagnostic heads required")
    if type(goal_queries) is not int or not 0 <= goal_queries <= 4:
        raise ValueError("bounded virtual goal query count required")
    teacher_config = TaskPoseTeacherConfig(coupled_base_tracking=coupled_teacher)
    output.mkdir(parents=True)
    spec = load_manifest("configs/benchmarks/mars_arm_splines_v2.json")
    cases = [
        c
        for split in ("train", "audit")
        for c in [c for c in spec["cases"] if c["split"] == split and c["level"] == 0][:count]
    ]
    env = ArmSplineTrackingEnv(
        cases,
        levels=tuple(SplineLevel(**x) for x in spec["levels"]),
        timing=spec["reference_timing"],
        finish_after_hold=True,
        spread_requirements=spec["spread_requirements"],
    )
    report = {
        "format": "motion-head-fitting-diagnostic-v1",
        "sources": source_hashes(),
        "runner_sha256": file_digest(__file__),
        "manifest_sha256": digest(spec),
        "configuration": {
            "updates_per_head": updates,
            "backbone_frozen": not backbone_trainable,
            "seed": 20261118,
            "learning_rate": 1e-3 if backbone_trainable else 3e-3,
            "recovery_offsets": recovery_offsets,
            "teacher_configuration": asdict(teacher_config),
            "heads": list(heads),
            "goal_queries_every_five_ticks": goal_queries,
        },
        "scope": "teacher-action fit with explicit backbone mode, followed by unchanged closed-loop development cases; not PPO",
        "hardware_used": False,
        "safety_promoted": False,
        "complete": False,
        "teacher_cases": [],
        "heads": {},
        "virtual_goal_labels": {"train": 0, "audit": 0, "static_rejections": []},
    }
    data = {split: {k: [] for k in ("history", "preview", "motion", "actions")} for split in ("train", "audit")}
    try:
        source = WholeBodyBundle(initializer, robot="innate_mars", fingerprint=env.unwrapped.fingerprint)
        report["initializer_weights_sha256"] = source.manifest["weights_sha256"]
        for case, variant in [
            (case, variant)
            for case in cases
            for variant in (("nominal", "recovery") if recovery_offsets else ("nominal",))
        ]:
            observation, _ = env.reset(options={"case_id": case["id"]})
            acquisition = None
            if variant == "recovery":
                try:
                    observation, acquisition = recovery_reference(env, seed=case["seed"] + 900_000_000)
                except ValueError as exc:
                    report["teacher_cases"].append(
                        {
                            "case": case,
                            "variant": variant,
                            "success": False,
                            "code": str(exc),
                            "stop_confirmed": stop(env),
                        }
                    )
                    save(output / "report.json", report)
                    raise
            samples = data[case["split"]]
            e = env.unwrapped
            for _ in range(int(np.ceil(e.config.horizon / e.profile.dt)) + 1):
                if goal_queries and e.steps % 5 == 0:
                    labels, rejections = label_nearby_goals(
                        env,
                        seed=case["seed"] + e.steps * 1000003,
                        count=goal_queries,
                    )
                    report["virtual_goal_labels"][case["split"]] += len(labels)
                    report["virtual_goal_labels"]["static_rejections"].extend(
                        {"case": case["id"], "tick": e.steps, **r} for r in rejections
                    )
                    for label in labels:
                        for name in ("history", "preview", "motion"):
                            samples[name].append(label["observation"][name])
                        samples["actions"].append(label["action"])
                action = task_pose_teacher_action(e.control, config=teacher_config)
                for name in ("history", "preview", "motion"):
                    samples[name].append(observation[name].copy())
                samples["actions"].append(action.copy())
                observation, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            report["teacher_cases"].append(
                {
                    "case": case,
                    "variant": variant,
                    "acquisition": acquisition,
                    "success": info["primitive_success"],
                    "code": info["code"],
                    "stop_confirmed": stop(env),
                    "metrics": {
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
                }
            )
            save(output / "report.json", report)
            print("TEACHER", case["id"], variant, info["primitive_success"], flush=True)
        # Audit labels never select training data or change the fitting budget.
        # Retain failed audit expert trials explicitly; they are not a teacher
        # qualification claim and never enter optimizer minibatches.
        report["audit_teacher_failures"] = [
            row
            for row in report["teacher_cases"]
            if row["case"]["split"] == "audit" and (not row["success"] or not row["stop_confirmed"])
        ]
        if any(
            not row["success"] or not row["stop_confirmed"]
            for row in report["teacher_cases"]
            if row["case"]["split"] == "train"
        ):
            raise ValueError("diagnostic teacher population failed; preserve all rows")
        data = {split: {k: np.asarray(v, np.float32) for k, v in samples.items()} for split, samples in data.items()}
        np.savez_compressed(
            output / "teacher-samples.npz",
            **{f"{split}_{k}": v for split, samples in data.items() for k, v in samples.items()},
        )
        torch.set_num_threads(1)
        for name, actor in matched_heads(source.actor, backbone_frozen=not backbone_trainable).items():
            if name not in heads:
                continue
            before = {k: v.clone() for k, v in actor.state_dict().items()}
            fit = fit_head(
                actor,
                {k: data["train"][k] for k in ("history", "preview", "motion")},
                data["train"]["actions"],
                updates=updates,
                learning_rate=report["configuration"]["learning_rate"],
            )
            for key, value in actor.state_dict().items():
                if key.startswith("grip.") or (
                    not backbone_trainable and not key.startswith(("gain.", "motion_head."))
                ):
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            result = {
                "trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
                "fit": fit,
                "action_mse": {},
                "cases": [],
            }
            for split, samples in data.items():
                total, active = 0.0, 0.0
                for start in range(0, len(samples["actions"]), 128):
                    batch = {
                        k: torch.from_numpy(samples[k][start : start + 128]) for k in ("history", "preview", "motion")
                    }
                    with torch.no_grad():
                        mask = actor.action_mask(batch)[:, :-1]
                        error = actor(batch)[:, :-1] - torch.from_numpy(samples["actions"][start : start + 128, :-1])
                        total += float((error.square() * mask).sum())
                        active += float(mask.sum())
                result["action_mse"][split] = total / max(active, 1)
            report["heads"][name] = result
            torch.save(
                {"format": "experimental-motion-fitting-v1", "head": name, "state_dict": actor.state_dict()},
                output / (name + ".pt"),
            )
            for case in cases:
                row = episode(env, case, policy=lambda obs: infer(actor, obs))
                result["cases"].append(row)
                save(output / "report.json", report)
                print(name, case["id"], row["success"], row["code"], flush=True)
        if source_hashes() != report["sources"] or file_digest(__file__) != report["runner_sha256"]:
            raise RuntimeError("source changed during head comparison")
        report["complete"] = True
        save(output / "report.json", report)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initializer", type=Path, required=True)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--backbone-trainable", action="store_true")
    p.add_argument("--recovery-offsets", action="store_true")
    p.add_argument("--coupled-teacher", action="store_true")
    p.add_argument("--head", action="append", choices=("gain", "direct"))
    p.add_argument("--goal-queries", type=int, default=0)
    a = p.parse_args()
    run(
        a.output,
        a.initializer,
        updates=a.updates,
        count=a.count,
        backbone_trainable=a.backbone_trainable,
        recovery_offsets=a.recovery_offsets,
        coupled_teacher=a.coupled_teacher,
        heads=tuple(a.head or ("gain", "direct")),
        goal_queries=a.goal_queries,
    )
