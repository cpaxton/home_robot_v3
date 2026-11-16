# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compact the fixed task-arrival/reset studies without running more episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from validate_mars_current_feedback import sha, write

from emet.controller.task.tamp.mars_pose_executor import pose_error
from emet.eval.mars_tamp_control import fixture
from emet.eval.physical_tamp import make_scene_checker
from emet.rl.manipulation.grasp_setup import support_margin
from emet.rl.manipulation.mastery import read_report
from emet.rl.manipulation.whole_body_observation import pose_rotation


def physical_identity(raw):
    fields = (
        "time",
        "normalized_action",
        "physics_before",
        "physics_after",
        "hand",
        "base",
        "payload_rotation",
        "payload_height",
        "gripper_position",
    )
    rows = [{k: t.get(k) for k in fields} for t in raw["ticks"]]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def retention_curve(raw):
    close = [r for r in raw["ticks"] if r["phase"] == "close"]
    if not close:
        return []
    before = close[-1]
    initial = pose_rotation(before["hand"]).T @ np.asarray(before["payload_rotation"]).reshape(3, 3)
    rows = []
    for tick in raw["ticks"]:
        if tick["phase"] != "carry":
            continue
        relative = pose_rotation(tick["hand"]).T @ np.asarray(tick["payload_rotation"]).reshape(3, 3)
        angle = float(np.arccos(np.clip((np.trace(initial.T @ relative) - 1) / 2, -1, 1)))
        rows.append(
            {
                "elapsed_s": tick["time"] - before["time"],
                "rotation_rad": angle,
                "lift_m": tick["payload_height"] - before["payload_height"],
                "failure": tick["failure"],
            }
        )
        if tick["failure"]:
            break
    return rows


def nominal_lift_audit(raw):
    """Reconstruct the admitted vertices with the planner's attached payload."""
    env, admission = fixture(raw["row"]["case"], wheel_feedback_current_velocity=True)
    try:
        state = hashlib.sha256(
            np.r_[env.data.qpos.copy(), env.data.qvel.copy(), float(env.data.time)].tobytes()
        ).hexdigest()
        if not admission["accepted"] or state != raw["row"]["initial_state_sha256"]:
            raise ValueError("offline fixture differs from the recorded initial state")
        event = next(e for e in raw["events"] if e.get("phase") == "lift" and "task_pose_trajectory" in e)
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = event["task_pose_trajectory"]["fk_anchor_qpos"]
        mujoco.mj_forward(env.model, scratch)
        checker = make_scene_checker(
            env.model, {"object_body": env.payload_name, "gripper_bodies": env.profile.finger_bodies}
        )
        checker.set_payload(env.model, scratch, env.payload_name, env.profile.ee_body)
        margins = []
        for q in event["joint_witness"]:
            for group, value in zip(env.control.q, q, strict=True):
                scratch.qpos[group] = value / len(group)
            checker.configuration_collides(env.model, scratch)
            margins.append(support_margin(env.model, scratch, env.profile, env.payload)["margin_m"])
        return {
            "scope": "offline admitted witness vertices; private attached-payload kinematics, no physics steps or robust certificate",
            "model_fingerprint": env.fingerprint,
            "initial_state_sha256": state,
            "minimum_margin_m": min(margins),
            "runtime_cutoff_m": 0.005,
            "reserve_above_cutoff_m": min(margins) - 0.005,
            "vertices": len(margins),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival", required=True, type=Path)
    parser.add_argument("--maximum-margin", required=True, type=Path)
    parser.add_argument("--grasp-region", required=True, type=Path)
    parser.add_argument("--hold-reacquire", required=True, type=Path)
    parser.add_argument("--physical-stance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "format": "mars-task-arrival-readout-v1",
        "analysis_script_sha256": sha(__file__),
        "new_ppo_steps": 0,
        "hardware_used": False,
        "qualified": False,
        "studies": {},
    }
    raw = {}
    for name, directory in (
        ("arrival", args.arrival),
        ("maximum_margin", args.maximum_margin),
        ("grasp_region", args.grasp_region),
        ("hold_reacquire", args.hold_reacquire),
        ("physical_stance", args.physical_stance),
    ):
        summary = json.loads((directory / "summary.json").read_text())
        if not summary["complete"]:
            raise ValueError("complete bounded studies required")
        study = {k: v for k, v in summary.items() if k != "groups"}
        study.update(summary_sha256=sha(directory / "summary.json"), groups={})
        for group, row in summary["groups"].items():
            archive = directory / (group + ".json.gz")
            if sha(archive) != row["archive_sha256"]:
                raise ValueError("trace hash changed: " + group)
            trace = raw[name + "/" + group] = read_report(archive)
            compact = {k: v for k, v in row.items() if k not in {"calls", "traceback"}}
            compact["physical_trace_identity"] = physical_identity(trace)
            compact["hold_reacquisitions"] = [e for e in trace["events"] if e.get("reason") == "hold_reacquire"]
            if row.get("physical_preparation"):
                preparation = row["physical_preparation"]
                base = np.asarray(preparation["trajectory"]["base"])
                direction = (base[-1, :2] - base[0, :2]) / 0.02
                end = min(
                    (c["sim_time"] for c in trace["commands"] if c.get("phase") == "approach_clearance"),
                    default=float("inf"),
                )
                ticks = [t for t in trace["ticks"] if t["time"] <= end]
                compact["physical_preparation_tracking"] = {
                    "requested_forward_m": 0.02,
                    "achieved_forward_m": float(
                        (np.asarray(preparation["measured_base"])[:2] - base[0, :2]) @ direction
                    ),
                    "peak_abs_base_request_m_s": max(abs(t["commanded_base_velocity"][0]) for t in ticks),
                    "peak_abs_applied_base_velocity_m_s": max(abs(t["applied_base_velocity"][0]) for t in ticks),
                }
            compact["terminal_tracking_rejections"] = [
                e
                for e in trace["events"]
                if e.get("reason") in {"cartesian_tracking_timeout", "cartesian_path_tracking_envelope"}
            ]
            curve = retention_curve(trace)
            compact["first_loaded_failure"] = curve[-1] if curve else None
            study["groups"][group] = compact
        result["studies"][name] = study
    manifests = [s["source_hashes"] for s in result["studies"].values()]
    common = {k: v for k, v in manifests[0].items() if all(m.get(k) == v for m in manifests)}
    result["source_hashes_common"] = common
    for study in result["studies"].values():
        manifest = study.pop("source_hashes")
        study["source_hashes_delta"] = {k: v for k, v in manifest.items() if k not in common}
        if {**common, **study["source_hashes_delta"]} != manifest:
            raise ValueError("compact source manifest reconstruction failed")
        study["source_hashes_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    result["teacher_precision_region_physics_identical"] = physical_identity(
        raw["arrival/teacher_precision"]
    ) == physical_identity(raw["arrival/teacher_region"])
    result["teacher_grasp_before_after_hold_fix_physics_identical"] = physical_identity(
        raw["grasp_region/teacher_grasp_region"]
    ) == physical_identity(raw["hold_reacquire/teacher_grasp_region"])
    result["nominal_lift_support_audit"] = nominal_lift_audit(raw["arrival/teacher_precision"])
    plt.rcParams["svg.hashsalt"] = "mars-task-arrival-v1"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for group in (
        "arrival/teacher_precision",
        "arrival/actor_precision",
        "arrival/actor_region",
        "hold_reacquire/actor_grasp_region",
    ):
        ticks = [t for t in raw[group]["ticks"] if t["phase"] == "pregrasp" and not t["failure"]]
        times = [t["time"] - 1 for t in ticks]
        errors = np.asarray([pose_error(t["hand"], t["hand_goal"]) for t in ticks])
        label = group.replace("_", " ")
        axes[0, 0].plot(times, errors[:, 0] * 1000, label=label)
        axes[0, 1].plot(times, errors[:, 1], label=label)
    axes[0, 0].set_ylabel("Active hand position error (mm)")
    axes[0, 1].set_ylabel("Active hand orientation error (rad)")
    for axis in axes[0]:
        axis.set_xlabel("Seconds since execution start")
    for group in (
        "arrival/teacher_precision",
        "maximum_margin/teacher_maximum_margin",
        "physical_stance/teacher_physical_stance",
    ):
        curve = retention_curve(raw[group])
        axes[1, 0].plot(
            [r["elapsed_s"] for r in curve], [r["lift_m"] * 1000 for r in curve], label=group.replace("_", " ")
        )
        axes[1, 1].plot(
            [r["elapsed_s"] for r in curve], [r["rotation_rad"] for r in curve], label=group.replace("_", " ")
        )
    axes[1, 0].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0.1, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("Measured object lift (mm)")
    axes[1, 1].set_ylabel("Object rotation in hand (rad)")
    for axis in axes[1]:
        axis.set_xlabel("Seconds since loaded lift starts")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Mars task-level diagnosis: reaching, grasping and loaded lift")
    output = args.output / "task-stages.svg"
    fig.savefig(output, metadata={"Date": None})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)
    write(args.output / "results.json", result)


if __name__ == "__main__":
    main()
