# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Four fixed full-task attempts, comparing precision and pregrasp arrival.

The arrival study affects endpoint acceptance only. The separate grasp-region
study changes grasp admission/retiming/arrival together. Collision, force,
wheel-support, retention, measured-stop and later-stage contracts remain active.
Raw traces stay in ignored logs. This never trains or touches hardware.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import mujoco
import numpy as np
from validate_mars_current_feedback import LEGACY, PrimitiveBundle, sha, write

from emet.agent.tools import get_tools
from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.controller.task.tamp.native_adapter import adapter_for
from emet.eval.mars_tamp_control import final_predicates, fixture, sources
from emet.rl.manipulation.command_runtime import TaskPoseRuntimeTiming, task_pose_teacher_action
from emet.rl.manipulation.mastery import save_report
from emet.rl.manipulation.policy import load_control_policy
from emet.rl.manipulation.tamp_policy import robot_identity
from emet.rl.manipulation.whole_body_observation import actor_observation
from emet.rl.manipulation.whole_body_training import load_development

GROUPS = ("teacher_precision", "actor_precision", "teacher_region", "actor_region")
ROBUST_GROUPS = ("teacher_maximum_margin", "actor_maximum_margin")
GRASP_GROUPS = ("teacher_grasp_region", "actor_grasp_region")
REGION = (0.004, 0.04)
WEIGHTS = "e7b425699510cc635485bdd8f59a5b25692fb165faf248aaecbc2707c0dbb883"


def execute_stage_contract(executor, original, phase, path, *, grasp_region):
    previous = (executor.position_tolerance_m, executor.orientation_tolerance_rad)
    if phase == "grasp" and grasp_region:
        executor.position_tolerance_m, executor.orientation_tolerance_rad = REGION
    try:
        return original(phase, path)
    finally:
        executor.position_tolerance_m, executor.orientation_tolerance_rad = previous


def finger_geometry(env):
    """Signed current-configuration distances, not an uncertainty certificate."""
    model = env.model
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, env.data)
    mujoco.mj_forward(model, scratch)
    bodies = np.asarray(model.geom_bodyid).copy()
    active = (np.asarray(model.geom_contype).copy() | np.asarray(model.geom_conaffinity).copy()) != 0
    fingers = {model.body(n).id for n in env.profile.finger_bodies}
    finger_geoms = np.flatnonzero(np.isin(bodies, list(fingers)) & active)
    distances = {}
    for label, target in (("source_support", model.body("table").id), ("payload", env.payload)):
        pairs = [
            float(mujoco.mj_geomDistance(model, scratch, int(a), int(b), 1.0, None))
            for a in finger_geoms
            for b in np.flatnonzero((bodies == target) & active)
        ]
        distances[label + "_min_distance_m"] = min(pairs) if pairs else None
    distances["jaw_angle_rad"] = float(env.data.qpos[env.control.grip_q])
    distances["scope"] = "measured_configuration_only; benign payload contact permitted"
    return distances


def episode(group, output):
    development, _ = load_development()
    case = next(c for c in development["train_cases"] if c["id"] == "development:right:50g:1")
    maximum_margin = group in ROBUST_GROUPS
    if maximum_margin:
        case = {**case, "id": case["id"] + "@maximum_margin"}
    env = robot = adapter = None
    row = {"group": group, "case": case, "success": False, "stage_results": [], "calls": []}
    raw = {}
    try:
        env, admission = fixture(
            case, wheel_feedback_current_velocity=True, grasp_setup_prefer_nominal=not maximum_margin
        )
        row["fixture_admission"] = admission
        if not admission["accepted"]:
            raise ValueError("fixture admission failed")
        row["reset_setup"] = env.randomization.get("grasp_setup")
        row["grasp_setup_prefer_nominal"] = not maximum_margin
        row["initial_state_sha256"] = hashlib.sha256(
            np.r_[env.data.qpos.copy(), env.data.qvel.copy(), float(env.data.time)].tobytes()
        ).hexdigest()
        bundle = load_control_policy(LEGACY)
        if bundle.manifest["weights_sha256"] != WEIGHTS:
            raise ValueError("frozen retained actor changed")
        compatibility = PrimitiveBundle(bundle)
        if compatibility.compiled_robot_identity != robot_identity(env.model):
            raise ValueError("compiled robot mismatch before motion")
        env.control.retention_context_enabled = True
        env.control.history = None
        env.control.base_velocity_feedback_gain = 0.5
        env.control.turn_progress_guard_enabled = True
        teacher = group.startswith("teacher")

        class Actor:
            is_reference_controller = teacher
            controller_name = "task_arrival_teacher" if teacher else "task_arrival_retained_actor"

            def __call__(self, observation):
                if teacher:
                    return task_pose_teacher_action(env.control)
                sensor = actor_observation(env.control, observation)
                return bundle({k: sensor[k] for k in bundle.normalization})

        robot = MarsSimRobot(
            env,
            policy=Actor(),
            timing=TaskPoseRuntimeTiming(60.0, 61.0),
            record_observations=True,
            arm_execution="cartesian",
            arm_reference="s_curve",
        )
        context = {
            "robot": robot,
            "manip_mode": "physical",
            "tamp_inputs": "privileged",
            "_native_tamp_model": env.model,
        }
        tools = {t.name: t for t in get_tools(context)}

        def call(name, **kwargs):
            result = json.loads(tools[name].func(**kwargs))
            row["calls"].append({"tool": name, "result": result})
            return result

        discovered = call("scene_tasks", object_filter="blue")
        task = next(t for t in discovered["data"]["tasks"] if t["receptacle_name"] == "destination support")
        planned = call("plan_pick_place", task_ref=task["task_ref"], placement_region=admission["placement_region"])
        if planned["status"] != "ok":
            raise ValueError("public_plan:" + planned["code"])
        adapter = adapter_for(context, robot)
        plan = adapter.plans[planned["data"]["plan_ref"]]
        executor = plan.executor
        executor.pregrasp_arrival_tolerance = REGION if group.endswith("region") or maximum_margin else None
        row["arrival_region"] = executor.pregrasp_arrival_tolerance
        row["other_stage_tolerances"] = [executor.position_tolerance_m, executor.orientation_tolerance_rad]
        row["grasp_stage_region"] = REGION if group in GRASP_GROUPS else None
        row["pregrasp_to_grasp_offset_m"] = (
            np.asarray(executor.grasp_targets[0][0]) - executor.grasp_targets[1][0]
        ).tolist()
        row["initial_finger_geometry"] = finger_geometry(env)
        original = executor._execute_path

        def record_stage(phase, path):
            start = float(env.data.time)
            # This separate task-contract experiment changes grasp admission,
            # retiming and arrival together. It is NOT the endpoint-only study.
            # Fresh planning/configuration and all independent runtime guards
            # remain active; later stages restore the strict contract.
            result = execute_stage_contract(executor, original, phase, path, grasp_region=group in GRASP_GROUPS)
            row["stage_results"].append(
                {
                    "phase": phase,
                    "success": bool(result.success),
                    "reason": result.message,
                    "start_s": start,
                    "end_s": float(env.data.time),
                    "residual": result.residual,
                    "finger_geometry": finger_geometry(env),
                }
            )
            return result

        executor._execute_path = record_stage
        with robot.stepped_clock():
            executed = call("execute_pick_place_plan", plan_ref=planned["data"]["plan_ref"])
        predicates = final_predicates(robot, admission)
        stable = executed["status"] == "ok" and all(predicates.values())
        if stable:
            for _ in range(40):
                robot.step(env.profile.dt)
                predicates = final_predicates(robot, admission)
                stable &= all(predicates.values())
        row.update(
            success=bool(stable),
            final_predicates=predicates,
            post_execution_stable_s=2.0 if stable else 0.0,
            executor_result=executed,
        )
    except Exception as exc:
        row.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        if robot is not None:
            row.update(
                robot_failure=robot.failure,
                inference_counts=robot.inference_counts,
                max_payload_translation_m=robot.max_translation,
                max_payload_rotation_rad=robot.max_rotation,
                monitor=env.monitor.metrics(),
            )
            robot.stop_and_confirm("task_arrival_diagnostic_complete")
            row["stop_confirmed"] = robot.stop_confirmed
            row["success"] &= robot.stop_confirmed is True
            raw.update(ticks=robot.trace, commands=robot.commands, events=adapter.events if adapter else [])
        else:
            row["stop_confirmed"] = None
        raw["row"] = row
        archive = output / (group + ".json.gz")
        save_report(archive, raw)
        row["archive_sha256"] = sha(archive)
        write(output / (group + ".json"), row)
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode", choices=GROUPS + ROBUST_GROUPS + GRASP_GROUPS)
    parser.add_argument("--study", choices=("arrival", "maximum_margin", "grasp_region"), default="arrival")
    args = parser.parse_args()
    if args.episode:
        episode(args.episode, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "mars-task-arrival-diagnostic-v1",
        "source_hashes": sources(),
        "producer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha(__file__),
        "weights_sha256": WEIGHTS,
        "groups": {},
        "new_ppo_steps": 0,
        "hardware_used": False,
        "qualified": False,
        "complete": False,
    }
    try:
        summary["study"] = args.study
        groups = {"arrival": GROUPS, "maximum_margin": ROBUST_GROUPS, "grasp_region": GRASP_GROUPS}[args.study]
        for group in groups:
            process = subprocess.run(
                [sys.executable, "-u", __file__, "--episode", group, "--output", str(args.output)], timeout=180
            )
            path = args.output / (group + ".json")
            if process.returncode or not path.exists():
                summary["groups"][group] = {
                    "success": False,
                    "worker_returncode": process.returncode,
                    "stop_confirmed": None,
                }
                raise RuntimeError("native worker failed: " + group)
            summary["groups"][group] = json.loads(path.read_text())
            write(args.output / "progress.json", summary)
            print(
                "TASK",
                group,
                summary["groups"][group]["success"],
                summary["groups"][group].get("robot_failure"),
                flush=True,
            )
        if sources() != summary["source_hashes"]:
            raise RuntimeError("source changed during diagnostic")
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        write(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
