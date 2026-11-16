#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Native Panda Lift/Stack references for the MuJoCo realism progression.

Privileged feedback, scripted reference and hold controls; no TAMP/agent score.
Only the upstream reset initializes physics. All subsequent motion and settling
uses native env.step through EpisodeRuntime. No renderer or VLM is needed.
Launch through emet jobs with MUJOCO_GL=disable and CUDA_VISIBLE_DEVICES=.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from emet.controller.task.tamp.task_contracts import EpisodeSpec
from emet.eval.tamp_benchmark.adapters import RobosuiteAdapter
from emet.eval.tamp_benchmark.runtime import EpisodeRuntime


def run_case(output, *, task, seed, policy, max_steps):
    output.mkdir(parents=True, exist_ok=False)
    controller = load_composite_controller_config(robot="Panda")
    arm = controller["body_parts"]["right"]
    if arm["type"] != "OSC_POSE" or arm["input_type"] != "delta":
        raise ValueError("unsupported_reference_controller")
    arm["input_ref_frame"] = "world"
    (output / "controller.json").write_text(json.dumps(controller, indent=2) + "\n")
    env = robosuite.make(
        task,
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        seed=seed,
    )
    episode = EpisodeSpec(
        f"development-panda-{task}-{seed}-{policy}-cap{max_steps}",
        "robosuite",
        task,
        f"development-seeded-reset-{seed}",
        seed,
        "Lift the cube" if task == "Lift" else "Stack the red cube on the green cube",
        env.horizon,
    )
    adapter = RobosuiteAdapter(env, suite="robosuite", install_reset=lambda e, _: e.reset(), success=env._check_success)
    phases, held, placed, reason = [], [], [], None
    with EpisodeRuntime(adapter, episode) as runtime, (output / "trace.jsonl").open("x") as trace:
        if env.action_dim != 7:
            raise ValueError("unsupported_reference_action_layout")
        eef_id = env.robots[0].eef_site_id["right"]
        target = env.cube if task == "Lift" else env.cubeA
        target_id = env.cube_body_id if task == "Lift" else env.cubeA_body_id
        initial = np.asarray(env.sim.get_state().flatten()).copy()
        initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
        np.save(output / "initial_state.npy", initial)
        initial_ctrl = env.sim.data.ctrl.copy()
        np.save(output / "initial_ctrl.npy", initial_ctrl)
        mujoco.mj_saveModel(env.sim.model._model, str(output / "model.mjb"))
        xml = env.model.get_xml()
        (output / "model.xml").write_text(xml)
        task_source = Path(inspect.getfile(type(env))).read_bytes()
        (output / "upstream_task_source.py").write_bytes(task_source)
        initial_cube = env.sim.data.body_xpos[target_id].copy()

        def sample(phase, action):
            cube = env.sim.data.body_xpos[target_id].copy()
            eef = env.sim.data.site_xpos[eef_id].copy()
            grasp = bool(env._check_grasp(env.robots[0].gripper["right"], target.contact_geoms))
            row = {
                "step": runtime.control_steps,
                "phase": phase,
                "action": action.tolist(),
                "sim_time": float(env.sim.data.time),
                "cube": cube.tolist(),
                "eef": eef.tolist(),
                "bilateral_grasp_contact": grasp,
                "relative_position": (cube - eef).tolist(),
            }
            if not np.isfinite(np.r_[cube, eef, env.sim.data.qpos, env.sim.data.qvel]).all():
                raise RuntimeError("nonfinite_physics_state")
            if task == "Stack":
                row["support"] = env.sim.data.body_xpos[env.cubeB_body_id].tolist()
                row["support_contact"] = bool(env.check_contact(target, env.cubeB))
                if phase == "hold_place":
                    placed.append(row)
            trace.write(json.dumps(row, allow_nan=False) + "\n")
            if phase == "hold_lift":
                held.append(row)

        def step(phase, point=None, gripper=-1.0):
            if runtime.stopped or runtime.control_steps >= max_steps:
                return False
            action = np.zeros(env.action_dim)
            if point is not None:
                error = point - env.sim.data.site_xpos[eef_id]
                action[:3] = np.clip(error / np.asarray(arm["output_max"][:3]), -1.0, 1.0)
            action[-1] = gripper
            result = runtime.step(action)
            sample(phase, action)
            return not result.terminated and not result.truncated

        def move(phase, point, gripper, limit=90):
            stable = 0
            for _ in range(limit):
                if not step(phase, point, gripper):
                    return False
                stable = stable + 1 if np.linalg.norm(point - env.sim.data.site_xpos[eef_id]) < 0.004 else 0
                if stable >= 5:
                    phases.append(phase)
                    return True
            return False

        if policy == "hold":
            while step("hold_initial"):
                pass
        else:
            targets = [
                ("pregrasp", initial_cube + [0.0, 0.0, 0.12], -1.0),
                ("grasp", initial_cube + [0.0, 0.0, 0.005], -1.0),
            ]
            for phase, point, gripper in targets:
                if not move(phase, point, gripper):
                    reason = phase + "_tracking_failed"
                    break
            if reason is None:
                pinch = env.sim.data.site_xpos[eef_id].copy()
                for _ in range(30):
                    if not step("close", pinch, 1.0):
                        reason = "close_budget_exhausted"
                        break
                if reason is None:
                    lift = pinch + [0.0, 0.0, 0.15]
                    if not move("lift", lift, 1.0):
                        reason = "lift_tracking_failed"
                    else:
                        for _ in range(30):
                            if not step("hold_lift", lift, 1.0):
                                reason = "hold_budget_exhausted"
                                break
        relative = np.array([r["relative_position"] for r in held])
        spread = float(np.linalg.norm(np.ptp(relative, axis=0))) if len(held) else None
        lift_valid = bool(
            reason is None
            and len(held) == 30
            and all(r["bilateral_grasp_contact"] and r["cube"][2] > initial_cube[2] + 0.08 for r in held)
            and spread < 0.01
        )
        if task == "Stack" and policy == "scripted_reference" and reason is None:
            if not lift_valid:
                reason = "lift_not_retained"
            else:
                offset = env.sim.data.body_xpos[target_id] - env.sim.data.site_xpos[eef_id]
                support = env.sim.data.body_xpos[env.cubeB_body_id].copy()
                goal = support + [0.0, 0.0, target.size[2] + env.cubeB.size[2] + 0.002]
                release = goal - offset
                for phase, point in (("preplace", release + [0.0, 0.0, 0.12]), ("place", release)):
                    if not move(phase, point, 1.0):
                        reason = phase + "_tracking_failed"
                        break
                if reason is None:
                    for _ in range(25):
                        if not step("release", release, -1.0):
                            reason = "release_budget_exhausted"
                            break
                if reason is None:
                    retreat = release + [0.0, 0.0, 0.15]
                    if not move("retreat", retreat, -1.0):
                        reason = "retreat_tracking_failed"
                    else:
                        for _ in range(30):
                            if not step("hold_place", retreat, -1.0):
                                reason = "place_settle_budget_exhausted"
                                break
        place_spread = float(np.linalg.norm(np.ptp(np.array([r["cube"] for r in placed]), axis=0))) if placed else None
        place_valid = (
            bool(
                len(placed) == 30
                and place_spread < 0.005
                and all(
                    r["support_contact"]
                    and not r["bilateral_grasp_contact"]
                    and np.linalg.norm(np.array(r["cube"]) - r["eef"]) > 0.10
                    and np.linalg.norm(np.array(r["cube"][:2]) - r["support"][:2]) < 0.015
                    and abs(r["cube"][2] - r["support"][2] - target.size[2] - env.cubeB.size[2]) < 0.01
                    for r in placed
                )
            )
            if task == "Stack"
            else False
        )
        strict = bool(reason is None and lift_valid and (task == "Lift" or place_valid))
        stage_detail = reason
        if reason is not None and runtime.control_steps >= max_steps:
            reason = "diagnostic_step_budget_exhausted"
        scores = runtime.scores(final_goal=None, physical_integrity=None)
        result = {
            "task": task,
            "seed": seed,
            "policy": policy,
            "status": "completed",
            "scores": scores,
            "strict_reference_success": strict,
            "expected_task_success": policy == "scripted_reference",
            "diagnostic_passed": strict == (policy == "scripted_reference")
            and scores["upstream_success"] == (policy == "scripted_reference"),
            "completed_phases": phases,
            "failure_reason": reason,
            "stage_detail": stage_detail,
            "settled_lift_samples": len(held),
            "lift_retained": lift_valid,
            "relative_position_spread_m": spread,
            "settled_place_samples": len(placed),
            "placement_settled": place_valid,
            "placement_position_spread_m": place_spread,
            "initial_state_sha256": initial_hash,
            "initial_ctrl_sha256": hashlib.sha256(initial_ctrl.tobytes()).hexdigest(),
            "controller_sha256": hashlib.sha256((output / "controller.json").read_bytes()).hexdigest(),
            "compiled_model_sha256": hashlib.sha256((output / "model.mjb").read_bytes()).hexdigest(),
            "upstream_task_source_sha256": hashlib.sha256(task_source).hexdigest(),
            "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "control_steps": runtime.control_steps,
            "upstream_horizon": env.horizon,
            "diagnostic_step_cap": max_steps,
            "diagnostic_stop_reason": reason
            or ("diagnostic_step_cap" if runtime.control_steps >= max_steps else "scripted_sequence_complete"),
            "episode_identity": episode.identity,
            "physical_integrity_qualified": False,
        }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--tasks", nargs="+", choices=["Lift", "Stack"], default=["Lift", "Stack"])
    parser.add_argument(
        "--policies", nargs="+", choices=["scripted_reference", "hold"], default=["scripted_reference", "hold"]
    )
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()
    if (
        not args.seeds
        or min(args.seeds) < 0
        or len(set(args.seeds)) != len(args.seeds)
        or not 1 <= args.max_steps <= 1000
    ):
        parser.error("unique nonnegative seeds and 1..1000 diagnostic steps required")
    if len(set(args.policies)) != len(args.policies):
        parser.error("policies must be unique")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "reference_source.py").write_bytes(Path(__file__).read_bytes())
    manifest = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "robosuite_version": importlib.metadata.version("robosuite"),
        "mujoco_version": importlib.metadata.version("mujoco"),
        "scope": "privileged native reference diagnostic; no TAMP/agent or observed task score",
        "seeds": args.seeds,
        "tasks": args.tasks,
        "policies": args.policies,
        "max_steps": args.max_steps,
        "upstream_horizon": 1000,
        "reset_certified": False,
        "execution": "native_env_step",
        "cameras": False,
        "vlm_calls": 0,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cases = []
    for task in args.tasks:
        for seed in args.seeds:
            for policy in args.policies:
                case_dir = args.output / f"{task}-{seed}-{policy}"
                try:
                    result = run_case(case_dir, task=task, seed=seed, policy=policy, max_steps=args.max_steps)
                except Exception as error:
                    result = {
                        "task": task,
                        "seed": seed,
                        "policy": policy,
                        "status": "error",
                        "diagnostic_passed": False,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                    case_dir.mkdir(parents=True, exist_ok=True)
                    (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                cases.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
    paired = {}
    for task in args.tasks:
        for seed in args.seeds:
            rows = [r for r in cases if r["seed"] == seed and r["task"] == task]
            hashes = [
                tuple(
                    r.get(k)
                    for k in ("initial_state_sha256", "initial_ctrl_sha256", "model_xml_sha256", "controller_sha256")
                )
                for r in rows
            ]
            paired[f"{task}:{seed}"] = len(rows) == 2 and all(None not in h for h in hashes) and len(set(hashes)) == 1
    summary = {
        "scope": manifest["scope"],
        "cases": cases,
        "paired_initial_states_match": paired,
        "reference_successes": sum(
            r.get("strict_reference_success", False) for r in cases if r["policy"] == "scripted_reference"
        ),
        "reference_cases": sum(r["policy"] == "scripted_reference" for r in cases),
        "diagnostic_passed": all(r["diagnostic_passed"] for r in cases)
        and (len(args.policies) == 1 or all(paired.values())),
        "observed_task_success_assessed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0 if summary["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
