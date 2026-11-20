"""Screen a shared Mars actor in closed-loop simulator task families."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.eval import mars_tamp_control as tamp
from emet.rl.manipulation.arm_spline_benchmark import save, stop
from emet.rl.manipulation.carry_benchmark import default_config
from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.expert_routing import bind_expert
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.safety import relative_pose, retention_error, whole_body_stopped
from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_observation import ACTOR_KEYS, actor_observation
from emet.rl.regression_gate import file_digest

FAMILIES = ("base", "arm", "coupled", "carry", "tamp", "lift", "preplace")
FLAGS = ("unsafe_contact", "retention_lost", "wheel_lift", "invalid_physics")


def pass_flags(info, post):
    return {key: bool(info.get(key) or post.get(key)) for key in FLAGS}


def track(case, actor):
    env = CoupledTrackingEnv("innate_mars", [case], suite_path=COARSE_SUITE)
    plant = env.unwrapped
    mode = case["mode"]
    plant.config = replace(plant.config, wheel_feedback_base_only_gains=(0.0, 0.5))
    try:
        obs, _ = env.reset(options={"case_id": case["id"]})
        base_origin = plant.control.base_pose()[:2].copy()
        peak_base_drift = peak_arm_speed = 0.0
        family = "coupled_hand" if mode in {"coupled", "hand_free_base"} else mode
        student = bind_expert(
            mode,
            plant.control,
            {family: lambda observation: actor({k: observation[k] for k in ACTOR_KEYS})},
            arm_speed_limit=0.25,
        )
        for _ in range(550):
            obs, _, done, _, info = env.step(student(obs))
            if mode == "arm_only":
                peak_base_drift = max(
                    peak_base_drift, float(np.linalg.norm(plant.control.base_pose()[:2] - base_origin))
                )
                peak_arm_speed = max(peak_arm_speed, float(np.max(np.abs(plant.control.velocities()))))
            if done:
                break
        confirmed = stop(env)
        flags = pass_flags(info, plant._info())
        success = bool(done and info.get("primitive_success") and confirmed and not any(flags.values()))
        if mode != "base_only":
            success &= bool(info["peak_hand_position_error"] <= 0.003 and info["peak_hand_orientation_error"] <= 0.02)
        if mode == "arm_only":
            success &= peak_base_drift <= 0.01 and peak_arm_speed <= 0.25
        return {
            "success": success,
            "code": info.get("code"),
            "flags": flags,
            "stop_confirmed": confirmed,
            "peak_hand_position_m": info["peak_hand_position_error"],
            "peak_hand_orientation_rad": info["peak_hand_orientation_error"],
            "arm_only_peak_base_drift_m": peak_base_drift if mode == "arm_only" else None,
            "arm_only_peak_joint_speed_rad_s": peak_arm_speed if mode == "arm_only" else None,
        }
    finally:
        env.close()


def approach(case, actor):
    env = TampPrimitiveEnv(
        [case],
        oracle=True,
        primitive="grasp_pose",
        planning_scope="unloaded_prefix",
        wheel_feedback_current_velocity=True,
        wheel_feedback_base_only_gains=(0.0, 0.5),
    )
    try:
        obs, info = env.reset(options={"case_id": case["id"]})
        fields = history_layout(MARS, retention_context=True)[0]
        calls = 0
        for _ in range(2000):
            action = np.asarray(info.get("oracle_action", np.zeros(MARS.action_size))).copy()
            hand_active = "oracle_action" in info and obs["history"][-1, fields["hand_mask"]].item() > 0.5
            if hand_active:
                action = actor({key: obs[key] for key in ACTOR_KEYS}).copy()
                action[:2] = 0.0
                action[2:-1] = np.clip(action[2:-1], -0.25 / np.asarray(MARS.speed), 0.25 / np.asarray(MARS.speed))
                calls += 1
            action[-1] = 0.0
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        flags = pass_flags(info, {})
        stopped = bool(info.get("stop_confirmed"))
        return {
            "success": bool(calls and info.get("primitive_success") and stopped and not any(flags.values())),
            "code": info.get("code"),
            "flags": flags,
            "stop_confirmed": stopped,
            "actor_calls": calls,
        }
    finally:
        env.close()


def loaded(family, case, actor):
    if family == "lift":
        config = json.loads(Path("configs/benchmarks/mars_grasp_lift_curriculum_v1.json").read_text())["environment"]
        config.update(
            scenario="carry",
            grasp_lift_probability=1.0,
            carry_push_probability=0.0,
            payload_mass_kg=0.02,
            payload_mass_range_kg=None,
            randomization_scale=0.0,
        )
        cfg = ManipulationConfig(**config)
    else:
        nominal = case["payload_mass_kg"]
        cfg = replace(
            default_config(),
            retention_context=True,
            payload_mass_kg=None,
            payload_mass_range_kg=(nominal, nominal),
            payload_com_randomization_m=0.003,
            randomization_scale=case["randomization_scale"],
            grasp_pitch_rad=case["grasp_pitch_rad"],
            grasp_setup_prefer_nominal=case["prefer_nominal_grasp"],
            carry_target_mode=case["motion"],
            carry_push_probability=float(case["pushed"]),
            carry_push_scale=0.1 if case["pushed"] else 0.0,
        )
    env = ManipulationEnv(cfg)
    try:
        raw, info = env.reset(seed=case["seed"])
        for _ in range(math.ceil(cfg.horizon / env.profile.dt) + 1):
            if info.get("initialization_failed"):
                break
            action = actor(actor_observation(env.control, raw)).copy()
            if family == "lift":
                # Supported lift delegates only the arm. The native servo
                # retains its stationary-base feedback.
                action[:2] = 0.0
            action[-1] = 0.0
            raw, _, done, truncated, info = env.step(action)
            if done or truncated or info.get("is_success"):
                break
        confirmed = stop(env)
        if family == "carry" and confirmed:
            for _ in range(round(0.3 / env.profile.dt)):
                if env._physics_invalid:
                    break
                env._tick(np.zeros(env.profile.action_size), score=False)
            confirmed = confirmed and not env._physics_invalid and whole_body_stopped(env.control)
        flags = pass_flags(info, env._info())
        slip, turn = retention_error(env.retention_reference, relative_pose(env.data, env.control.ee, env.payload))
        retained = slip is not None and slip <= 0.02 and turn <= 0.1
        lifted = family != "lift" or env.data.xpos[env.payload, 2] - env.initial_object_height > 0.04
        success = bool(info.get("is_success") and confirmed and not any(flags.values()) and retained and lifted)
        return {
            "success": success,
            "code": info.get("code"),
            "flags": flags,
            "stop_confirmed": confirmed,
            "stop_slip_m": slip,
            "stop_rotation_rad": turn,
        }
    finally:
        env.close()


def preplace(case, actor):
    class LearnedPreplace(MarsSimRobot):
        def __init__(self, env):
            super().__init__(env)
            self.servo.retention_context_enabled = True
            self.servo.history = None
            self.transported = False
            self.actor_calls = 0

        def follow_trajectory(self, path, *, loaded=True):
            result = super().follow_trajectory(path, loaded=loaded)
            self.transported |= bool(loaded and result.success)
            return result

        def _joint_teacher_action(self):
            action = super()._joint_teacher_action()
            if self.transported and self.phase == "carry":
                self.actor_calls += 1
                action[2:-1] = actor(actor_observation(self.servo))[2:-1]
            return action

    robots = []

    def factory(env):
        robot = LearnedPreplace(env)
        robots.append(robot)
        return robot

    with patch.object(tamp, "MarsSimRobot", factory):
        result = tamp.run_case(case)
    calls = robots[0].actor_calls if robots else 0
    return {
        "success": bool(result["success"] and calls),
        "code": result.get("robot_failure"),
        "final_predicates": result.get("final_predicates"),
        "actor_calls": calls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required")
    torch.set_num_threads(1)
    actor = WholeBodyBundle(args.actor, robot="innate_mars")
    manifest = json.loads(args.cases.read_text())
    declared = manifest["cases"]
    cases = [case for case in declared if case["family"] in args.families]
    if (
        not cases
        or any(case["split"] != "audit" or case["family"] not in FAMILIES for case in declared)
        or len({(case["family"], case["id"]) for case in declared}) != len(declared)
    ):
        raise ValueError("Explicit audit cases are required")
    report = {
        "complete": False,
        "mode": "audit",
        "weights_sha256": actor.manifest["weights_sha256"],
        "cases_sha256": file_digest(args.cases),
        "rows": [],
    }
    for case in cases:
        family = case["family"]
        if family in {"arm", "base", "coupled"}:
            item = track(case, actor)
        elif family == "tamp":
            item = approach(case, actor)
        elif family == "preplace":
            item = preplace(case, actor)
        else:
            item = loaded(family, case, actor)
        report["rows"].append({**{k: case[k] for k in ("id", "seed", "family")}, **item})
        save(args.output, report)
        print(family, case["id"], item["success"], item["code"], flush=True)
    report["complete"] = True
    save(args.output, report)


if __name__ == "__main__":
    main()
