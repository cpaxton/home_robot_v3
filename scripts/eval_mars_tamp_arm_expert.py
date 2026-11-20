"""Compare teacher and arm expert on the native unloaded TAMP path executor."""

import argparse
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.observation_layout import CURRENT_PREVIEW_ROW, history_layout, preview_layout
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_training import load_development

ARM_SPEED_LIMIT = 0.25


def rollout(case, actor):
    env = TampPrimitiveEnv(
        [case],
        oracle=True,
        primitive="grasp_pose",
        planning_scope="unloaded_prefix",
        wheel_feedback_current_velocity=True,
        wheel_feedback_base_only_gains=(0.0, 0.5),
    )
    fields = history_layout(MARS, retention_context=True)[0]
    hand_position_error = preview_layout()[0]["hand_error"]
    action_cap = np.minimum(1.0, ARM_SPEED_LIMIT / np.asarray(MARS.speed))
    arm_decisions = 0
    peak_error = peak_speed = 0.0
    try:
        observation, info = env.reset(options={"case_id": case["id"]})
        for _ in range(2000):
            action = np.asarray(info.get("oracle_action", np.zeros(MARS.action_size))).copy()
            h = observation["history"][-1]
            if "oracle_action" in info and h[fields["hand_mask"]].item() > 0.5:
                arm_decisions += 1
                peak_error = max(
                    peak_error,
                    float(np.linalg.norm(observation["preview"][CURRENT_PREVIEW_ROW, hand_position_error][:3])),
                )
                peak_speed = max(peak_speed, float(np.max(np.abs(h[fields["proprio/velocities"]]))))
                if actor is not None:
                    action = actor({key: observation[key] for key in ("history", "preview", "motion")}).copy()
                action[:2] = 0
                action[2:-1] = np.clip(action[2:-1], -action_cap, action_cap)
                action[-1] = 0
            observation, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        else:
            raise RuntimeError("TAMP executor exceeded its decision budget")
        path = [
            {"phase": command["phase"], "schedule_sha256": digest(command["task_pose_trajectory"])}
            for command in info["commands"]
            if command.get("target_source") == "certified_path_fk_schedule"
        ]
        flags = {
            name: bool(info[name]) for name in ("wheel_lift", "unsafe_contact", "invalid_physics", "retention_lost")
        }
        return {
            "id": case["id"],
            "success": bool(info["primitive_success"]),
            "stop_confirmed": info["stop_confirmed"],
            "code": info["code"],
            "initial_state_sha256": info["initial_state_sha256"],
            "model_fingerprint": info["model_fingerprint"],
            "arm_decisions": arm_decisions,
            "peak_sampled_hand_error_m": peak_error,
            "peak_sampled_joint_speed_rad_s": peak_speed,
            "terminal_pose": info["terminal_pose"],
            "gripper_open": info["gripper_open"],
            "path": path,
            "safety_flags": flags,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required")
    torch.set_num_threads(1)
    actor = WholeBodyBundle(args.expert, robot="innate_mars")
    sources = source_hashes()
    suite, _ = load_development()
    report = {
        "complete": False,
        "hardware_used": False,
        "safety_promoted": False,
        "scope": "native unloaded planner prefix through open-gripper grasp pose",
        "shared_base_route": "teacher",
        "arm_command_speed_cap_rad_s": ARM_SPEED_LIMIT,
        "expert_sha256": actor.manifest["weights_sha256"],
        "sources": sources,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    for split in ("train_cases", "audit_cases"):
        for case in suite[split]:
            teacher, expert = rollout(case, None), rollout(case, actor)
            if any(teacher[k] != expert[k] for k in ("initial_state_sha256", "model_fingerprint")):
                raise ValueError(f"Unpaired fixture for {case['id']}")
            report["rows"].append({"split": split, "teacher": teacher, "expert": expert})
            save(args.output, report)
            print(case["id"], teacher["code"], expert["code"], flush=True)
    if source_hashes() != sources:
        raise RuntimeError("Simulator source changed during expert evaluation")
    report["complete"] = True
    save(args.output, report)


if __name__ == "__main__":
    main()
