#!/usr/bin/env python3
"""Replay Herman's wrist-only wave with the feedback or PPO controller in MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from emet.rl.control import Goal, baseline_action
from emet.rl.env import EnvConfig, MarsReachEnv
from emet.rl.profile import MARS

WRIST = "joint5"
POSE_JOINTS = (*MARS.arm_joints, *MARS.held_joints)
WAVE_OFFSETS_RAD = (0.15, -0.15, 0.0)
POSITION_TOLERANCE_RAD = 0.025
VELOCITY_TOLERANCE_RAD_S = 0.04


def run(seed, pose=None, *, policy=None, max_speed=MARS.max_joint_speed):
    env = MarsReachEnv(EnvConfig(arm_only=True))
    try:
        env.reset(seed=seed)
        control = env.control
        joints = [env.model.joint(name) for name in POSE_JOINTS]
        position_indices = [joint.qposadr[0] for joint in joints]
        velocity_indices = [joint.dofadr[0] for joint in joints]
        wrist = POSE_JOINTS.index(WRIST)
        others = np.arange(len(joints)) != wrist
        report = {
            "seed": seed,
            "controller": "feedback" if policy is None else "ppo",
            "max_wrist_command_speed_rad_s": max_speed,
            "waypoints": [],
            "max_base_displacement_m": 0.0,
            "max_other_arm_drift_rad": 0.0,
            "peak_contact_force_n": 0.0,
            "success": False,
        }
        if pose is not None:
            report["start_pose_rad"] = dict(zip(POSE_JOINTS, pose, strict=True))
            for name, joint, value in zip(POSE_JOINTS, joints, pose, strict=True):
                if not joint.range[0] <= value <= joint.range[1]:
                    return {**report, "reason": f"{name} outside model limits"}
            env.data.qpos[position_indices] = pose
            # The MuJoCo gripper equality couples the follower to -joint6.
            env.data.qpos[env.model.joint("joint6M").qposadr[0]] = -pose[POSE_JOINTS.index("joint6")]
            mujoco.mj_forward(env.model, env.data)
            control.reset()
        start = env.data.qpos[position_indices].copy()
        report["start_pose_rad"] = dict(zip(POSE_JOINTS, start.tolist(), strict=True))
        report["start_arm_rad"] = {name: report["start_pose_rad"][name] for name in MARS.arm_joints}
        _, initial = control.observe()
        report["minimum_model_clearance_m"] = initial["clearance"]
        if initial["collision"]:
            contacts = sorted(
                {
                    tuple(sorted((env.model.geom(c.geom1).name, env.model.geom(c.geom2).name)))
                    for c in env.data.contact
                    if c.dist < -0.001 and (c.geom1 in control.robot_geoms or c.geom2 in control.robot_geoms)
                }
            )
            return {**report, "reason": "starting pose in model collision", "contact_pairs": contacts}

        base = control.base_pose().copy()
        mask = np.zeros(6)
        mask[wrist + 1] = 1
        targets = start[wrist] + np.array(WAVE_OFFSETS_RAD)
        lo, hi = control.limits[wrist]
        if np.any(targets < lo) or np.any(targets > hi):
            return {**report, "reason": "wrist target outside model limits"}

        # The environment's normal success tolerances are looser than the
        # wave's. Keep each goal active until the measured pose has settled.
        env._scenario_ready = False
        action_limit = max_speed / MARS.max_joint_speed
        settle_steps = round(MARS.settle_seconds / MARS.control_dt)
        for target in targets:
            arm = start[: len(MARS.arm_joints)].copy()
            arm[wrist] = target
            env.set_goal(Goal(base, arm, mask))
            observation, _ = control.observe()
            settled = 0
            for _ in range(round(3.0 / MARS.control_dt)):
                action = baseline_action(control) if policy is None else policy(observation)
                if action.shape != env.action_space.shape or not np.isfinite(action).all():
                    return {**report, "reason": "invalid controller action", "target_rad": target}
                action = np.clip(action, -action_limit, action_limit)
                observation, _, _, truncated, info = env.step(action)
                q = env.data.qpos[position_indices]
                report["max_base_displacement_m"] = max(
                    report["max_base_displacement_m"],
                    float(np.linalg.norm(control.base_pose()[:2] - base[:2])),
                )
                drift = float(np.max(np.abs((q - start)[others])))
                report["max_other_arm_drift_rad"] = max(report["max_other_arm_drift_rad"], drift)
                report["peak_contact_force_n"] = max(report["peak_contact_force_n"], info["peak_contact_force"])
                report["minimum_model_clearance_m"] = min(report["minimum_model_clearance_m"], info["clearance"])
                for failure in (
                    "collision",
                    "unsafe_contact",
                    "invalid_physics",
                    "joint_limit_violation",
                    "obstruction",
                ):
                    if info[failure]:
                        return {**report, "reason": failure, "target_rad": target}
                if drift > POSITION_TOLERANCE_RAD:
                    return {**report, "reason": "non-wrist joint drift", "target_rad": target}
                if truncated:
                    return {**report, "reason": "episode timeout"}
                wrist_error = abs(q[wrist] - target)
                stationary = np.max(np.abs(env.data.qvel[velocity_indices])) <= VELOCITY_TOLERANCE_RAD_S
                settled = settled + 1 if wrist_error <= POSITION_TOLERANCE_RAD and stationary else 0
                if settled >= settle_steps:
                    report["waypoints"].append(
                        {
                            "target_rad": target,
                            "measured_rad": float(q[wrist]),
                            "abs_error_rad": float(wrist_error),
                            "time_s": info["time"],
                        }
                    )
                    break
            else:
                return {**report, "reason": "waypoint timeout", "target_rad": target}
        report["success"] = True
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--pose", type=float, nargs=7, help="joint1 through joint6 followed by joint_head, in radians")
    parser.add_argument("--policy", type=Path, help="Arm-only PPO bundle directory")
    parser.add_argument("--max-speed", type=float, default=MARS.max_joint_speed, help="Wrist command cap, in rad/s")
    parser.add_argument("--output", type=Path, help="Save the full JSON report")
    args = parser.parse_args()
    if not 0 < args.max_speed <= MARS.max_joint_speed:
        parser.error(f"--max-speed must be above zero and at most {MARS.max_joint_speed} rad/s")
    if args.pose is not None and not np.isfinite(args.pose).all():
        parser.error("--pose must contain finite joint angles")
    policy = None
    if args.policy:
        from emet.rl.policy import PPOPolicy

        policy = PPOPolicy(args.policy)
        if policy.task != "reach" or not policy.manifest.get("training", {}).get("env", {}).get("arm_only"):
            parser.error("--policy must be a compatible arm-only reach bundle")
    seeds = args.seeds[:1] if args.pose is not None else args.seeds
    results = [run(seed, args.pose, policy=policy, max_speed=args.max_speed) for seed in seeds]
    success_count = sum(row["success"] for row in results)
    payload = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
        print(f"{success_count}/{len(results)} waves completed. Report saved to {args.output}")
    else:
        print(payload, end="")
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
