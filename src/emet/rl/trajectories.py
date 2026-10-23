"""Simulator tracking gate using navigation/TAMP waypoint constructors.

This tests the controller contract, not complete pick/place execution or payload
clearance. Intermediate points advance from measurements without resetting the
controller history; the final point requires velocity settling.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.motion.mujoco_arm_ik import interpolate_arm_waypoints
from emet.motion.navigation_sweep import differential_drive_waypoints
from emet.rl.base import MarsBaseEnv
from emet.rl.control import Goal, baseline_action
from emet.rl.env import EnvConfig, MarsReachEnv
from emet.rl.policy import PPOPolicy


def evaluate_trajectories(policy=None, *, arm=False, coordinated=False, episodes=20, seed=120000, horizon=15):
    if arm and coordinated:
        raise ValueError("arm-only and coordinated trajectory modes are mutually exclusive")
    task = "coordinated" if coordinated else "reach" if arm else "base"
    config = EnvConfig(
        task=task,
        arm_only=arm,
        pose_targets=coordinated,
        horizon_seconds=horizon,
    )
    env = MarsReachEnv(config) if arm or coordinated else MarsBaseEnv(config)
    rows = []
    try:
        for index in range(episodes):
            env.reset(seed=seed + index)
            final = env.control.goal
            if arm:
                points = interpolate_arm_waypoints(env.control.arm_positions(), final.arm, n_steps=5)[1:]
                goals = [Goal(final.base.copy(), point, final.mask.copy()) for point in points]
            elif coordinated:
                from emet.rl.waypoints import coordinated_waypoints

                goals = coordinated_waypoints(env.control.base_pose(), env.control.arm_positions(), final, 5)
            else:
                points = differential_drive_waypoints(env.control.base_pose(), final.base, max_translation_m=0.1)
                goals = [Goal(np.asarray(point), final.arm.copy(), final.mask.copy()) for point in points]
            waypoint = 0
            env.set_goal(goals[waypoint])
            env._scenario_ready = waypoint == len(goals) - 1
            peak_force = 0.0
            any_collision = False
            errors = []
            for _ in range(round(horizon / 0.05)):
                obs, measured = env.control.observe()
                if coordinated:
                    obs = env._with_pose_targets(obs)
                if arm:
                    arrived = measured["joint_error"] < 0.025
                elif coordinated:
                    arrived = (
                        measured["xy_error"] <= env.control.profile.xy_tolerance
                        and measured["yaw_error"] <= env.control.profile.yaw_tolerance
                        and (not env.control.goal.mask[1:].any() or (
                            measured["ee_position_error"] <= env.control.profile.ee_position_tolerance
                            and measured["ee_orientation_error"] <= env.control.profile.ee_orientation_tolerance
                        ))
                    )
                else:
                    arrived = measured["xy_error"] < 0.02 and measured["yaw_error"] < 0.035
                if arrived and waypoint < len(goals) - 1:
                    waypoint += 1
                    env.set_goal(goals[waypoint])
                    env._scenario_ready = waypoint == len(goals) - 1
                    obs, _ = env.control.observe()
                    if coordinated:
                        obs = env._with_pose_targets(obs)
                action = (
                    policy(obs)
                    if policy is not None
                    else baseline_action(env.control)
                    if arm or coordinated
                    else env.control.baseline_action()
                )
                _, _, terminated, truncated, info = env.step(action)
                peak_force = max(peak_force, info["peak_contact_force"])
                any_collision |= info["collision"]
                errors.append(info["ee_position_error"] if arm or coordinated else info["xy_error"])
                if terminated or truncated:
                    break
            success = bool(info["is_success"] and waypoint == len(goals) - 1)
            rows.append(
                {
                    "seed": seed + index,
                    "is_success": success,
                    "code": "ok" if success else "collision" if any_collision else "tracking_timeout",
                    "waypoints": len(goals),
                    "last_waypoint": waypoint,
                    "time": info["time"],
                    "collision": any_collision,
                    "peak_contact_force": peak_force,
                    "joint_error": info["joint_error"],
                    "xy_error": info["xy_error"],
                    "yaw_error": info["yaw_error"],
                    "ee_position_error": info["ee_position_error"],
                    "ee_orientation_error": info["ee_orientation_error"],
                    "mean_active_waypoint_position_error": float(np.mean(errors)),
                }
            )
    finally:
        env.close()
    return {
        "arm": arm,
        "coordinated": coordinated,
        "episodes": rows,
        "success_rate": float(np.mean([r["is_success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "scope": "simulator waypoint tracking only; no grasp, payload, or live TAMP certification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy")
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--coordinated", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--horizon", type=float, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    report = evaluate_trajectories(
        PPOPolicy(args.policy) if args.policy else None,
        arm=args.arm,
        coordinated=args.coordinated,
        episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
