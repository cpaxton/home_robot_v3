"""Paired, seeded whole-body tracking audit, including retargets and finite pushes.

Run with --output PATH; --policy adds a checkpoint evaluated on identical cases.
No hardware transport or fixture modifications are used.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv

KINDS = ("hand", "base", "combined", "retarget", "push")


def target(env, rng, kind):
    control = env.control
    base = control.base_pose()
    arm = control.positions()
    if kind != "hand":
        base += rng.uniform([-0.06, -0.06, -0.4], [0.06, 0.06, 0.4])
    if kind != "base":
        arm = np.clip(
            arm + rng.uniform(-0.12, 0.12, env.profile.arm_size),
            control.limits[:, 0] + 0.02,
            control.limits[:, 1] - 0.02,
        )
    hand = env._fk_hand_pose(base, arm)
    control.set_reference(arm, phase="reach")
    control.set_pose_targets(base, np.full(3, kind != "hand"), hand, kind != "base")
    env.settle_time = 0
    env._previous_cost = env._tracking_cost()
    return {
        "base": base.tolist(),
        "arm": arm.tolist(),
        "hand": hand.tolist(),
        "base_active": kind != "hand",
        "hand_active": kind != "base",
    }


def run_case(env, seed, kind, policy=None):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1000000)
    targets = [target(env, rng, kind)]
    obs = env.control.observe()
    changed = False
    for step in range(round(env.config.horizon / env.profile.dt)):
        if kind == "retarget" and step == 10:
            targets.append(target(env, rng, "combined"))
            obs = env.control.observe()
            changed = True
        env.data.xfrc_applied[env.control.ee, 0] = 0.5 if kind == "push" and 10 <= step < 15 else 0
        env.control.action_mode = policy.action_mode if policy is not None else env.config.action_mode
        action = env.teacher_action() if policy is None else policy(obs)
        obs, _, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.data.xfrc_applied[:] = 0
    base = env.control.base_pose().copy()
    hand = env.control.ee_pose().copy()
    # Keep the controller and physics alive after success: freezing the final
    # frame would hide drift. Safety monitoring remains active during this hold.
    hold_force = info["peak_contact_force"]
    if info["is_success"] and not env.monitor.wheel_lift and not info["unsafe_contact"]:
        env.control.stopped = False
        for _ in range(round(2 / env.profile.dt)):
            obs = env.control.observe()
            env._tick(env.teacher_action() if policy is None else policy(obs))
        hold_force = max(hold_force, env.monitor.peak_force)
    return {
        "seed": seed,
        "kind": kind,
        "targets": targets,
        "retarget_applied": changed,
        **info,
        "hold_base_drift_m": float(np.linalg.norm(env.control.base_pose()[:2] - base[:2])),
        "hold_hand_drift_m": float(np.linalg.norm(env.control.ee_pose()[:3] - hand[:3])),
        "including_hold_peak_force_n": hold_force,
        "including_hold_wheel_lift": bool(env.monitor.wheel_lift),
    }


def audit(*, robot="innate_mars", episodes=10, seed=810000, policy=None):
    config = ManipulationConfig(
        robot=robot,
        scenario="reach",
        horizon=20,
        initial_heading_range=np.pi,
        action_mode=policy.action_mode if policy else "residual",
        trajectory_preview=policy is not None and "preview" in policy.normalization,
        tracking_reward=policy is not None and policy.manifest.get("environment", {}).get("tracking_reward", False),
    )
    env = ManipulationEnv(config)
    if policy is not None and (policy.robot != robot or policy.manifest["fingerprint"] != env.fingerprint):
        env.close()
        raise ValueError("policy robot/model mismatch")
    rows = []
    try:
        for k, kind in enumerate(KINDS):
            for i in range(episodes):
                row = run_case(env, seed + k * 10000 + i, kind, policy)
                rows.append(row)
                print(kind, row["seed"], "success", row["is_success"], flush=True)
        summary = {}
        metrics = (
            "ee_position_error",
            "ee_orientation_error",
            "base_goal_error",
            "time",
            "peak_measured_jerk",
            "peak_base_jerk",
            "peak_base_linear_jerk_m_s3",
            "peak_base_yaw_jerk_rad_s3",
            "including_hold_peak_force_n",
            "hold_base_drift_m",
            "hold_hand_drift_m",
        )
        for kind in KINDS:
            group = [r for r in rows if r["kind"] == kind]
            summary[kind] = {
                "cases": len(group),
                "success_rate": float(np.mean([r["is_success"] for r in group])),
                "wheel_lift_count": sum(r["including_hold_wheel_lift"] for r in group),
                "unsafe_contact_count": sum(r["unsafe_contact"] for r in group),
            }
            for metric in metrics:
                values = [r[metric] for r in group]
                summary[kind][metric] = {
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(max(values)),
                }
        return {
            "config": asdict(config),
            "fingerprint": env.fingerprint,
            "controller": "teacher" if policy is None else policy.manifest["weights_sha256"],
            "seed": seed,
            "summary": summary,
            "episodes": rows,
            "scope": "simulation; nearby FK-reachable targets, not a dynamic feasibility certificate",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("innate_mars", "stretch"), default="innate_mars")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=810000)
    parser.add_argument("--policy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    from emet.utils.cpu_affinity import apply_eval_affinity

    apply_eval_affinity()
    reports = {"teacher": audit(robot=args.robot, episodes=args.episodes, seed=args.seed)}
    if args.policy:
        from emet.rl.manipulation.policy import ManipulationPolicy

        reports["policy"] = audit(
            robot=args.robot, episodes=args.episodes, seed=args.seed, policy=ManipulationPolicy(args.policy)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
