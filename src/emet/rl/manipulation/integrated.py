"""Continuous physical pick/transport/place benchmark; no attachment or stage resets."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv, ResetMotionUnreachable
from emet.rl.manipulation.tamp_cases import execute_case, plan_case


def _run_case(*, robot="innate_mars", seed=1800000, policy=None, action_mode="residual"):
    if policy is not None:
        action_mode = policy.action_mode
    env = ManipulationEnv(
        ManipulationConfig(
            robot=robot,
            scenario="pregrasp",
            horizon=30,
            randomization_scale=0,
            action_mode=action_mode,
            base_xy_tolerance=0.005,
            grasp_lift_height=0.045,
            support_half_width=0.04,
            grasp_depth=0.03,
            hand_position_tolerance=0.005,
            hand_orientation_tolerance=0.05,
            trajectory_preview=policy is not None and "preview" in policy.normalization,
            tracking_reward=policy is not None and policy.manifest.get("environment", {}).get("tracking_reward", False),
        )
    )
    stages = []
    try:
        if policy is not None and (policy.robot != robot or policy.manifest["fingerprint"] != env.fingerprint):
            raise ValueError("integrated policy/model mismatch")
        obs, info = env.reset(seed=seed)
        # The object starts on its physical pedestal and the open hand above it.
        # No further reset, qpos edits, welds, or object relocation is allowed.
        start = env.data.qpos.copy().tolist()
        placement_xy = env.data.xpos[env.payload, :2].copy()
        grasp_configuration = env.control.reference.copy()
        if info["initialization_failed"]:
            return {"seed": seed, "success": False, "failure": "initialization_failed", "stages": stages}
        env.control.set_pose_targets(
            env.control.base_pose(),
            np.ones(3, dtype=bool),
            env._fk_hand_pose(env.control.base_pose(), env.control.reference),
            True,
        )
        for _ in range(round(30 / env.profile.dt)):
            obs, _, done, truncated, info = env.step(env.teacher_action() if policy is None else policy(obs))
            if done or truncated:
                break
        stages.append({"stage": "approach_close_lift", **info})
        if not info["is_success"]:
            return {"seed": seed, "success": False, "failure": info["code"], "stages": stages, "initial_qpos": start}
        # Re-arm a completed contract without resetting physics or retention.
        env.control.stopped = False
        env.scenario = "carry"
        env._done = False
        env.settle_time = 0
        # A small closed excursion exercises actual TAMP planner trajectories
        # while returning above the original support for placement.
        lifted = env.control.positions().copy()
        offset = np.zeros(env.profile.arm_size)
        offset[2] = 0.03
        for target in (np.clip(lifted + offset, env.control.limits[:, 0], env.control.limits[:, 1]), lifted):
            case = plan_case(env, target)
            if case["planning_error"]:
                stages.append({"stage": "transport", "planning_error": case["planning_error"]})
                return {"seed": seed, "success": False, "failure": "planning_failed", "stages": stages}
            result = execute_case(env, case, policy, hold_base=True)
            stages.append({"stage": "transport", **result})
            if not result["execution_verified"]:
                return {
                    "seed": seed,
                    "success": False,
                    "failure": result["controller_failure"] or "transport_failed",
                    "stages": stages,
                }
        env.scenario = "place"
        env._done = False
        env.control.stopped = False
        env.settle_time = 0
        env.steps = 0
        target = grasp_configuration
        from emet.rl.manipulation.dynamic_trajectory import segment

        descent = segment(
            env.control.positions(),
            target,
            np.asarray(env.profile.speed) * 0.4,
            np.asarray(env.profile.acceleration) * 0.4,
            time_scale=2,
        )
        descent_start = float(env.data.time)
        env.control.set_reference(descent.sample(0)[0], phase="place")
        env._refresh_hand_target()
        env._previous_cost = env._tracking_cost()
        obs = env.control.observe()
        supported_stop = False
        probe_origin = None
        probe_rotation = None
        retreat = None
        retreat_start = None
        for _ in range(round(30 / env.profile.dt)):
            elapsed = float(env.data.time) - descent_start
            supported = env.payload_support_load() >= 0.5 * env.model.body_mass[env.payload] * 9.81
            if supported and not supported_stop:
                # Expected placement contact ends descent; do not keep driving
                # the grasp into the support while waiting for convergence.
                supported_stop = True
                env.control.set_reference(env.control.positions(), phase="place")
                env._refresh_hand_target()
            if not env._released and not supported_stop:
                reference = descent.sample(elapsed)[0]
                if elapsed > descent.duration:
                    if probe_origin is None:
                        probe_origin = float(env.data.xpos[env.control.ee, 2])
                        probe_rotation = env.data.xmat[env.control.ee].reshape(3, 3).copy()
                    if probe_origin - float(env.data.xpos[env.control.ee, 2]) < 0.005:
                        reference = env._vertical_reference(-0.0015, rotation=probe_rotation)
                    else:
                        reference = env.control.reference.copy()
                env.control.set_reference(reference, phase="place")
                env._refresh_hand_target()
            if env._released:
                open_grip = abs(env.data.qpos[env.control.grip_q] - env.profile.gripper_open) < 0.003
                if open_grip and retreat is None:
                    retreat = segment(
                        env.control.positions(),
                        env._vertical_reference(0.06),
                        np.asarray(env.profile.speed) * 0.4,
                        np.asarray(env.profile.acceleration) * 0.4,
                    )
                    retreat_start = float(env.data.time)
                if retreat is not None:
                    env.control.set_reference(
                        retreat.sample(float(env.data.time) - retreat_start)[0], phase="release", release=True
                    )
                    env._refresh_hand_target()
            obs = env.control.observe()
            obs, _, done, truncated, info = env.step(env.teacher_action() if policy is None else policy(obs))
            if done or truncated:
                break
        stages.append(
            {
                "stage": "lower_release_settle",
                **info,
                "released": env._released,
                "support_load_n": env.payload_support_load(),
                "settle_time": env.settle_time,
                "joint_velocity": env.control.velocities().tolist(),
                "base_velocity": env.last_base_velocity.tolist(),
                "object_velocity": env.data.qvel[
                    env.model.body(env.payload).dofadr[0] : env.model.body(env.payload).dofadr[0] + 6
                ].tolist(),
                "grip": float(env.data.qpos[env.control.grip_q]),
                "base": env.control.base_pose().tolist(),
                "base_goal": env.control.base_goal.tolist(),
                "object": env.data.xpos[env.payload].tolist(),
                "table": env.model.body_pos[env.model.body("table").id].tolist(),
                "ee": env.control.ee_pose().tolist(),
                "hand_goal": env.control.hand_goal.tolist(),
            }
        )
        placement_error = float(np.linalg.norm(env.data.xpos[env.payload, :2] - placement_xy))
        return {
            "placement_xy_error_m": placement_error,
            "payload": env.payload_name,
            "seed": seed,
            "robot": robot,
            "success": bool(info["is_success"] and placement_error <= 0.02),
            "failure": (None if placement_error <= 0.02 else "placement_accuracy")
            if info["is_success"]
            else info["code"],
            "stages": stages,
            "initial_qpos": start,
            "fingerprint": env.fingerprint,
            "scope": "continuous sim pick and return-place on one pedestal; not full public TAMP task",
        }
    except ResetMotionUnreachable as error:
        env.control.stop()
        return {"seed": seed, "success": False, "failure": "motion_infeasible", "detail": str(error), "stages": stages}
    finally:
        env.close()


def run_case(**kwargs):
    result = _run_case(**kwargs)
    policy = kwargs.get("policy")
    result.update(
        scope="continuous simulator pick and return-place with GT TAMP arm paths; not a full public-tool task",
        controller="velocity_teacher" if policy is None else policy.manifest["weights_sha256"],
        action_mode=policy.action_mode if policy else kwargs.get("action_mode", "residual"),
        contract={
            "lift_m": 0.045,
            "support_half_width_m": 0.04,
            "hand_position_m": 0.005,
            "hand_orientation_rad": 0.05,
            "placement_xy_m": 0.02,
            "retention_translation_m": 0.02,
            "retention_rotation_rad": 0.1,
            "settled_object_velocity": 0.02,
            "settled_seconds": 2,
        },
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=["stretch", "innate_mars"], default="innate_mars")
    p.add_argument("--seed", type=int, default=1800000)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--policy")
    p.add_argument("--action-mode", choices=["residual", "direct_velocity"], default="residual")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    from emet.rl.manipulation.policy import ManipulationPolicy

    policy = ManipulationPolicy(args.policy, robot=args.robot) if args.policy else None
    rows = [
        run_case(robot=args.robot, seed=args.seed + i, policy=policy, action_mode=args.action_mode)
        for i in range(args.episodes)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    print(
        json.dumps(
            {
                "successes": sum(r["success"] for r in rows),
                "episodes": len(rows),
                "failures": [r["failure"] for r in rows],
            }
        )
    )


if __name__ == "__main__":
    main()
