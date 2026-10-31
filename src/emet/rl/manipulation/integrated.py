"""Continuous physical pick/transport/place benchmark; no attachment or stage resets."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv, ResetMotionUnreachable
from emet.rl.manipulation.tamp_cases import execute_case, plan_case

# Detect expected contact early; authorization still requires half the weight.
CONTACT_WEIGHT_FRACTION = 0.05
MIN_CONTACT_FORCE_N = 0.01
SUPPORT_LOSS_GRACE_S = 0.1
MAX_SUPPORT_PROBE_M = 0.005
SUPPORT_PROBE_STEP_M = 0.0015
RELEASE_RETREAT_UP_M = 0.06


def _run_case(
    *,
    robot="innate_mars",
    seed=1800000,
    policy=None,
    action_mode="residual",
    payload_mass_kg=None,
    payload_asset_set="training",
    payload_object=None,
    grasp_depth=0.03,
    grasp_pitch_rad=0.0,
    contact_cone="pyramidal",
    contact_impratio=1.0,
    balanced_grasp_setup=False,
):
    if policy is not None:
        action_mode = policy.action_mode
    env = ManipulationEnv(
        ManipulationConfig(
            robot=robot,
            scenario="pregrasp",
            horizon=30,
            randomization_scale=0,
            payload_mass_kg=payload_mass_kg,
            payload_asset_set=payload_asset_set,
            contact_cone=contact_cone,
            contact_impratio=contact_impratio,
            balanced_grasp_setup=balanced_grasp_setup,
            payload_object=payload_object,
            action_mode=action_mode,
            base_xy_tolerance=0.005,
            grasp_lift_height=0.045,
            support_half_width=0.04,
            grasp_depth=grasp_depth,
            grasp_pitch_rad=grasp_pitch_rad,
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
            return {
                "seed": seed,
                "success": False,
                "failure": "initialization_failed",
                "initialization_failure_reason": info.get("initialization_failure_reason"),
                "fingerprint": env.fingerprint,
                "initial_qpos": start,
                "stages": stages,
                "payload_properties": info["randomization"],
            }
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
        env.resume_stage("carry")
        env.control.set_base_bias_compensation(True)
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
        env.resume_stage("place", reset_clock=True)
        # Both transport segments verified retention and clearance. Rearming
        # clears bias conservatively; explicitly authorize it for free-space
        # descent, just as for transport. The first support contact below
        # disables it before any support probe or grip release.
        env.control.set_base_bias_compensation(True)
        target = grasp_configuration
        from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, segment

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
        contact_seen = False
        support_lost_seconds = 0.0
        support_recontacts = 0
        probe_attitude_adjustments = 0
        probe_origin = None
        probe_rotation = None
        retreat = None
        retreat_start = None
        retreat_backoff_m = None
        completion_hold = False
        release_clear_seconds = 0.0
        lift_retreat_start = None
        release_clearance_verified = False
        for _ in range(round(30 / env.profile.dt)):
            elapsed = float(env.data.time) - descent_start
            support_load = env.payload_support_load()
            weight = env.model.body_mass[env.payload] * 9.81
            supported = support_load >= 0.5 * weight
            contact = support_load >= max(MIN_CONTACT_FORCE_N, CONTACT_WEIGHT_FRACTION * weight)
            if contact and not supported_stop and not env._released:
                env.control.set_base_bias_compensation(False)
                # Expected placement contact ends descent; do not keep driving
                # the grasp into the support while waiting for convergence.
                supported_stop = True
                contact_seen = True
                support_lost_seconds = 0.0
                if probe_origin is None:
                    probe_origin = float(env.data.xpos[env.control.ee, 2])
                    probe_rotation = env.data.xmat[env.control.ee].reshape(3, 3).copy()
                env.control.set_reference(env.control.positions(), phase="place")
                env.control.set_pose_targets(
                    env.control.base_goal, env.control.base_mask.copy(), env.control.ee_pose(), True
                )
                env.control.hold_arm_position()
            if supported:
                support_lost_seconds = 0.0
            if supported_stop and not supported and not env._released:
                support_lost_seconds += env.profile.dt
                if (
                    support_lost_seconds >= SUPPORT_LOSS_GRACE_S
                    and probe_origin - float(env.data.xpos[env.control.ee, 2]) < MAX_SUPPORT_PROBE_M
                ):
                    # A transient corner touch is not a permanent support latch.
                    # Resume only the existing bounded, slow contact probe.
                    supported_stop = False
                    support_recontacts += 1
            if not env._released and not supported_stop:
                reference = descent.sample(elapsed)[0]
                if elapsed > descent.duration or contact_seen:
                    if probe_origin is None:
                        probe_origin = float(env.data.xpos[env.control.ee, 2])
                        probe_rotation = env.data.xmat[env.control.ee].reshape(3, 3).copy()
                    if probe_origin - float(env.data.xpos[env.control.ee, 2]) < MAX_SUPPORT_PROBE_M:
                        remaining_probe = MAX_SUPPORT_PROBE_M - (probe_origin - float(env.data.xpos[env.control.ee, 2]))
                        probe_step = -min(SUPPORT_PROBE_STEP_M, remaining_probe)
                        try:
                            reference = env._vertical_reference(probe_step, rotation=probe_rotation)
                        except ResetMotionUnreachable:
                            current_rotation = env.data.xmat[env.control.ee].reshape(3, 3)
                            angle = np.arccos(np.clip((np.trace(probe_rotation.T @ current_rotation) - 1) / 2, -1, 1))
                            if angle > env.config.hand_orientation_tolerance / 2:
                                raise
                            # A small base attitude change can make a fixed
                            # orientation probe unreachable for a five-DOF arm.
                            # Preserve measured orientation within the bounded
                            # contact attitude window instead of forcing it.
                            reference = env._vertical_reference(probe_step)
                            probe_attitude_adjustments += 1
                    else:
                        reference = env.control.reference.copy()
                env.control.set_reference(reference, phase="place")
                env._refresh_hand_target()
            if env._released:
                open_grip = env.control.gripper_is_open()
                if open_grip and retreat is None:
                    approach_axis = env.data.xmat[env.control.ee].reshape(3, 3)[:2, 0]
                    retreat_goal = withdraw_goal = None
                    backoffs = (0.02, 0.03, 0.04, 0.05) if env.profile.robot == "innate_mars" else (0.0,)
                    for backoff in backoffs:
                        retreat_xy = env.data.xpos[env.control.ee, :2] - backoff * approach_axis
                        try:
                            withdraw_goal = env._vertical_reference(0, xy=retreat_xy)
                            retreat_goal = env._vertical_reference(RELEASE_RETREAT_UP_M, xy=retreat_xy)
                        except ResetMotionUnreachable:
                            continue
                        retreat_backoff_m = backoff
                        break
                    if retreat_goal is None:
                        raise ResetMotionUnreachable("no reachable bounded release retreat")
                    # Withdraw from the opened pincer before lifting. Encoder
                    # opening alone does not prove the object cleared the pads.
                    parts = []
                    if env.profile.robot == "innate_mars":
                        parts.append(
                            segment(
                                env.control.positions(),
                                withdraw_goal,
                                np.asarray(env.profile.speed) * 0.5,
                                np.asarray(env.profile.acceleration) * 0.5,
                            )
                        )
                    start = parts[-1].duration if parts else 0.0
                    parts.append(
                        segment(
                            withdraw_goal if parts else env.control.positions(),
                            retreat_goal,
                            np.asarray(env.profile.speed) * 0.5,
                            np.asarray(env.profile.acceleration) * 0.5,
                            start=start,
                        )
                    )
                    retreat = ArmTrajectory(parts)
                    retreat_start = float(env.data.time)
                if retreat is not None:
                    command_time = float(env.data.time) - retreat_start
                    if len(retreat.segments) > 1:
                        boundary = retreat.segments[1].start
                        if command_time >= boundary and lift_retreat_start is None:
                            touching = False
                            for contact in env.data.contact[: env.data.ncon]:
                                a, b = (int(env.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                                other = b if a == env.payload else a if b == env.payload else None
                                if (
                                    other is not None
                                    and contact.dist < 0
                                    and any(env.monitor._descendant(other, finger) for finger in env.monitor.fingers)
                                ):
                                    touching = True
                                    break
                            clear = not touching and env.payload_support_load() >= 0.9 * weight
                            release_clear_seconds = release_clear_seconds + env.profile.dt if clear else 0.0
                            if release_clear_seconds >= 0.1:
                                release_clearance_verified = True
                                env.control.set_base_bias_compensation(True)
                                lift_retreat_start = float(env.data.time)
                        command_time = (
                            boundary + float(env.data.time) - lift_retreat_start
                            if lift_retreat_start is not None
                            else min(command_time, boundary)
                        )
                    env.control.set_reference(retreat.sample(command_time)[0], phase="release", release=True)
                    env._refresh_hand_target()
                    if command_time >= retreat.end and not completion_hold:
                        position_error, angle_error = env._hand_pose_error()
                        delta = env.control.base_pose() - env.control.base_goal
                        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
                        if (
                            position_error < 0.8 * env.config.hand_position_tolerance
                            and angle_error < 0.8 * env.config.hand_orientation_tolerance
                            and np.linalg.norm(delta[:2]) < 0.8 * env.config.base_xy_tolerance
                            and abs(delta[2]) < 0.8 * env.config.base_heading_tolerance
                            and np.max(abs(env.control.velocities())) < 0.03
                            and abs(env.last_base_velocity[0]) < 0.005
                            and abs(env.last_base_velocity[1]) < 0.02
                        ):
                            # Preserve the requested hand/base goals. A quiet,
                            # accurate endpoint can enter a normal hold while
                            # physics and the unchanged two-second gate run.
                            env.control.hold_position()
                            completion_hold = True
            obs = env.control.observe()
            obs, _, done, truncated, info = env.step(env.teacher_action() if policy is None else policy(obs))
            if done or truncated:
                break
        stages.append(
            {
                "stage": "lower_release_settle",
                **info,
                "released": env._released,
                "support_recontacts": support_recontacts,
                "probe_attitude_adjustments": probe_attitude_adjustments,
                "retreat_backoff_m": retreat_backoff_m,
                "completion_hold": completion_hold,
                "release_clearance_verified": release_clearance_verified,
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
    from emet.rl.manipulation.carry_benchmark import digest, source_hashes

    sources = digest(source_hashes())
    result = _run_case(**kwargs)
    if digest(source_hashes()) != sources:
        raise RuntimeError("control sources changed during pick/place execution")
    policy = kwargs.get("policy")
    result.update(
        scope="continuous simulator pick and return-place with GT TAMP arm paths; not a full public-tool task",
        controller="velocity_teacher" if policy is None else policy.manifest["weights_sha256"],
        control_source_sha256=sources,
        action_mode=policy.action_mode if policy else kwargs.get("action_mode", "residual"),
        requested_payload_mass_kg=kwargs.get("payload_mass_kg"),
        fixture_grasp_depth_m=kwargs.get("grasp_depth", 0.03),
        fixture_grasp_pitch_rad=kwargs.get("grasp_pitch_rad", 0.0),
        payload_properties=(
            result["stages"][0].get("randomization", {})
            if result.get("stages")
            else result.get("payload_properties", {})
        ),
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
    p.add_argument("--payload-asset-set", choices=["training", "native_table"], default="training")
    p.add_argument("--payload-object", choices=["object1", "object2"])
    p.add_argument("--grasp-depth", type=float, default=0.03)
    p.add_argument("--grasp-pitch-rad", type=float, default=0.0)
    p.add_argument("--contact-cone", choices=["pyramidal", "elliptic"], default="pyramidal")
    p.add_argument("--balanced-grasp-setup", action="store_true")
    p.add_argument("--contact-impratio", type=float, default=1.0)
    p.add_argument(
        "--payload-masses-kg",
        type=float,
        nargs="+",
        help="simulation mass sweep; uniform density scaling at fixed geometry and COM",
    )
    p.add_argument("--action-mode", choices=["residual", "direct_velocity"], default="residual")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    from emet.rl.manipulation.policy import ManipulationPolicy

    policy = ManipulationPolicy(args.policy, robot=args.robot) if args.policy else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mass in args.payload_masses_kg or [None]:
        for i in range(args.episodes):
            row = run_case(
                robot=args.robot,
                seed=args.seed + i,
                policy=policy,
                action_mode=args.action_mode,
                payload_mass_kg=mass,
                payload_asset_set=args.payload_asset_set,
                payload_object=args.payload_object,
                grasp_depth=args.grasp_depth,
                grasp_pitch_rad=args.grasp_pitch_rad,
                contact_cone=args.contact_cone,
                contact_impratio=args.contact_impratio,
                balanced_grasp_setup=args.balanced_grasp_setup,
            )
            rows.append(row)
            args.output.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
            print(
                "payload", mass, "seed", args.seed + i, "success", row["success"], "failure", row["failure"], flush=True
            )
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
