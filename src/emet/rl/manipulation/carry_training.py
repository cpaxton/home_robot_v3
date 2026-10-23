"""Generate nontrivial, timed TAMP carry references on scratch planner state."""

import numpy as np

from emet.eval.physical_tamp import make_scene_checker
from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, segment
from emet.rl.manipulation.grasp_setup import support_margin
from emet.rl.manipulation.tamp_cases import plan_case
from emet.rl.manipulation.tracking import TaskFrameFK


def target_offsets(env, *, lift_end=None):
    """Nearby reachable witnesses; IK is offline generation, never actor input."""
    mode = env.config.carry_target_mode
    if mode == "orientation":
        offsets = np.zeros((12, env.profile.arm_size))
        offsets[:, -1] = env.np_random.choice([-1, 1], 12) * env.np_random.uniform(0.1, 0.3, 12)
        return offsets
    if mode == "translation":
        start = env.control.positions() if lift_end is None else lift_end
        height = 0 if lift_end is None else env.config.grasp_lift_height
        offsets = []
        for delta in (-0.015, 0.015):
            try:
                offsets.append(env._vertical_reference(height + delta) - start)
            except ValueError:
                continue
        return np.array(offsets) if offsets else np.zeros((1, env.profile.arm_size))
    return (
        env.np_random.uniform(-1, 1, (12, env.profile.arm_size))
        * np.array([0.03, 0.02, 0.12, 0.08, 0.08])
        * env.target_scale
    )


def prepare(env):
    control, profile = env.control, env.profile
    base, start = control.base_pose().copy(), control.positions().copy()
    fk = TaskFrameFK(env.model, env.data, control)
    initial = fk(base, start).copy()
    offsets = target_offsets(env)
    for attempt in range(12):
        evidence = {"candidate_id": attempt, "accepted": False}

        def reject(reason, **details):
            env.trajectory_rejections.append({**evidence, "reason": reason, **details})

        goal = (
            start
            + np.asarray(env.randomization.get("grasp_setup", {}).get("carry_joint_offset") or [0.0] * profile.arm_size)
            if attempt == 0
            else np.clip(
                start + offsets[(attempt - 1) % len(offsets)],
                control.limits[:, 0] + 0.02,
                control.limits[:, 1] - 0.02,
            )
        )
        hand = fk(base, goal)
        a, b = initial[3:].reshape(3, 2), hand[3:].reshape(3, 2)
        rotation_a = np.column_stack((a, np.cross(a[:, 0], a[:, 1])))
        rotation_b = np.column_stack((b, np.cross(b[:, 0], b[:, 1])))
        distance = float(np.linalg.norm(hand[:3] - initial[:3]))
        angle = float(np.arccos(np.clip((np.trace(rotation_a.T @ rotation_b) - 1) / 2, -1, 1)))
        evidence.update(endpoint_translation_m=distance, endpoint_orientation_rad=angle, goal=goal.tolist())
        if distance < 0.01 and angle < 0.1:
            reject("trivial_carry_target")
            continue
        if distance > 0.05 or angle > 0.3:
            reject("carry_target_outside_nearby_distribution")
            continue
        case = plan_case(env, goal)
        if case["planning_error"]:
            reject("planning_failure", planning_error=case["planning_error"], contacts=case["contacts"])
            continue
        names = list(case["joint_names"])
        points = [
            np.array([sum(q[names.index(name)] for name in group) for group in profile.joint_groups])
            for q in case["waypoints"]
        ]
        parts, elapsed = [], 0.0
        for q, target in zip(points[:-1], points[1:], strict=True):
            part = segment(
                q, target, np.asarray(profile.speed) * 0.5, np.asarray(profile.acceleration) * 0.5, start=elapsed
            )
            parts.append(part)
            elapsed += part.duration
        if not parts or elapsed + 2 > env.config.horizon:
            reject("carry_reference_exceeds_horizon", required_duration_s=elapsed + 2, horizon_s=env.config.horizon)
            continue
        path = ArmTrajectory(parts)
        retarget_time = None
        if env.np_random.random() < env.config.retarget_probability:
            # Change the announced endpoint midway through the first segment.
            # A derivative-continuous return is rechecked with the held payload.
            retarget_time = max(profile.dt, round(parts[0].duration * 0.5 / profile.dt) * profile.dt)
            q, v, a, _ = path.sample(retarget_time)
            returning = segment(
                q,
                start,
                np.asarray(profile.speed) * 0.5,
                np.asarray(profile.acceleration) * 0.5,
                start=retarget_time,
                velocity=v,
                accel=a,
            )
            path = ArmTrajectory([parts[0], returning], replan_time=retarget_time)
            if path.end + 2 > env.config.horizon:
                reject(
                    "carry_reference_exceeds_horizon", required_duration_s=path.end + 2, horizon_s=env.config.horizon
                )
                continue
        checker = make_scene_checker(
            env.model, {"object_body": env.payload_name, "gripper_bodies": profile.finger_bodies}
        )
        checker.set_payload(env.model, env.data, env.payload_name, profile.ee_body)
        minimum = float("inf")
        rejected = False
        rejection = None
        for t in np.linspace(0, path.end, max(10, int(path.end / profile.dt) + 1)):
            sampled_hand = fk(base, path.sample(t)[0]).copy()
            if checker.configuration_collides(env.model, fk.data):
                rejected = True
                rejection = {
                    "reason": "carry_path_collision",
                    "time_s": float(t),
                    "failure_hand": sampled_hand.tolist(),
                    "contacts": list(checker.last_contacts),
                }
                break
            support = support_margin(env.model, fk.data, profile, env.payload)
            minimum = min(minimum, support["margin_m"])
            if minimum < 0.005:
                rejected = True
                rejection = {
                    "reason": "carry_support_margin",
                    "time_s": float(t),
                    "failure_hand": sampled_hand.tolist(),
                    **support,
                }
                break
        if rejected or minimum < 0.005:
            reject(**rejection)
            continue
        env.arm_trajectory = path
        # Initialization established a physical grasp and clearance. Never
        # accumulate wheel bias against a table-supported grasp or approach.
        control.set_base_bias_compensation(True)
        if retarget_time is not None:
            env.retarget_step = round(retarget_time / profile.dt)
            env.retarget_goal = (base.copy(), start.copy(), initial.copy())
        env.tamp_reference_case = case
        env.trajectory_admission = {
            **evidence,
            "accepted": True,
            "reason": "timed_tamp_carry",
            "planner": case["planner"],
            "endpoint_translation_m": distance,
            "endpoint_orientation_rad": angle,
            "minimum_sampled_margin_m": minimum,
            "dynamic_feasibility_verified": False,
        }
        control.set_pose_targets(base, np.ones(3, dtype=bool), initial, True)
        env._advance_dynamic_reference()
        return
    env.initialization_failed = True
    env.initialization_failure_reason = "no_admitted_tamp_carry_reference"
    env.trajectory_admission = {"accepted": False, "reason": env.initialization_failure_reason}
    control.stop()
