"""Opt-in Cartesian arm execution for checkpointed Mars simulation only.

The joint path is a planning witness. Execution follows its FK pose schedule,
with fresh configuration/collision/support checks at each control cycle, rather
than claiming the actor reproduced that particular IK solution. This is sampled
online checking, not a continuous collision certificate or hardware backend.
"""

import mujoco
import numpy as np

from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
from emet.rl.manipulation.dynamic_trajectory import segment
from emet.rl.manipulation.grasp_setup import support_margin
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory

TRACKING_POSITION_LIMIT_M = 0.01
TRACKING_ORIENTATION_LIMIT_RAD = 0.10


def lag_scaled_rate(position_error, orientation_error, *, position_scale_m, orientation_scale_rad):
    """Slow inside the path tube without making endpoint accuracy a waypoint gate."""
    if (
        not np.isfinite([position_error, orientation_error, position_scale_m, orientation_scale_rad]).all()
        or min(position_error, orientation_error) < 0
        or min(position_scale_m, orientation_scale_rad) <= 0
    ):
        raise ValueError("finite nonnegative lag and positive retiming scales required")
    return float(
        min(
            position_scale_m / max(position_error, position_scale_m),
            orientation_scale_rad / max(orientation_error, orientation_scale_rad),
        )
    )


def pose_error(current, target):
    a, b = np.asarray(current), np.asarray(target)
    if a.shape != (9,) or b.shape != (9,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf"), float("inf")
    ca, cb = a[3:].reshape(3, 2), b[3:].reshape(3, 2)
    ra = np.column_stack([ca, np.cross(ca[:, 0], ca[:, 1])])
    rb = np.column_stack([cb, np.cross(cb[:, 0], cb[:, 1])])
    return float(np.linalg.norm(a[:3] - b[:3])), float(np.arccos(np.clip((np.trace(rb.T @ ra) - 1) / 2, -1, 1)))


def execute_pose_path(robot, executor, phase, path, *, goal_pose=None):
    from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot

    if not isinstance(robot, MarsSimRobot) or robot.policy is None or robot.arm_execution != "cartesian":
        raise ValueError("Cartesian arm execution requires the explicit Mars simulator policy backend")

    def fail(reason, **detail):
        robot.stop_and_confirm(reason)
        executor.synchronize(executor.data)
        executor.event(phase=phase, execution_contract="cartesian_arm_v1", reason=reason, **detail)
        return PhysicalMotionResult(False, reason, phase)

    if robot.failure or robot.servo.stopped or robot.env._physics_invalid:
        return fail(robot.failure or "controller_stopped")
    try:
        lag_scaled_rate(
            0,
            0,
            position_scale_m=executor.position_tolerance_m,
            orientation_scale_rad=executor.orientation_tolerance_rad,
        )
    except ValueError:
        return fail("invalid_cartesian_tolerance")
    witness = np.asarray(path, dtype=float)
    if (
        witness.ndim != 2
        or witness.shape[1:] != (5,)
        or not 1 <= len(witness) <= 1000
        or not np.isfinite(witness).all()
    ):
        return fail("invalid_cartesian_witness")
    executor.synchronize(executor.data)
    current = executor.data.qpos[executor.qadr].copy()
    if np.max(abs(current - witness[0])) > executor.joint_tolerance:
        return fail("stale_arm_plan")
    limits = np.array([executor.model.joint(n).range for n in executor.joint_names])
    margins = np.array([executor.joint_limit_margins.get(n, 0) for n in executor.joint_names])
    base = robot.servo.base_pose().copy()
    origin = robot.data.xpos[robot.servo.base].copy()
    origin_rotation = robot.data.xmat[robot.servo.base].reshape(3, 3).copy()
    scratch = mujoco.MjData(robot.model)

    def configuration_reason(q=None):
        # Copy all C-backed views before planner/contact operations.
        mujoco.mj_copyData(scratch, robot.model, robot.data)
        if q is not None:
            scratch.qpos[executor.qadr] = q
        measured = scratch.qpos[executor.qadr].copy()
        if (
            not np.isfinite(measured).all()
            or np.any(measured < limits[:, 0] + margins)
            or np.any(measured > limits[:, 1] - margins)
        ):
            return "arm_joint_margin"
        if executor.collision.configuration_collides(robot.model, scratch):
            return "arm_configuration_collision"
        payload = robot.payload if robot.retention_reference is not None else None
        if support_margin(robot.model, scratch, MARS, payload)["margin_m"] < 0.005:
            return "arm_static_support_margin"
        return None

    def guard():
        if robot.failure:
            return robot.failure
        # Retention queries and adapter events read this private snapshot.
        # Refresh it alongside the checks that copy directly from physics.
        executor.synchronize(executor.data)
        position = robot.data.xpos[robot.servo.base].copy()
        rotation = robot.data.xmat[robot.servo.base].reshape(3, 3).copy()
        angle = np.arccos(np.clip((np.trace(origin_rotation.T @ rotation) - 1) / 2, -1, 1))
        if not np.isfinite([*position, angle]).all() or np.linalg.norm(position - origin) > 0.01 or angle > 0.02:
            return "base_drift_during_arm"
        if not executor.payload_retained():
            return "payload_not_retained"
        return configuration_reason()

    # Certify a densely sampled, derivative-bounded witness before issuing any
    # motion. Only private data is modified; the live robot is never reset.
    continuous = robot.arm_reference == "jerk_limited"
    reference_path = derivative_certificate = governor = None
    q_samples, times = [witness[0].copy()], [0.0]
    if continuous:
        from emet.rl.manipulation.jerk_reference import SmoothPhase, compile_joint_path

        tau = 0.5
        reference_path, derivative_certificate = compile_joint_path(
            witness,
            MARS.speed,
            MARS.acceleration,
            4 * np.asarray(MARS.acceleration),
            phase_acceleration=1 / tau,
            phase_jerk=2 / tau**2,
        )
        times = sorted(
            {
                0.0,
                reference_path.end,
                *np.arange(0, reference_path.end, MARS.dt),
                *(part.start for part in reference_path.segments),
            }
        )
        q_samples = [reference_path.sample(t)[0] for t in times]
        for q in q_samples:
            reason = configuration_reason(q)
            if reason:
                return fail(reason, check="planning_witness")
    else:
        for start, end in zip(witness[:-1], witness[1:], strict=True):
            if np.max(abs(end - start)) < 1e-10:
                continue
            part = segment(start, end, MARS.speed, MARS.acceleration)
            for elapsed in np.linspace(0, part.duration, max(2, int(np.ceil(part.duration / MARS.dt)) + 1))[1:]:
                q = part.sample(elapsed)[0]
                reason = configuration_reason(q)
                if reason:
                    return fail(reason, check="planning_witness")
                q_samples.append(q)
                times.append(times[-1] + part.duration / (max(2, int(np.ceil(part.duration / MARS.dt)) + 1) - 1))
    reason = guard()
    if reason:
        return fail(reason, check="measured_start")
    if times[-1] > 30:
        return fail("cartesian_schedule_too_long")
    if len(times) == 1:
        times.append(MARS.dt)
        q_samples.append(q_samples[0].copy())
    release = robot.phase == "release"
    if release and not robot.servo.release_authorized:
        return fail("release_authorization_required")
    if phase in {"pregrasp", "grasp"} and not robot.servo.gripper_is_open():
        return fail("gripper_open_unverified")
    # Opening/closing the gripper changes the servo's phase. Restore the
    # current execution stage through its public setter before announcing a
    # Cartesian path, just as command_arm_reference does for joint execution.
    # Keep measured history and grip; never clear a fault or authorize release.
    grip = None
    if robot.phase in {"reach", "pregrasp"} and robot.retention_reference is None:
        # A previous supported release may leave the nominal closed reference
        # intact. Preserve the measured unloaded aperture when motion resumes;
        # never rebase a contact-loaded grasp and thereby relax its grip.
        grip = float(np.clip(robot.data.qpos[robot.servo.grip_q], *robot.servo.grip_limits))
    robot.servo.set_reference(robot.servo.positions(), phase=robot.phase, grip=grip, release=release)
    grip_goal = robot.env.profile.gripper_open if release else robot.servo.grip_reference
    fk = TaskFrameFK(robot.model, robot.data, robot.servo)
    hand = np.array([fk(base, q) for q in q_samples])
    if goal_pose is not None:
        # TAMP's actual requested pose is the endpoint. The IK witness only
        # establishes nearby geometric admission; do not add execution error to
        # its allowed IK residual and thereby double the final task tolerance.
        distance, angle = pose_error(hand[-1], goal_pose)
        if distance > executor.position_tolerance_m or angle > executor.orientation_tolerance_rad:
            return fail("cartesian_witness_goal_mismatch", position_error_m=distance, orientation_error_rad=angle)
    if goal_pose is not None and not continuous:
        correction = TaskPoseTrajectory(
            [0, 1],
            [base, base],
            [hand[-1], goal_pose],
            [grip_goal] * 2,
            base_mask=np.ones(3, dtype=bool),
        )
        tail, start = [], times[-1]
        for u in np.linspace(0, 1, 11)[1:]:
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            tail.append(correction.sample(blend)[1])
            times.append(start + 0.5 * u)
        hand = np.vstack([hand, tail])
    if continuous:
        from emet.rl.manipulation.jerk_pose_reference import FKPoseReference

        schedule = FKPoseReference(
            fk,
            base,
            reference_path,
            grip_goal,
            derivative_certificate,
            goal_pose=goal_pose,
            phase_acceleration=1 / tau,
            phase_jerk=2 / tau**2,
        )
        if schedule.end > 30:
            return fail("cartesian_schedule_too_long")
        governor = SmoothPhase(schedule.end, tau=tau)
        hand = schedule.hand
    else:
        schedule = TaskPoseTrajectory(
            times,
            np.tile(base, (len(times), 1)),
            hand,
            np.full(len(times), grip_goal),
            base_mask=np.ones(3, dtype=bool),
        )
    executor.event(
        phase=phase,
        execution_contract="cartesian_arm_v1",
        task_pose_trajectory=schedule.record(),
        joint_witness=witness.tolist(),
        position_tolerance_m=executor.position_tolerance_m,
        orientation_tolerance_rad=executor.orientation_tolerance_rad,
        configuration_check_period_s=MARS.dt,
        reference_clock="filtered_lag_scaling" if continuous else "continuous_lag_scaling",
        arm_reference=robot.arm_reference,
        arm_derivative_certificate=derivative_certificate,
        tracking_position_limit_m=TRACKING_POSITION_LIMIT_M,
        tracking_orientation_limit_rad=TRACKING_ORIENTATION_LIMIT_RAD,
    )
    robot.commands.append(
        {
            "phase": phase,
            "sim_time": float(robot.data.time),
            "target_source": "certified_path_fk_schedule",
            "execution_contract": "cartesian_arm_v1",
            "task_pose_trajectory": schedule.record(),
            "controller": robot.controller_name,
        }
    )
    robot.servo.set_task_trajectory(schedule)
    robot._pose_command_received = True
    progress, stable = 0.0, 0.0
    anchor = None
    robot.servo.set_trajectory_progress(0, rate=0)
    robot.reference_phase = governor
    robot.reference_phase_schedule = schedule if governor is not None else None
    if governor is not None:
        robot.servo.preview_sampler = lambda stamp: schedule.sample(
            governor.preview_progress(stamp - float(robot.data.time))
        )
    for _ in range(int(np.ceil((3 * schedule.end + 4) / MARS.dt))):
        reason = guard()
        if reason:
            return fail(reason, check="measured_configuration")
        distance, angle = pose_error(robot.servo.ee_pose(), robot.servo.hand_goal)
        # Retiming waits for the actual hand; the whole future pose path stays
        # visible to the actor. Large tracking departures fail independently.
        if distance > TRACKING_POSITION_LIMIT_M or angle > TRACKING_ORIENTATION_LIMIT_RAD:
            return fail("cartesian_path_tracking_envelope", position_error_m=distance, orientation_error_rad=angle)
        rate = lag_scaled_rate(
            distance,
            angle,
            position_scale_m=executor.position_tolerance_m,
            orientation_scale_rad=executor.orientation_tolerance_rad,
        )
        if not robot.servo.holding:
            progress = governor.advance(rate, MARS.dt) if governor else min(schedule.end, progress + rate * MARS.dt)
            robot.servo.set_trajectory_progress(progress, rate=governor.rate if governor else rate)
        robot.step(MARS.dt)
        reason = guard()
        if reason:
            return fail(reason, check="measured_configuration")
        distance, angle = pose_error(robot.servo.ee_pose(), hand[-1])
        within = distance <= executor.position_tolerance_m and angle <= executor.orientation_tolerance_rad
        if progress >= schedule.end and within:
            if not robot.servo.holding:
                robot.servo.hold_position()
                anchor = robot.servo.positions().copy()
            if (
                whole_body_stopped(robot.servo)
                and np.max(abs(robot.servo.positions() - anchor)) <= executor.ENDPOINT_STILLNESS_TOLERANCE
            ):
                stable += MARS.dt
            else:
                stable = 0
                anchor = robot.servo.positions().copy()
            if stable >= max(MARS.dt, executor.endpoint_settle_s):
                executor.event(
                    phase=f"{phase}_complete",
                    execution_contract="cartesian_arm_v1",
                    position_error_m=distance,
                    orientation_error_rad=angle,
                    joint_witness_residual_rad=float(np.max(abs(robot.servo.positions() - witness[-1]))),
                    controller_success=True,
                )
                return PhysicalMotionResult(True, "measured_cartesian_path", phase, distance)
        else:
            stable = 0
    return fail("cartesian_tracking_timeout", position_error_m=distance, orientation_error_rad=angle)
