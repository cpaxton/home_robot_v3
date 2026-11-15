"""Cartesian policy execution keeps independent geometric and measured gates."""

import mujoco
import numpy as np
import pytest

from emet.controller.task.tamp.mars_native_adapter import MarsNativeTableAdapter
from emet.controller.task.tamp.mars_pose_executor import execute_pose_path, pose_error
from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.eval.mars_tamp_control import fixture, suite
from emet.rl.manipulation.tracking import TaskFrameFK

pytestmark = pytest.mark.sim


@pytest.fixture
def pose_executor(request):
    env, admission = fixture(suite()["cases"][0])
    assert admission["accepted"], admission
    robot = MarsSimRobot(
        env,
        policy=lambda obs: np.zeros(8),
        arm_execution="cartesian",
        arm_reference=getattr(request, "param", "legacy"),
    )
    robot.servo.base_velocity_feedback_gain = 0.5
    adapter = MarsNativeTableAdapter(robot, model=env.model)
    executor = adapter._executor(env.payload_name)
    yield env, robot, executor, adapter
    env.close()


def test_cartesian_endpoint_does_not_require_arbitrary_joint_witness(pose_executor, monkeypatch):
    env, robot, executor, adapter = pose_executor
    q = robot.servo.positions()
    goal = q + [0, -0.015, 0.03, -0.015, 0]
    fk = TaskFrameFK(env.model, env.data, robot.servo)
    distance, angle = pose_error(robot.servo.ee_pose(), fk(robot.servo.base_pose(), goal))
    assert distance < executor.position_tolerance_m and angle < executor.orientation_tolerance_rad
    assert np.max(abs(goal - q)) > executor.joint_tolerance
    monkeypatch.setattr(robot, "_joint_teacher_action", lambda: pytest.fail("teacher fallback"))
    qpos = env.data.qpos.copy()
    result = executor._execute_path("pregrasp", [q, goal])
    assert result.success, result
    assert robot.policy_calls > 0 and robot.servo.holding and not robot.failure
    assert env.data.time > 1 and np.max(abs(robot.servo.positions() - goal)) > executor.joint_tolerance
    assert np.linalg.norm(env.data.qpos[:2] - qpos[:2]) < 0.01
    event = [x for x in adapter.events if x.get("phase") == "pregrasp_complete"][-1]
    assert event["execution_contract"] == "cartesian_arm_v1"
    assert event["position_error_m"] < executor.position_tolerance_m
    assert robot.commands[-1]["target_source"] == "certified_path_fk_schedule"


def test_stale_joint_witness_stops_before_policy(pose_executor):
    _, robot, executor, _ = pose_executor
    result = execute_pose_path(robot, executor, "pregrasp", [robot.servo.positions() + 0.02])
    assert not result.success and result.message == "stale_arm_plan"
    assert robot.policy_calls == 0 and robot.servo.stopped and robot.stop_confirmed


def test_late_ik_failure_does_not_hide_a_solved_pose_collision(pose_executor, monkeypatch):
    from emet.motion import mujoco_arm_ik

    _, robot, executor, adapter = pose_executor
    before = executor.data.qpos.copy()
    attempts = 0

    def solve(model, scratch, **kwargs):
        nonlocal attempts
        attempts += 1
        success = attempts == 1
        return mujoco_arm_ik.MujocoArmIkResult(success, scratch.qpos.copy(), 0.001 if success else 0.02, 1, 0.001)

    monkeypatch.setattr(mujoco_arm_ik, "solve_pose_ik", solve)
    monkeypatch.setattr(
        executor, "plan_joint_target", lambda *args, **kwargs: (None, "arm_path_failed:invalid_start:collision")
    )
    path, reason = executor.plan_pose(robot.servo.ee_pose()[:3], np.eye(3))
    assert path is None and reason == "arm_path_failed:invalid_start:collision"
    assert attempts > 1
    np.testing.assert_array_equal(executor.data.qpos, before)
    rows = [e for e in adapter.events if e.get("phase") == "pose_replan_attempt"]
    assert rows[0]["ik_success"] and not rows[-1]["ik_success"]


@pytest.mark.parametrize("witness_matches", [True, False])
def test_cartesian_stage_reconnects_only_a_valid_world_pose_witness(pose_executor, monkeypatch, witness_matches):
    from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
    from emet.controller.task.tamp import mars_pose_executor

    _, robot, executor, adapter = pose_executor
    measured = robot.servo.positions().copy()
    goal = measured.copy()
    pose = robot.servo.ee_pose()
    axes = pose[3:].reshape(3, 2)
    rotation = np.column_stack([axes, np.cross(axes[:, 0], axes[:, 1])])
    point = pose[:3].copy()
    if not witness_matches:
        point[0] += 0.02
    executor.grasp_targets = [(point, rotation)] * 3
    planned = []
    pose_planned = []

    def plan(goal_q):
        assert witness_matches
        planned.append(goal_q.copy())
        return [measured, goal_q], None

    def plan_pose(position, orientation):
        assert not witness_matches
        pose_planned.append(position.copy())
        return [measured, goal], None

    monkeypatch.setattr(executor, "plan_joint_target", plan)
    monkeypatch.setattr(executor, "plan_pose", plan_pose)
    monkeypatch.setattr(executor, "payload_retained", lambda: True)
    captured = []

    def execute(robot_arg, executor_arg, phase, path, **kwargs):
        captured.extend(path)
        return PhysicalMotionResult(True, "complete", phase)

    monkeypatch.setattr(mars_pose_executor, "execute_pose_path", execute)
    result = executor._execute_path("grasp", [measured + 0.02, goal])
    assert result.success
    if witness_matches:
        np.testing.assert_array_equal(planned, [goal])
        assert not pose_planned
    else:
        np.testing.assert_array_equal(pose_planned, [point])
        assert not planned
    np.testing.assert_array_equal(captured, [measured, goal])
    expected = "measured_to_joint_witness" if witness_matches else "measured_world_pose"
    assert any(e.get("reference_replan") == expected for e in adapter.events)


def test_measured_collision_after_inference_stops_even_if_pose_is_close(pose_executor, monkeypatch):
    _, robot, executor, _ = pose_executor
    monkeypatch.setattr(executor.collision, "configuration_collides", lambda *args: robot.policy_calls > 0)
    result = execute_pose_path(robot, executor, "pregrasp", [robot.servo.positions()])
    assert not result.success and result.message == "arm_configuration_collision"
    assert robot.policy_calls == 1 and robot.servo.stopped and robot.stop_confirmed


def test_cartesian_execution_keeps_full_base_drift_guard(pose_executor, monkeypatch):
    env, robot, executor, _ = pose_executor
    original = robot.step
    injected = False

    def shifted_step(duration):
        nonlocal injected
        original(duration)
        if not injected:
            env.data.qpos[0] += 0.02
            mujoco.mj_forward(env.model, env.data)
            injected = True

    monkeypatch.setattr(robot, "step", shifted_step)
    result = execute_pose_path(robot, executor, "pregrasp", [robot.servo.positions()])
    assert not result.success and result.message == "base_drift_during_arm"
    assert robot.servo.stopped


def test_zero_actor_cannot_complete_a_displaced_hand_path(pose_executor):
    _, robot, executor, _ = pose_executor
    q = robot.servo.positions()
    result = execute_pose_path(robot, executor, "pregrasp", [q, q + [0, -0.15, 0, 0, 0]])
    assert not result.success
    assert result.message in {"cartesian_tracking_timeout", "cartesian_path_tracking_envelope"}
    assert robot.policy_calls > 0 and robot.servo.stopped and robot.stop_confirmed


def test_cartesian_mode_never_selects_a_teacher_implicitly(pose_executor):
    env, _, _, _ = pose_executor
    with pytest.raises(ValueError, match="explicit simulator policy"):
        MarsSimRobot(env, arm_execution="cartesian")


def test_executor_can_close_gripper_during_terminal_arm_base_hold(pose_executor):
    _, robot, _, _ = pose_executor
    robot.servo.hold_position()
    pose, target = robot.servo.ee_pose().copy(), robot.servo.target.copy()
    opening = robot.servo.grip_target
    assert robot.close_gripper(blocking=True)
    assert robot.servo.grip_target < opening - 0.1
    assert robot.servo.holding and not robot.failure and robot.policy_calls == 0
    np.testing.assert_array_equal(robot.servo.target, target)
    assert pose_error(robot.servo.ee_pose(), pose)[0] < 0.003
    # The same request cannot move through a latched fault or authorize opening.
    robot.stop_and_confirm("test_fault")
    grip = robot.servo.grip_target
    assert not robot.close_gripper(blocking=True)
    assert robot.servo.grip_target == grip and not robot.servo.release_authorized


def test_pose_endpoint_uses_actual_tamp_goal_without_compounding_ik_tolerance(pose_executor):
    env, robot, executor, _ = pose_executor
    q = robot.servo.positions()
    goal_q = q + [0, -0.01, 0, 0, 0]
    goal = robot.servo.ee_pose().copy()
    witness_pose = TaskFrameFK(env.model, env.data, robot.servo)(robot.servo.base_pose(), goal_q)
    direction = witness_pose[:3] - goal[:3]
    goal[:3] += 0.005 * direction / np.linalg.norm(direction)
    goal[3:] = witness_pose[3:]
    distance, angle = pose_error(witness_pose, goal)
    assert distance < executor.position_tolerance_m and angle < executor.orientation_tolerance_rad
    result = execute_pose_path(robot, executor, "pregrasp", [q, goal_q], goal_pose=goal)
    assert not result.success and result.message == "cartesian_tracking_timeout"
    np.testing.assert_allclose(robot.commands[-1]["task_pose_trajectory"]["hand"][-1], goal)
    assert robot.stop_confirmed


def test_retention_checks_and_events_use_a_fresh_measured_snapshot(pose_executor, monkeypatch):
    _, robot, executor, adapter = pose_executor
    times = []

    def retained():
        times.append((float(executor.data.time), float(robot.data.time)))
        return True

    monkeypatch.setattr(executor, "payload_retained", retained)
    result = execute_pose_path(robot, executor, "pregrasp", [robot.servo.positions()])
    assert result.success and len(times) > 2
    assert all(a == pytest.approx(b) for a, b in times)
    assert adapter.events[-1]["sim_time"] == pytest.approx(robot.data.time)


@pytest.mark.parametrize("reason", ["wheel_lift", "unsafe_contact", "retention_lost"])
def test_transport_hold_failure_preserves_the_first_safety_cause(pose_executor, monkeypatch, reason):
    _, robot, _, _ = pose_executor
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    base, hand = robot.servo.base_pose(), robot.servo.ee_pose()
    path = TaskPoseTrajectory(
        [0, 0.05], [base, base], [hand, hand], [robot.servo.grip_reference] * 2, base_mask=np.ones(3, dtype=bool)
    )

    def unsafe_hold(*args):
        robot.failure = reason
        robot.servo.stop()
        return False

    monkeypatch.setattr(robot, "_verify_quiet_hold", unsafe_hold)
    result = robot.follow_trajectory(path, loaded=False)
    assert not result.success and result.message == reason and robot.failure == reason
    from emet.controller.task.tamp.api import failure_code

    assert failure_code("transport_failed:" + reason) == reason


def test_explicit_analytic_reference_is_not_reported_as_checkpoint_inference(pose_executor):
    _, robot, _, adapter = pose_executor

    class Teacher:
        controller_name = "analytic_test_reference"
        is_reference_controller = True

    robot.policy = Teacher()
    robot.policy_calls = 7
    data = adapter._data()
    assert data["controller"] == "analytic_test_reference"
    assert data["policy_calls"] == 0 and data["reference_controller_calls"] == 7


def test_cartesian_transport_brakes_before_rebasing_position_hold(pose_executor, monkeypatch):
    _, robot, _, _ = pose_executor
    from emet.rl.manipulation.safety import whole_body_stopped
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    base, hand = robot.servo.base_pose(), robot.servo.ee_pose()
    path = TaskPoseTrajectory(
        [0, 0.05], [base, base], [hand, hand], [robot.servo.grip_reference] * 2, base_mask=np.ones(3, dtype=bool)
    )
    robot.data.qvel[robot.servo.base_dof] = 0.008
    original = robot.servo.hold_position
    entered = []

    def hold():
        entered.append(float(robot.data.time))
        assert whole_body_stopped(robot.servo)
        assert not robot.servo.base_bias_compensation_enabled
        original()

    monkeypatch.setattr(robot.servo, "hold_position", hold)
    result = robot.follow_trajectory(path, loaded=False)
    assert result.success and entered


@pytest.mark.parametrize(
    "reason,code",
    [
        ("cartesian_tracking_timeout", "tracking_timeout"),
        ("cartesian_path_tracking_envelope", "tracking_timeout"),
        ("arm_configuration_collision", "collision"),
        ("arm_joint_margin", "joint_bounds"),
        ("cartesian_witness_goal_mismatch", "invalid_plan"),
    ],
)
def test_pose_execution_failures_keep_public_failure_categories(reason, code):
    from emet.controller.task.tamp.api import failure_code

    assert failure_code("pregrasp_failed:" + reason) == code


@pytest.mark.parametrize(
    "stage,phase_index,retain", [("pregrasp", 1, False), ("carry", 3, True), ("release", 5, False)]
)
def test_cartesian_stage_restores_actor_phase_retention_and_grip_intent(pose_executor, stage, phase_index, retain):
    from emet.rl.manipulation.observation_layout import history_layout
    from emet.rl.manipulation.profiles import MARS

    _, robot, executor, _ = pose_executor
    robot.servo.retention_context_enabled = True
    robot.servo.history = None
    robot.servo.set_reference(robot.servo.positions(), phase="release", grip=MARS.gripper_closed, release=True)
    robot.phase = stage
    seen = []

    def actor(obs):
        seen.append(obs["history"][-1].copy())
        return np.zeros(8)

    robot.policy = actor
    result = execute_pose_path(robot, executor, "pregrasp", [robot.servo.positions()])
    assert result.success and seen
    fields, _ = history_layout(MARS, retention_context=True)
    assert np.argmax(seen[0][fields["phase"]]) == phase_index
    assert bool(seen[0][fields["retain_object_command"]][0]) == retain
    assert robot.servo.release_authorized == (stage == "release")
    if stage == "release":
        assert robot.commands[-1]["task_pose_trajectory"]["gripper"][-1] == MARS.gripper_open
    if stage == "pregrasp":
        assert robot.servo.grip_reference == pytest.approx(MARS.gripper_open, abs=0.001)


def test_cartesian_path_cannot_authorize_release(pose_executor):
    _, robot, executor, _ = pose_executor
    robot.phase = "release"
    assert not robot.servo.release_authorized
    result = execute_pose_path(robot, executor, "retreat", [robot.servo.positions()])
    assert not result.success and result.message == "release_authorization_required"
    assert robot.servo.stopped and robot.policy_calls == 0


def test_reference_clock_advances_smoothly_outside_endpoint_precision_inside_the_existing_tube():
    from emet.controller.task.tamp.mars_pose_executor import lag_scaled_rate

    def rate(p, a):
        return lag_scaled_rate(p, a, position_scale_m=0.003, orientation_scale_rad=0.02)

    assert rate(0, 0) == 1 and rate(0.003, 0.02) == 1
    assert 0.9 < rate(0.0031, 0.0201) < 1
    assert rate(0.008, 0.06) == pytest.approx(1 / 3)
    assert rate(0.00301, 0.02) > 0.99
    with pytest.raises(ValueError):
        rate(float("nan"), 0)


def test_invalid_retiming_contract_stops_before_motion(pose_executor):
    _, robot, executor, _ = pose_executor
    executor.position_tolerance_m = 0
    result = execute_pose_path(robot, executor, "pregrasp", [robot.servo.positions()])
    assert not result.success and result.message == "invalid_cartesian_tolerance"
    assert robot.policy_calls == 0 and robot.servo.stopped


@pytest.mark.parametrize("pose_executor", ["jerk_limited"], indirect=True)
def test_continuous_fk_reference_executes_without_vertex_stops_or_teacher_fallback(pose_executor, monkeypatch):
    _, robot, executor, adapter = pose_executor
    q = robot.servo.positions()
    goal = q + [0, -0.005, 0.01, -0.005, 0]
    monkeypatch.setattr(robot, "_joint_teacher_action", lambda: pytest.fail("teacher fallback"))
    result = execute_pose_path(robot, executor, "pregrasp", np.linspace(q, goal, 41))
    assert result.success, result
    record = robot.commands[-1]["task_pose_trajectory"]
    assert record["format"] == "task_pose_fk_reference_v2"
    assert record["arm_derivative_certificate"]["retained_vertices"] == 2
    assert robot.policy_calls > 0 and not robot.failure
    assert robot.reference_phase.progress == pytest.approx(robot.servo.task_trajectory.end)
    assert any(e.get("reference_clock") == "filtered_lag_scaling" for e in adapter.events)


@pytest.mark.parametrize("pose_executor", ["jerk_limited"], indirect=True)
def test_reference_motion_logging_does_not_follow_a_stale_arm_phase_into_transport(pose_executor):
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    _, robot, executor, _ = pose_executor
    robot.record_observations = True
    q = robot.servo.positions().copy()
    result = execute_pose_path(robot, executor, "pregrasp", [q, q + [0, -0.005, 0.01, -0.005, 0]])
    assert result.success, result
    assert "reference_motion" in robot.trace[-1]
    base, hand = robot.servo.base_pose().copy(), robot.servo.ee_pose().copy()
    path = TaskPoseTrajectory(
        [0, 1], [base, base], [hand, hand], [robot.servo.grip_reference] * 2, base_mask=[True] * 3
    )
    robot.servo.set_task_trajectory(path)
    robot.step(robot.env.profile.dt)
    assert not robot.failure
    assert "reference_motion" not in robot.trace[-1]


def test_transport_reserves_quiet_verification_even_at_last_motion_tick(pose_executor, monkeypatch):
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    _, robot, _, _ = pose_executor
    base, hand = robot.servo.base_pose(), robot.servo.ee_pose()
    path = TaskPoseTrajectory(
        [0, 0.05], [base, base], [hand, hand], [robot.servo.grip_reference] * 2, base_mask=[True] * 3
    )
    started = float(robot.data.time)
    robot.data.qvel[:] = 0

    def step(dt):
        robot.data.time += dt

    monkeypatch.setattr(robot, "step", step)
    monkeypatch.setattr(
        robot.servo, "velocities", lambda: np.full(5, 0.1) if robot.data.time < started + 4.1 else np.zeros(5)
    )
    checked = []

    def verify(required, timeout):
        checked.append(timeout)
        assert timeout >= required
        return True

    monkeypatch.setattr(robot, "_verify_quiet_hold", verify)
    result = robot.follow_trajectory(path, loaded=False)
    assert result.success and checked
