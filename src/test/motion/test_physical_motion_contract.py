"""Regressions for physical acceptance: no movement on missing evidence."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.task.tamp.task_search import TaskPlan, TaskPlanStep, execute_task_plan
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.mujoco_arm_ik import solve_pose_ik
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker
from emet.simulation.physical_execution import audit_physical_action


def model():
    return mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="robot"><joint name="yaw" type="hinge" axis="0 0 1" range="-180 180"/>
    <geom type="box" size=".1 .1 .1"/><body name="ee" pos=".4 0 0"><geom size=".04"/></body></body>
    <geom name="wall" type="box" pos=".4 0 0" size=".02 .02 .2"/>
    </worldbody></mujoco>""")


@pytest.mark.parametrize("result", [False, None, {}, SimpleNamespace(success=False, reason="blocked")])
def test_failed_approach_never_grasps_or_reuses_success(result):
    robot = SimpleNamespace(move_base_to=lambda *a, **kw: result)
    executor = SimpleNamespace(grasp_only=lambda *a, **kw: pytest.fail("grasp after failed approach"))
    plan = TaskPlan(
        [TaskPlanStep("approach", {"xyt": [0, 0, 0]}), TaskPlanStep("grasp")],
        "obj",
        None,
        success=True,
        completed_ops=["grasp"],
    )
    out = execute_task_plan(robot, plan, executor=executor, grasp_poses=[])
    assert not out.success and out.failed_op == "approach" and out.completed_ops == []


def test_physical_mode_does_not_fall_back_to_teleport():
    robot = SimpleNamespace(move_base_to=lambda *a, **kw: pytest.fail("must reject before actuation"))
    plan = TaskPlan([TaskPlanStep("approach", {"xyt": [0, 0, 0]})], "obj", None)
    assert (
        execute_task_plan(robot, plan, executor=None, grasp_poses=[], manip_mode="physical").message
        == "unsupported_physical_executor"
    )


@pytest.mark.parametrize(
    "action", [{"sim_attach_body": {}}, {"nav_teleport": True}, {"sim_set_body_pose": {}}, {"set_joint": True}]
)
def test_guard_rejects_before_dispatch_and_persists_audit(monkeypatch, tmp_path, action):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("EMET_PHYSICAL_EXECUTION", "1")
    monkeypatch.setenv("EMET_PHYSICAL_AUDIT", str(path))
    with pytest.raises(RuntimeError, match="forbidden_physical_actuation"):
        audit_physical_action(action, source="test")
    assert json.loads(path.read_text())["accepted"] is False


def test_implicit_teleport_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("EMET_PHYSICAL_EXECUTION", "1")
    monkeypatch.setenv("EMET_PHYSICAL_AUDIT", str(tmp_path / "audit"))
    with pytest.raises(RuntimeError, match="nav_teleport"):
        audit_physical_action({"xyt": [0, 0, 0]}, source="test", implicit_teleport=True)


def test_near_goal_still_checks_collision():
    m = model()
    data = mujoco.MjData(m)
    checker = MujocoSceneCollisionChecker(m, robot_body="robot")
    result = plan_arm_joint_path(m, data, joint_names=["yaw"], q_start=[0], q_goal=[0.001], collision=checker)
    assert not result.success and result.reason == "invalid_start"


def test_arm_rejects_nonfinite():
    m = model()
    result = plan_arm_joint_path(m, mujoco.MjData(m), joint_names=["yaw"], q_start=[0], q_goal=[np.nan])
    assert not result.success and result.reason == "nonfinite_configuration"


def test_pose_ik_enforces_orientation_at_same_position():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="ee">
    <joint name="yaw" axis="0 0 1" range="-180 180"/><geom size=".1"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(m)
    target = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = solve_pose_ik(m, data, ee_body="ee", joint_names=["yaw"], target_pos=[0, 0, 0], target_rotation=target)
    assert result.success and result.orientation_error_rad < 0.1
    assert abs(data.qpos[0] - np.pi / 2) < 0.1
    impossible = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    result = solve_pose_ik(m, data, ee_body="ee", joint_names=["yaw"], target_pos=[0, 0, 0], target_rotation=impossible)
    assert not result.success and result.pos_error_m == 0


def test_attempt_clutter_cannot_use_latch():
    from emet.controller.task.tamp.clutter_chain import plan_clear_clutter

    result = plan_clear_clutter(None, objects=[], mode="cleanup", bin_query="bin", manip_mode="attempt")
    assert not result["task_success"] and result["status"] == "unsupported_capability"


def test_route_executor_rejects_unknown_before_actuation():
    from emet.motion.navigation_sweep import execute_measured_route

    robot = SimpleNamespace(move_base_to=lambda *a, **kw: pytest.fail("unknown route executed"))
    space = SimpleNamespace(is_valid=lambda pose: False, last_validity={"reason": "unobserved_footprint"})
    result = execute_measured_route(
        robot, goal=[1, 0, 0], measure=lambda: np.zeros(3), plan_route=lambda start, goal: [goal], space=space
    )
    assert not result.success and result.reason == "rejected_swept_footprint:unobserved_footprint"


def test_route_executor_bounds_failed_tracking():
    from emet.motion.navigation_sweep import execute_measured_route

    calls = []
    robot = SimpleNamespace(move_base_to=lambda *a, **kw: calls.append(a) or True, cancel_navigation=lambda: True)
    space = SimpleNamespace(is_valid=lambda pose: True)
    result = execute_measured_route(
        robot,
        goal=[1, 0, 0],
        measure=lambda: np.zeros(3),
        plan_route=lambda start, goal: [goal],
        space=space,
        max_replans=2,
    )
    assert not result.success and result.reason == "base_tracking_failed" and len(calls) == 3


def test_payload_geometry_is_checked_offline():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="robot"><joint name="x" type="slide" axis="1 0 0" range="-2 2"/><geom size=".02"/>
    <body name="ee" pos="0 .3 0"><geom size=".02"/></body></body>
    <body name="payload" pos="0 .4 0"><freejoint/><geom size=".1"/></body>
    <geom name="wall" pos=".5 .4 0" size=".03"/>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    checker = MujocoSceneCollisionChecker(m, robot_body="robot", allowed_pairs=[("ee", "payload")])
    checker.set_payload(m, d, "payload", "ee")
    d.qpos[0] = 0.5
    assert checker.configuration_collides(m, d)
    assert any("payload" in c["bodies"] for c in checker.last_contacts)
    # No payload: the thin arm clears the wall.
    checker.set_payload(m, d, None)
    assert not checker.configuration_collides(m, d)


def test_pose_ik_saturated_joint_does_not_freeze_other_axes():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="ee">
    <joint name="x" type="slide" axis="1 0 0" range="0 1"/>
    <joint name="z" type="slide" axis="0 0 1" range="0 1"/><geom size=".02"/>
    </body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    result = solve_pose_ik(
        m,
        d,
        ee_body="ee",
        joint_names=["x", "z"],
        target_pos=[-0.1, 0, 0.5],
        target_rotation=np.eye(3),
    )
    assert not result.success  # X is unreachable; Z should still converge.
    assert abs(d.qpos[1] - 0.5) < 0.01
    assert d.qpos[0] >= 0


@pytest.mark.parametrize("target,expected", [(.99, True), (.999, False)])
def test_pose_ik_reserves_tracking_margin_inside_joint_limits(target, expected):
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="ee">
      <joint name="extension" type="slide" axis="1 0 0" range="0 1"/>
      <geom size=".02"/></body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    d.qpos[0] = 1.  # Even an initially satisfied pose must respect the reserve.
    result = solve_pose_ik(
        m, d, ee_body="ee", joint_names=["extension"], target_pos=[target, 0, 0],
        target_rotation=np.eye(3), tol_m=.001, joint_limit_margins={"extension": .005},
    )
    assert result.success == expected
    assert .005 <= d.qpos[0] <= .995 + 1e-9


@pytest.mark.parametrize("start, goal, reason", [(0, 4, "invalid_goal"), (4, 0, "invalid_start")])
def test_arm_does_not_silently_replace_out_of_limit_endpoints(start, goal, reason):
    m = model()
    result = plan_arm_joint_path(m, mujoco.MjData(m), joint_names=["yaw"], q_start=[start], q_goal=[goal])
    assert not result.success and result.reason == reason


def test_navigation_posture_requires_collision_path_and_measured_tracking():
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = model()
    d = mujoco.MjData(m)
    calls = []
    executor = PhysicalPickPlaceExecutor(
        SimpleNamespace(),
        model=m,
        data=d,
        ee_body="ee",
        joint_names=["yaw"],
        collision=MujocoSceneCollisionChecker(m, robot_body="robot"),
        synchronize=lambda state: mujoco.mj_forward(m, state),
        command_joints=lambda q: calls.append(q) or True,
    )
    path, error = executor.plan_joint_target(np.array([1.0]))
    assert path is None and error == "arm_path_failed:invalid_start" and not calls
    # With a clear scene, a successful controller return still needs state progress.
    m.geom_pos[m.geom("wall").id] = [2, 0, 0]
    path, error = executor.plan_joint_target(np.array([0.2]))
    assert error is None
    d.qpos[0] = 0
    measured = d.qpos.copy()
    executor.synchronize = lambda state: state.qpos.__setitem__(slice(None), measured)
    result = executor.prepare_for_navigation(path)
    assert not result.success and result.message == "arm_tracking_failed"
    assert calls


@pytest.mark.parametrize("pose", [[0, 0], [np.nan, 0, 0]])
def test_invalid_measured_base_pose_never_reaches_planner_or_controller(pose):
    from emet.motion.navigation_sweep import execute_measured_route

    result = execute_measured_route(
        SimpleNamespace(),
        goal=[1, 0, 0],
        measure=lambda: np.asarray(pose),
        plan_route=lambda *args: pytest.fail("planning from invalid measurement"),
        space=SimpleNamespace(),
    )
    assert not result.success and result.reason == "invalid_measured_pose"
    assert result.position_residual_m is None


def test_invalid_base_feedback_cancels_motion_and_never_replans():
    from emet.motion.navigation_sweep import execute_measured_route

    measurements = iter([np.zeros(3), np.zeros(3), np.array([np.nan, 0, 0])])
    cancels = []
    robot = SimpleNamespace(
        move_base_to=lambda *args, **kwargs: True, cancel_navigation=lambda: cancels.append(True) or True
    )
    result = execute_measured_route(
        robot,
        goal=[1, 0, 0],
        measure=lambda: next(measurements),
        plan_route=lambda start, goal: [goal],
        space=SimpleNamespace(is_valid=lambda p: True),
    )
    assert not result.success and result.reason == "invalid_measured_pose" and cancels == [True]


def test_structured_false_controller_response_is_not_truthy_navigation_success():
    from emet.motion.navigation_sweep import execute_measured_route

    state = np.zeros(3)

    def move(waypoint, **kwargs):
        state[:] = waypoint
        return {"success": False}

    robot = SimpleNamespace(move_base_to=move, cancel_navigation=lambda: True)
    result = execute_measured_route(
        robot,
        goal=[1, 0, 0],
        measure=lambda: state.copy(),
        plan_route=lambda start, goal: [goal],
        space=SimpleNamespace(is_valid=lambda p: True),
        max_replans=0,
    )
    assert not result.success and result.reason == "base_tracking_failed"


def test_grasp_rechecks_pose_ik_after_measured_arrival_before_actuation():
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = model()
    d = mujoco.MjData(m)
    executor = PhysicalPickPlaceExecutor(
        SimpleNamespace(open_gripper=lambda **kwargs: pytest.fail("grasp despite infeasible measured pose")),
        model=m,
        data=d,
        ee_body="ee",
        joint_names=["yaw"],
        collision=MujocoSceneCollisionChecker(m, robot_body="robot"),
        synchronize=lambda state: mujoco.mj_kinematics(m, state),
        command_joints=lambda q: True,
    )
    executor.grasp_paths = [[np.zeros(1)]] * 3
    executor.grasp_targets = [([0, 0, 0], np.eye(3))] * 3
    executor.plan_pose = lambda *args: (None, "unreachable_after_arrival")
    result = executor.grasp_only("target", object_gt_body="target")
    assert not result.success and result.phase == "pregrasp"
    assert "unreachable_after_arrival" in result.message


def test_missing_payload_stops_arm_before_next_command():
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="robot"><joint name="x" type="slide" axis="1 0 0" range="-2 2"/><geom size=".02"/>
    <body name="ee" pos="0 .3 0"><geom size=".02"/></body></body>
    <body name="payload" pos="0 .4 0"><freejoint/><geom size=".02"/></body>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    checker = MujocoSceneCollisionChecker(m, robot_body="robot")
    executor = PhysicalPickPlaceExecutor(
        SimpleNamespace(),
        model=m,
        data=d,
        ee_body="ee",
        joint_names=["x"],
        collision=checker,
        synchronize=lambda state: mujoco.mj_kinematics(m, state),
        command_joints=lambda q: pytest.fail("motion with lost payload"),
    )
    checker.set_payload(m, d, "payload", "ee")
    executor.payload_body = "payload"
    assert executor.payload_retained()
    d.qpos[2] += 0.1
    result = executor._execute_path("lift", [np.zeros(1), np.array([0.2])])
    assert not result.success and result.message == "payload_not_retained"


def test_coupled_arm_can_raise_then_extend_around_blocked_diagonal():
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="ee"><joint name="z" type="slide" axis="0 0 1" range="0 1"/>
    <joint name="x1" type="slide" axis="1 0 0" range="0 .5"/>
    <joint name="x2" type="slide" axis="1 0 0" range="0 .5"/><geom size=".05"/></body>
    <geom pos=".5 0 .5" size=".15"/>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    checker = MujocoSceneCollisionChecker(m, robot_body="ee")
    executor = PhysicalPickPlaceExecutor(
        SimpleNamespace(),
        model=m,
        data=d,
        ee_body="ee",
        joint_names=["z", "x1", "x2"],
        collision=checker,
        synchronize=lambda state: None,
        command_joints=lambda q: True,
        coupled_groups=(("x1", "x2"),),
    )
    goal = np.array([1.0, 0.5, 0.5])
    path, error = executor.plan_joint_target(goal)
    assert error is None
    assert any(q[0] > 0.8 and q[1] + q[2] < 0.1 for q in path)
    for q in path:
        assert q[1] == q[2]
        d.qpos[:] = q
        assert not checker.configuration_collides(m, d)
    np.testing.assert_allclose(path[-1], goal)


def test_rigid_collision_only_pass_matches_full_position_contacts():
    m = model()
    d = mujoco.MjData(m)
    checker = MujocoSceneCollisionChecker(m, robot_body="robot")
    for yaw in np.linspace(-0.5, 0.5, 9):
        d.qpos[0] = yaw
        checker.configuration_collides(m, d)
        fast = sorted((tuple(c.geom), float(c.dist)) for c in d.contact[: d.ncon])
        mujoco.mj_fwdPosition(m, d)
        full = sorted((tuple(c.geom), float(c.dist)) for c in d.contact[: d.ncon])
        assert fast == full


def test_planned_lift_leaves_offline_payload_at_lifted_pose():
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="ee"><joint name="z" type="slide" axis="0 0 1" range="0 1"/><geom size=".02"/></body>
    <body name="payload" pos="0 .2 0"><freejoint/><geom size=".02"/></body>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    checker = MujocoSceneCollisionChecker(m, robot_body="ee")
    checker.set_payload(m, d, "payload", "ee")
    executor = PhysicalPickPlaceExecutor(
        SimpleNamespace(),
        model=m,
        data=d,
        ee_body="ee",
        joint_names=["z"],
        collision=checker,
        synchronize=lambda state: None,
        command_joints=lambda q: True,
    )
    path, error = executor.plan_joint_target(np.array([0.2]))
    assert error is None
    assert abs(d.body("payload").xpos[2] - 0.2) < 1e-9


@pytest.mark.parametrize("advance, expected", [(0.1, True), (0.0, False)])
def test_contact_closure_needs_settled_motor_and_advancing_feedback(monkeypatch, advance, expected):
    import emet.controller.manipulation.physical_pick_place as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    m = model()
    d = mujoco.MjData(m)
    commands = []
    robot = SimpleNamespace(
        close_gripper=lambda **kwargs: commands.append(kwargs) or True, get_gripper_position=lambda: 0.2
    )  # Contact stops short of the empty-jaw endpoint.

    def sync(state):
        state.time += advance

    executor = module.PhysicalPickPlaceExecutor(
        robot,
        model=m,
        data=d,
        ee_body="ee",
        joint_names=["yaw"],
        collision=MujocoSceneCollisionChecker(m, robot_body="robot"),
        synchronize=sync,
        command_joints=lambda q: True,
    )
    result = executor._close_gripper_until_still(timeout_s=1.0)
    assert result.success == expected
    assert commands == [{"blocking": False}]
    assert executor.payload_body is None  # Motor settling never asserts pickup or attachment.


def test_arm_acknowledgment_waits_for_measured_joint_convergence(monkeypatch):
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = model()
    m.geom_pos[m.geom('wall').id] = [2, 0, 0]
    d = mujoco.MjData(m)
    measured = np.array([0.0])
    commanded = []
    sleeps = []

    def settle(delay):
        sleeps.append(delay)
        measured[:] = commanded[-1]

    monkeypatch.setattr('emet.controller.manipulation.physical_pick_place.time.sleep', settle)
    executor = PhysicalPickPlaceExecutor(
        SimpleNamespace(), model=m, data=d, ee_body='ee', joint_names=['yaw'],
        collision=MujocoSceneCollisionChecker(m, robot_body='robot'),
        synchronize=lambda state: state.qpos.__setitem__(slice(None), measured),
        command_joints=lambda q: commanded.append(q.copy()) or True,
    )
    result = executor._execute_path('navigation_posture', [np.array([0.0]), np.array([0.2])])
    assert result.success and result.residual == pytest.approx(0)
    assert sleeps == [0.05]


def test_arm_stops_after_unplanned_base_motion():
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base"><freejoint/><geom size=".1"/>
        <body name="ee" pos="0 0 .4"><joint name="arm" type="slide" range="0 1"/>
          <geom size=".04"/></body></body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    measured = d.qpos.copy()
    commands = []

    def sync(state):
        state.qpos[:] = measured
        mujoco.mj_kinematics(m, state)

    def command(q):
        commands.append(q.copy())
        measured[-1] = q[0]
        measured[0] += .03  # Arm tracks, but an unintended base command moves it.
        return True

    executor = PhysicalPickPlaceExecutor(
        SimpleNamespace(), model=m, data=d, ee_body="ee", joint_names=["arm"],
        collision=MujocoSceneCollisionChecker(m, robot_body="base"),
        synchronize=sync, command_joints=command, base_body="base",
    )
    result = executor._execute_path("grasp", [np.array([0.]), np.array([.1]), np.array([.2])])
    assert not result.success and result.message == "base_drift_during_arm"
    assert len(commands) == 1


@pytest.mark.parametrize('yaw_error,success', [(0.02, True), (0.04, False)])
def test_precision_route_uses_policy_and_checks_measured_arrival(yaw_error, success):
    from emet.motion.navigation_sweep import execute_measured_route

    measured = np.zeros(3)
    policies = []

    def move(goal, **kwargs):
        policies.append(kwargs['navigation_policy'])
        measured[:] = goal
        measured[2] += yaw_error
        return True

    result = execute_measured_route(
        SimpleNamespace(move_base_to=move, cancel_navigation=lambda: True),
        goal=[.1, 0, 0], measure=lambda: measured.copy(), plan_route=lambda start, goal: [goal],
        space=SimpleNamespace(is_valid=lambda q: True), navigation_policy='precision',
        position_tolerance_m=.02, yaw_tolerance_rad=.03, max_replans=0,
    )
    assert result.success is success
    assert policies == ['precision']


def test_pose_ik_uses_orientation_slack_without_relaxing_tolerances():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="yaw" type="hinge" axis="0 0 1" range="-90 90"/>
      <geom size=".01"/><body name="ee" pos=".5 0 0"><geom size=".01"/></body>
    </body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    angle = .05
    result = solve_pose_ik(
        m, d, ee_body='ee', joint_names=['yaw'],
        target_pos=[.5 * np.cos(angle), .5 * np.sin(angle), 0], target_rotation=np.eye(3),
        tol_m=.01, tol_rad=.1,
    )
    assert result.success and result.pos_error_m <= .01 and result.orientation_error_rad <= .1


def test_measured_route_rejects_required_steering_turn_before_command():
    from emet.motion.navigation_sweep import differential_drive_waypoints, execute_measured_route

    space = SimpleNamespace(
        is_valid=lambda pose: abs(pose[2]) < .5,
        execution_waypoints=differential_drive_waypoints,
    )
    robot = SimpleNamespace(move_base_to=lambda *a, **kw: pytest.fail('uncertified steering executed'),
                            cancel_navigation=lambda: True)
    result = execute_measured_route(robot, goal=[0,.2,0], measure=lambda: np.zeros(3),
                                    plan_route=lambda *a: [[0,.2,0]], space=space, max_replans=0)
    assert not result.success
    assert result.reason.startswith('rejected_swept_footprint')


def test_grasp_alternatives_replan_all_phases_before_any_gripper_command():
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    m = model()
    d = mujoco.MjData(m)
    calls = []
    checker = SimpleNamespace(payload_body=None, payload_parent=None, payload_transform=None,
                              set_payload=lambda *args: None)
    robot = SimpleNamespace(open_gripper=lambda **kw: calls.append('open') or False)
    executor = PhysicalPickPlaceExecutor(robot, model=m, data=d, ee_body='ee', joint_names=['yaw'],
                                         collision=checker, synchronize=lambda data: None, command_joints=lambda q: True)
    nominal = [([-1,0,0], np.eye(3))] * 3
    alternative = [([1,0,0], np.eye(3))] * 3
    executor.grasp_paths = [[np.zeros(1)]] * 3
    executor.grasp_targets = nominal
    executor.grasp_target_options = [nominal, alternative]

    def plan(point, rotation):
        calls.append(float(point[0]))
        return (None, 'unreachable') if point[0] < 0 else ([np.zeros(1)], None)

    executor.plan_pose = plan
    result = executor.grasp_only('target', object_gt_body='target')
    assert result.message == 'gripper_open_failed'
    assert calls == [-1., 1., 1., 1., 'open']
    assert executor.grasp_targets is alternative
