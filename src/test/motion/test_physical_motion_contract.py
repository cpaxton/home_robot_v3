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
