"""An admitted unloaded handle cannot plan or execute closure/loaded motion."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult, PhysicalPickPlaceExecutor
from emet.controller.task.tamp.native_adapter import NativeTableAdapter


@pytest.mark.parametrize(
    "scope,phases",
    [("pregrasp", ("pregrasp",)), ("grasp_pose", ("pregrasp", "grasp")), (None, ("pregrasp", "grasp", "lift"))],
)
def test_measured_replanning_covers_exact_requested_scope(scope, phases):
    ex = object.__new__(PhysicalPickPlaceExecutor)
    ex.input_mode = "privileged"
    ex.grasp_paths = [[np.zeros(1)]] * len(phases)
    ex.grasp_targets = [(np.zeros(3), np.eye(3))] * len(phases)
    ex.grasp_target_options = []
    ex.gripper_open_configuration = ()
    ex.robot = SimpleNamespace(open_gripper=Mock(return_value=True))
    ex.event = Mock()
    calls = []

    def replan(actual, targets, **options):
        calls.append(tuple(actual))
        assert len(targets) == len(actual)
        if "lift" in actual:
            return None, PhysicalMotionResult(False, "arm_path_static_support_margin", "lift")
        return ex.grasp_paths, None

    ex._replan_at_measured_pose = replan
    ex._execute_path = Mock(return_value=PhysicalMotionResult(True, "ok"))
    ex._close_gripper_until_still = Mock(side_effect=AssertionError("closure forbidden"))
    result = ex.grasp_only("", object_gt_body="object", stop_after=scope)
    assert result.success is (scope is not None)
    assert calls == [phases] * (2 if scope is not None else 1)
    assert (
        [c.args[0] for c in ex._execute_path.call_args_list] == list(phases[:2])
        if scope
        else not ex._execute_path.called
    )
    ex._close_gripper_until_still.assert_not_called()


def test_prefix_witness_cannot_be_executed_as_a_full_grasp():
    ex = object.__new__(PhysicalPickPlaceExecutor)
    ex.grasp_paths = [[], []]
    ex.grasp_targets = [None, None]
    assert not ex.grasp_only("", object_gt_body="object").success


@pytest.mark.parametrize("scope", ["", "lift", "close", None, True, [], 1])
def test_invalid_public_scope_rejected_before_state_access(scope):
    adapter = object.__new__(NativeTableAdapter)
    adapter._failure = lambda tool, reason: reason
    assert adapter.plan(execution_scope=scope) == "invalid_execution_scope"


def test_prefix_requires_measured_hold_and_stop_interface_before_state_access():
    adapter = object.__new__(NativeTableAdapter)
    adapter.robot = SimpleNamespace()
    adapter._failure = lambda tool, reason: reason
    assert adapter.plan(execution_scope="grasp_pose") == "execution_scope_unsupported"


@pytest.mark.sim
def test_prefix_plan_preserves_live_physics_and_has_no_downstream_witness(monkeypatch):
    import json

    from emet.agent.tools import get_tools
    from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
    from emet.controller.task.tamp.native_adapter import adapter_for
    from emet.eval.mars_tamp_control import fixture
    from emet.rl.manipulation.whole_body_training import load_development

    development, _ = load_development()
    env, admission = fixture(development["train_cases"][2], wheel_feedback_current_velocity=True)
    try:
        assert admission["accepted"]
        from emet.rl.manipulation.command_runtime import task_pose_teacher_action

        def teacher(_observation):
            return task_pose_teacher_action(env.control)

        teacher.is_reference_controller = True
        robot = MarsSimRobot(env, policy=teacher, arm_execution="cartesian", arm_reference="s_curve")
        context = {
            "robot": robot,
            "manip_mode": "physical",
            "tamp_inputs": "privileged",
            "_native_tamp_model": env.model,
        }
        tools = {t.name: t for t in get_tools(context)}
        tasks = json.loads(tools["scene_tasks"].func(object_filter="blue"))["data"]["tasks"]
        task = next(t for t in tasks if t["receptacle_name"] == "destination support")
        adapter = adapter_for(context, robot)
        monkeypatch.setattr(
            adapter, "_prepare_transport", Mock(side_effect=AssertionError("loaded planning forbidden"))
        )
        before = env.data.qpos.copy(), env.data.qvel.copy(), float(env.data.time)
        result = json.loads(tools["plan_pick_place"].func(task_ref=task["task_ref"], execution_scope="grasp_pose"))
        assert result["status"] == "ok", result
        plan = adapter.plans[result["data"]["plan_ref"]]
        assert plan.execution_scope == "grasp_pose" and plan.goal is None and plan.transport is None
        assert len(plan.executor.grasp_paths) == len(plan.executor.grasp_targets) == 2
        assert not plan.executor.place_paths and not plan.executor.place_targets
        assert result["data"]["full_task_admitted"] is False
        debug = adapter.pose_references(result["data"]["plan_ref"])
        assert [x["stage"] for x in debug["arm_targets"]] == ["pregrasp", "grasp"]
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.qvel, before[1])
        assert env.data.time == before[2] and not robot.commands
        monkeypatch.setattr(robot, "close_gripper", Mock(side_effect=AssertionError("closure forbidden")))
        with robot.stepped_clock():
            executed = json.loads(tools["execute_pick_place_plan"].func(plan_ref=result["data"]["plan_ref"]))
        assert executed["status"] == "ok", executed
        assert executed["data"]["stop_confirmed"] is True
        assert executed["data"]["full_pick_place_success"] is False
        assert executed["data"]["completed_ops"] == ["grasp_pose"]
        assert not any(c.get("gripper") == "close" or c.get("phase") == "lift" for c in robot.commands)
        assert (
            json.loads(tools["execute_pick_place_plan"].func(plan_ref=result["data"]["plan_ref"]))["code"]
            == "unknown_plan"
        )
    finally:
        env.close()


def test_prefix_scope_survives_primary_and_fallback_grasp_search():
    adapter = object.__new__(NativeTableAdapter)
    adapter.pregrasp_fallback_offset_m = np.array([0, 0.02, 0.03])
    adapter.max_planning_ik_calls = 100
    adapter.planning_ik_calls = 0
    adapter.fallback_ik_reserve_fraction = 0.25
    adapter._strategy_ik_limit = None
    calls = []

    def candidates(*args, **kwargs):
        calls.append(kwargs)
        if "approach_offset" in kwargs:
            yield "fallback", "paths"

    adapter._grasp_candidates = candidates
    result = list(
        adapter._grasp_options(None, None, None, None, None, None, execution_scope="grasp_pose", aligned_only=True)
    )
    assert result == [("fallback", "paths")]
    assert len(calls) == 2
    assert all(c["execution_scope"] == "grasp_pose" and c["aligned_only"] for c in calls)
    assert "approach_offset" not in calls[0] and "approach_offset" in calls[1]
    assert adapter._strategy_ik_limit is None
