"""An uncertain manipulation must not be followed by another grasp or navigation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.controller.task.tamp import clutter_chain, task_search
from emet.controller.task.tamp.task_search import TaskPlan
from emet.memory.graph_eqa import sim_ground_truth_graph


@pytest.mark.parametrize("failure", ["grasp", "place", "verification"])
def test_uncertain_execution_stops_chain_and_navigation(monkeypatch, failure):
    placements = {name: {"pos": [index, 0, 0], "cat": name}
                  for index, name in enumerate(("one", "two", "bin"))}
    monkeypatch.setattr(sim_ground_truth_graph, "read_sim_object_placements", lambda _: placements)
    plan = TaskPlan([], "one", "bin", success=True)
    plan.grasp_poses = []
    planner = Mock(return_value=plan)
    monkeypatch.setattr(task_search, "plan_pick_place_mcts", planner)

    def execute(*args, **kwargs):
        plan.success = failure == "verification"
        plan.failed_op = None if plan.success else failure
        plan.completed_ops = ["approach", "grasp"] if failure != "grasp" else ["approach"]
        return plan

    execution = Mock(side_effect=execute)
    navigation = Mock()
    monkeypatch.setattr(task_search, "execute_task_plan", execution)
    monkeypatch.setattr(clutter_chain, "nav_to_landmark_if_clear", navigation)
    result = clutter_chain.plan_clear_clutter(
        SimpleNamespace(get_emet_session=lambda: {}),
        objects=[{"object_query": name, "object_gt_body": name} for name in ("one", "two")],
        mode="nav_goal", goal_xy=[3, 0], bin_query="bin", bin_body_override="bin",
        manip_mode="sim",
    )
    assert execution.call_count == planner.call_count == 1
    navigation.assert_not_called()
    assert result["n_objects"] == 2
    assert result["failed_bodies"] == ["one"]
    assert result["unattempted_bodies"] == ["two"]
    assert result["recovery_required"] is True
    assert result["task_success"] is False
    assert result["nav_path_open"] is False
