"""Public diagnostics retain placement causes without exposing private identifiers."""
import json
from types import SimpleNamespace

import pytest

from emet.controller.task.tamp.api import failure_code, placement_evidence
from emet.controller.task.tamp.task_search import TaskPlan, TaskPlanStep, execute_task_plan


@pytest.mark.parametrize("code", ["no_collision_free_placement", "placement_invalidated",
                                 "placement_geometry_unavailable", "release_execution_error"])
def test_placement_failure_code_survives_wrapping(code):
    assert failure_code("place_failed:" + code) == code


def test_private_search_details_are_not_public():
    executor = SimpleNamespace(last_placement_search=SimpleNamespace(
        paths=[], geometry_source="ground_truth", collision_scope="sampled_arm_and_payload;base_endpoint_only",
        rejections={"private_body_name": 1, "pose_ik_failed": 2,
                    "max_iter reached with nodes fwd = 900": 3}),
        last_release_evidence={"detach_command": "completed", "placement_verified": True})
    result = placement_evidence(executor)
    assert "private_body_name" not in json.dumps(result)
    assert result["placement_search"]["rejections"] == {
        "other": 1, "pose_ik_failed": 2, "rrt_budget_exhausted": 3}
    assert result["release"]["held_state"] == "unknown"


@pytest.mark.parametrize("detach", ["not_attempted", "unknown", "completed"])
def test_failed_place_retains_current_operation_release_evidence(detach):
    class Executor:
        def begin_operation(self, operation_id):
            self.operation_id = operation_id

        def place_only(self, *args, **kwargs):
            self.last_release_evidence = {"detach_command": detach, "placement_verified": None}
            return SimpleNamespace(success=False, message="retract_execution_error")

    plan = TaskPlan([TaskPlanStep("place", {"receptacle_query": "table"})], "private", "table")
    result = execute_task_plan(None, plan, executor=Executor(), grasp_poses=[])
    assert not result.success
    assert result.completed_ops == []
    assert result.measurements[0]["release"]["detach_command"] == detach
    assert result.measurements[0]["release"]["held_state"] == "unknown"
