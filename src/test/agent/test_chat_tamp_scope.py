"""Extracted CHAT handlers retain the simulator's explicit unloaded scope."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.agent.chat_tamp import build_tamp_handlers
from emet.controller.task.tamp.api import response


@pytest.mark.parametrize("scope", ["full_task", "pregrasp", "grasp_pose"])
def test_extracted_handler_preserves_explicit_native_scope_without_changing_default(monkeypatch, scope):
    from emet.controller.task.tamp import native_adapter

    adapter = SimpleNamespace(plan=Mock(return_value=response("plan_pick_place", data={"plan_ref": "opaque"})))
    monkeypatch.setattr(native_adapter, "adapter_for", lambda context, robot: adapter)
    robot = SimpleNamespace(get_emet_session=lambda: {"is_simulation": True})
    context = {"robot": robot, "manip_mode": "physical", "tamp_inputs": "privileged"}
    result = json.loads(build_tamp_handlers(context)["plan_pick_place"](task_ref="task", execution_scope=scope))
    assert result["status"] == "ok"
    expected = {} if scope == "full_task" else {"execution_scope": scope}
    adapter.plan.assert_called_once_with("task", "", "", None, **expected)


def test_unloaded_scope_stays_unavailable_for_the_nonphysical_planner():
    context = {"manip_mode": "auto"}
    result = json.loads(build_tamp_handlers(context)["plan_pick_place"](execution_scope="pregrasp"))
    assert result["code"] == "execution_scope_unsupported"
