"""Prompt/parser/dispatcher contracts, not real-model or observed task results."""

import json

from emet.agent.tools import Tool
from emet.eval.mars_tamp_agent import AgentToolDriver


def registry():
    return {
        n: Tool(n, n, {"type": "object", "properties": {}}, lambda: None, returns_info=True)
        for n in ("scene_tasks", "plan_pick_place", "execute_pick_place_plan")
    }


def test_agent_uses_real_parser_dispatcher_and_stops_after_partial_execution():
    names = ["scene_tasks", "plan_pick_place", "execute_pick_place_plan"]
    responses = [json.dumps({"tool_calls": [{"name": n, "arguments": {}}]}) for n in names]
    calls = []

    def infer(system, prompt):
        assert "blue cube" in prompt
        assert "tool" in system.lower()
        return responses.pop(0)

    def call(name, **kwargs):
        calls.append(name)
        return {"status": "partial" if name == "execute_pick_place_plan" else "ok", "code": "diagnostic", "data": {}}

    driver = AgentToolDriver(infer, input_mode="privileged")
    result = driver(call, {"placement_region": [[0, 0], [1, 1]]}, registry())
    assert result["status"] == "partial"
    assert driver.terminal == "partial_execution_requires_recovery"
    assert calls == names and len(driver.trace) == 3


def test_invalid_multiaction_reply_never_dispatches():
    response = json.dumps(
        {"tool_calls": [{"name": "scene_tasks", "arguments": {}}, {"name": "execute_pick_place_plan", "arguments": {}}]}
    )
    driver = AgentToolDriver(lambda system, prompt: response, input_mode="privileged", max_turns=2)
    calls = []
    driver(lambda *args, **kwargs: calls.append(args), {"placement_region": [[0, 0], [1, 1]]}, registry())
    assert not calls
    assert driver.terminal == "turn_budget_exhausted"
    assert len(driver.trace) == 2


def test_missing_observed_provider_blocks_before_model_call_without_gt_fallback():
    def infer(*args):
        raise AssertionError("model must not run without observed capabilities")

    calls = []

    def call(name, **kwargs):
        calls.append(name)
        return {"status": "error", "code": "observed_scene_unavailable", "data": {}}

    driver = AgentToolDriver(infer, input_mode="observed")
    result = driver(call, {"placement_region": [[0, 0], [1, 1]]}, registry())
    assert result["code"] == "observed_scene_unavailable"
    assert calls == ["scene_tasks"]
    assert driver.trace[0]["model_called"] is False
