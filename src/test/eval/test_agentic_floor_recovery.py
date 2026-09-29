# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Physical find uses the same stationary recovery primitive as chat."""

from types import SimpleNamespace
from unittest.mock import Mock

from emet.agent.skills.eqa import bind_eqa_episode_tools
from emet.agent.skills.specs import CHAT_SKILL_SPECS
from emet.controller.dynamem.look import supports_floor_observation
from emet.memory.graph_eqa.agentic import tools
from emet.memory.graph_eqa.agentic.action import _action_signature
from emet.memory.graph_eqa.agentic.executor_init import _handle_observe_floor


def executor(*, physical=True):
    robot = SimpleNamespace(head_to=Mock(), get_pan_tilt=Mock(), get_observation=Mock())
    if physical:
        robot._seq_id = 1
    return SimpleNamespace(agent=SimpleNamespace(robot=robot), mode="explore", handle_tool=Mock())


def test_floor_tool_capability_and_canonical_schema():
    ex = executor()
    tool = next(t for t in bind_eqa_episode_tools(ex) if t.name == "observe_floor")
    spec = next(s for s in CHAT_SKILL_SPECS if s.name == "observe_floor")
    assert tool.parameters == spec.parameters
    assert tool.description == spec.description
    assert not supports_floor_observation(executor(physical=False).agent)
    assert "observe_floor" not in {t.name for t in bind_eqa_episode_tools(executor(physical=False))}


def test_floor_handler_reuses_capture_and_preserves_evidence(monkeypatch):
    ex = executor()
    ex._append_trace = Mock()
    result = {"status": "observed", "footprint_after": {"unobserved_cells": 9}}
    capture = Mock(return_value=result)
    monkeypatch.setattr("emet.controller.dynamem.look.observe_floor", capture)
    assert _handle_observe_floor(ex, {"pan_rad": 0.2, "tilt_rad": -1.2}) == result
    capture.assert_called_once_with(ex.agent, pan_rad=0.2, tilt_rad=-1.2)
    assert ex._last_floor_observation == result
    ex._append_trace.assert_called_once_with({"tool": "observe_floor", **result})


def test_recovery_state_only_for_supported_unknown_footprint(monkeypatch):
    monkeypatch.setattr(tools, "_build_state_message", lambda ex: "existing state")
    feedback = {"outcome": "rejected_swept_footprint:unobserved_footprint", "missing_cells": 9}
    monkeypatch.setattr("emet.agent.tools.navigation_feedback", lambda agent: feedback)
    assert tools.build_state_message(executor(physical=False)) == "existing state"
    ex = executor()
    ex._last_floor_observation = {"footprint_after": {"unobserved_cells": 9}}
    text = tools.build_state_message(ex)
    assert "observe_floor" in text and '"missing_cells": 9' in text
    assert "Last floor observation" in text and "Replan before any motion" in text
    feedback["outcome"] = "arrived"
    assert tools.build_state_message(ex) == "existing state"


def test_floor_views_have_distinct_action_signatures():
    ex = SimpleNamespace(query_text="find mug", _robot_xyt_world=lambda: [0, 0, 0])
    first = _action_signature(ex, "observe_floor", {"tilt_rad": -1.0})
    second = _action_signature(ex, "observe_floor", {"tilt_rad": -1.3})
    assert first != second
