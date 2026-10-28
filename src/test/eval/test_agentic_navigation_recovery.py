# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.core.navigation_result import UNOBSERVED_FOOTPRINT_STATUS
from emet.memory.graph_eqa.agentic import navigation_recovery as recovery

UNKNOWN = UNOBSERVED_FOOTPRINT_STATUS


def setup(monkeypatch, codes, *, cleared=True, supported=True):
    monkeypatch.setattr(recovery, "supports_floor_observation", lambda agent: supported)
    ex = SimpleNamespace(agent=SimpleNamespace(), _append_trace=Mock())
    ex.handle_tool = Mock(
        return_value={
            "ok": True,
            "status": "floor_observed",
            "observation": {"footprint_after": {"pose_valid": cleared}},
        }
    )
    outcomes = []

    def navigate():
        code = codes[len(outcomes)]
        ex.agent._last_nav_attempt = SimpleNamespace(
            success=code == "ok",
            finished=code == "ok",
            status_code=code,
            note=code,
        )
        outcome = object()
        outcomes.append(outcome)
        return outcome

    return ex, Mock(side_effect=navigate), outcomes


def test_two_blockers_replan_within_one_decision(monkeypatch):
    ex, nav, outcomes = setup(monkeypatch, [UNKNOWN, UNKNOWN, "ok"])
    assert recovery.navigate_with_floor_recovery(ex, nav) is outcomes[-1]
    assert nav.call_count == 3
    assert ex.handle_tool.call_count == 2
    ex.handle_tool.assert_called_with(
        "observe_floor", {"target_blocker": True, "max_observations": 1}, validate_arguments=False
    )
    rows = [call.args[0] for call in ex._append_trace.call_args_list]
    assert all(row["resume_same_goal"] for row in rows if row["event"] == "navigation_floor_recovery")
    assert len([row for row in rows if row["event"] == "navigation_attempt_timing"]) == 3
    assert all(row["wall_s"] >= 0 for row in rows)


def test_floor_recovery_through_public_executor_keeps_internal_budget_private(monkeypatch):
    from emet.memory.graph_eqa.agentic import executor_init
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = Mock()
    agent.robot = None
    ex = AgenticEQAExecutor(agent, "Where is the bowl?", router=False)
    monkeypatch.setattr(recovery, "supports_floor_observation", lambda agent: True)
    dispatch = Mock(
        return_value={"ok": True, "observation": {"footprint_after": {"pose_valid": True}}}
    )
    monkeypatch.setattr(executor_init, "_dispatch_tool", dispatch)
    attempts = []

    def navigate():
        code = UNKNOWN if not attempts else "ok"
        agent._last_nav_attempt = SimpleNamespace(
            status_code=code, note=code, success=code == "ok", finished=code == "ok"
        )
        attempts.append(object())
        return attempts[-1]

    assert recovery.navigate_with_floor_recovery(ex, navigate) is attempts[-1]
    assert len(attempts) == 2
    dispatch.assert_called_once_with(ex, "observe_floor", {"target_blocker": True, "max_observations": 1})
    assert "observe_floor" in ex._conversation.messages[-1]["content"]

    # A model cannot request the same internal bypass/budget in its JSON args.
    dispatch.reset_mock()
    rejected = ex.handle_tool("observe_floor", {"max_observations": 2, "validate_arguments": False})
    assert rejected["status"] == "invalid_tool_arguments"
    dispatch.assert_not_called()


def test_real_sweep_rejection_triggers_floor_recovery(monkeypatch):
    from emet.controller.dynamem.navigation import _filter_unsafe_nav_traj

    producer = SimpleNamespace(
        space=SimpleNamespace(
            obstacle_map_mode="physical",
            is_valid=lambda pose: False,
            last_validity={"reason": "unobserved_footprint", "checked_pose": [0, 0, 0]},
        )
    )
    _, status, _ = _filter_unsafe_nav_traj(producer, [[1, 0, 0]], start_xyt=[0, 0, 0])
    ex, nav, outcomes = setup(monkeypatch, [status, "ok"])
    assert recovery.navigate_with_floor_recovery(ex, nav) is outcomes[-1]
    assert ex.handle_tool.call_count == 1
    assert producer._last_nav_sweep_failure["checked_pose"] == [0, 0, 0]


@pytest.mark.parametrize(
    "code,cleared,supported",
    [
        ("ok", True, True),
        ("rejected_swept_footprint:occupied_footprint", True, True),
        (UNKNOWN, False, True),
        (UNKNOWN, True, False),
    ],
)
def test_no_retry_without_cleared_unknown_footprint(monkeypatch, code, cleared, supported):
    ex, nav, outcomes = setup(monkeypatch, [code], cleared=cleared, supported=supported)
    assert recovery.navigate_with_floor_recovery(ex, nav) is outcomes[0]
    assert nav.call_count == 1


def test_recovery_is_bounded_even_when_each_checked_pose_clears(monkeypatch):
    ex, nav, outcomes = setup(monkeypatch, [UNKNOWN] * 3)
    assert recovery.navigate_with_floor_recovery(ex, nav) is outcomes[-1]
    assert nav.call_count == 3
    assert ex.handle_tool.call_count == 2


def test_stale_attempt_cannot_trigger_recovery(monkeypatch):
    ex, nav, outcomes = setup(monkeypatch, [UNKNOWN])
    nav()
    ex.handle_tool.reset_mock()
    unchanged = Mock(return_value=outcomes[0])
    recovery.navigate_with_floor_recovery(ex, unchanged)
    ex.handle_tool.assert_not_called()
