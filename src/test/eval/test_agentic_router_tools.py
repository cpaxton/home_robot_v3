# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agentic EQA tool-calling router claims (T*)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from emet.controller.habitat_nav import NavOutcome

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from _agentic import (  # noqa: E402
    _require_agentic,
)


def test_T1_eqa_tools_valid_schemas():
    """T1: Tool.schema() round-trips as JSON; names unique; mode swaps submit/finish."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_tools import build_agentic_eqa_tools

    ex = MagicMock()
    ex.mode = "answer"
    tools = build_agentic_eqa_tools(ex)
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    assert "submit_answer" in names
    assert "finish" not in names
    for t in tools:
        s = t.schema()
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"].get("type") == "object"
        json.loads(json.dumps(s))

    ex.mode = "explore"
    names_x = [t.name for t in build_agentic_eqa_tools(ex)]
    assert "finish" in names_x
    assert "submit_answer" not in names_x


def test_T2_router_parses_and_dispatches_nav():
    """T2: parse_tool_calls_response reply dispatches navigate_to_obs."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    reply = '{"tool_calls": [{"name": "navigate_to_obs", "arguments": {"obs_id": 3}}], "message": ""}'
    client = MagicMock(return_value=reply)
    gm.eqa_client = client
    gm._navigation_waypoint_for_obs.return_value = np.array([1.0, 2.0, 0.0])
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)

    ex = AgenticEQAExecutor(agent, "Where is the sink?", max_rounds=3, max_nav_steps=2)
    calls, picked_by, meta = ex._route_tool_calls()
    assert picked_by == "vlm"
    assert meta["parse_ok"] is True
    assert calls == [("navigate_to_obs", {"obs_id": 3})]
    # System prompt is fixed and passed on the routing turn (prefix-cache contract).
    assert client.call_args.kwargs.get("system_prompt") == ex._system_prompt
    out = ex.handle_tool(*calls[0])
    assert out["ok"] is True
    agent.navigate_to_target_pose.assert_called_once()


def test_T3_multi_tool_reply_ordered_submit_gated():
    """T3: verify then submit executes in order; submit rejected before PRESENT."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    reply = (
        '{"tool_calls": [{"name": "verify_siglip", "arguments": {"phrase": "sink", "obs_id": 7}}, '
        '{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}'
    )
    gm.eqa_client = MagicMock(return_value=reply)
    gm.verify_phrase_at_obs.return_value = MagicMock(
        status="ABSENT", sim=0.1, ok=False, obs_id=7, phrase="sink", text_feat=None, img_feat=None
    )
    agent.robot.get_observation.return_value = None

    ex = AgenticEQAExecutor(agent, "Where is the sink?", max_rounds=4)
    ex._fresh_obs_ids.add(7)
    calls, _picked_by, meta = ex._route_tool_calls()
    assert meta["tool_calls"] == ["verify_siglip", "submit_answer"]
    outs = [ex.handle_tool(n, a) for n, a in calls]
    assert outs[0]["ok"] is True
    assert outs[0]["status"] == "CANDIDATE"
    assert outs[1]["ok"] is False
    assert "verif" in str(outs[1].get("error", "")).lower()
    assert 7 in ex._tried


def test_T4_router_env_off_fallback_only(monkeypatch):
    """T4: EMET_EQA_AGENTIC_ROUTER=0 → no tool-router VLM; fallback still answers.

    Target extract / view assess may still call eqa_client (VLM-first gate).
    """
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    monkeypatch.setenv("EMET_EQA_AGENTIC_ROUTER", "0")
    order: list[str] = []
    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    client = MagicMock(return_value='{"target_phrase":"sink","question_type":"location","notes":""}')
    gm.eqa_client = client
    gm.hypothesize_nav_targets.return_value = [
        MagicMock(obs_id=7, xyz=np.array([1.0, 2.0, 0.0]), phrase="sink", score=0.9, source="graph")
    ]
    gm._navigation_waypoint_for_obs.return_value = np.array([1.0, 2.0, 1.0])
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    def _nav(*_a, **_k):
        order.append("nav")
        return NavOutcome.REACHED

    def _verify(*_a, **_k):
        order.append("verify")
        return MagicMock(status="PRESENT", sim=0.9, ok=True, obs_id=7, phrase="sink", text_feat=None, img_feat=None)

    def _answer(*_a, **_k):
        order.append("answer")
        return ("ok", "A", True, "", None, [])

    agent.navigate_to_target_pose = _nav
    gm.verify_phrase_at_obs = _verify
    gm.query_answer = _answer

    class _Assess:
        target = "sink"
        present = True
        answerable = True
        need_more_views = False
        suggested_answer = "A"
        reason = "sink in view"
        raw = "{}"

    monkeypatch.setattr(
        "emet.eval.agentic_vlm_assess.assess_view_with_vlm",
        lambda *a, **k: _Assess(),
    )

    ex = AgenticEQAExecutor(agent, "Where is the sink?", max_rounds=4, max_nav_steps=2)
    assert ex._router_enabled is False
    result = ex.run()
    # No tool-routing system prompt (identity + Response format block).
    for call in client.call_args_list:
        sp = str((call.kwargs or {}).get("system_prompt") or "")
        assert "Response format" not in sp
        assert "tool_calls" not in sp
    assert order.index("nav") < order.index("verify") < order.index("answer")
    assert result.answer == "A"


def test_T5_state_message_marks_tried_hypotheses():
    """T5: after a failed verify, the state block annotates the hypothesis as tried."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message

    agent = MagicMock()
    agent.parameters = {}
    ex = AgenticEQAExecutor(agent, "Where is the mug?", max_rounds=4)
    hyp = MagicMock(obs_id=7, phrase="mug", score=0.9, source="graph")
    hyp.xyz = np.array([1.0, 2.0, 0.0])
    ex._hypotheses = [hyp]
    ex._tried[7] = "verify ABSENT sim=0.10"

    msg = build_state_message(ex)
    assert "Question: Where is the mug?" in msg
    assert "obs_id=7" in msg
    assert "tried: verify ABSENT" in msg
    assert "Investigate" in msg
    assert "#1 best" not in msg
    assert "highest-score" not in msg
    assert "score=" not in msg.split("Investigate", 1)[-1]


def test_T6_explore_mode_finish_gated():
    """T6: question=None → explore mode; finish rejected until budget done; explores first."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.eqa_client = None
    gm.memory_summary_enabled = False
    gm.hypothesize_nav_targets.return_value = []
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_emet_session.return_value = None
    explored = {"n": 0}

    def _explore():
        explored["n"] += 1
        return True

    agent.run_exploration = _explore

    with patch("emet.controller.habitat_nav.pick_uncovered_explore_target", return_value=None):
        ex = AgenticEQAExecutor(agent, None, goal="map the apartment", max_rounds=5, max_nav_steps=2)
        assert ex.mode == "explore"
        early = ex.handle_tool("finish", {})
        assert early["ok"] is False
        result = ex.run()

    assert explored["n"] >= 1
    assert "explore_frontier" in ex._tool_log
    assert "finish" in ex._tool_log
    assert result.answer
    assert "Explore finished" in result.discord_text
