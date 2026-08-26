# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.habitat_nav import NavOutcome
from emet.memory.graph_eqa.action_history import ActionHistoryEntry, ProgressToken
from emet.memory.graph_eqa.agentic_eqa import (
    AgenticEQAExecutor,
    AnswerEvidenceRecord,
    FinalAnswerDecision,
    PlaceInspectRecord,
)
from emet.memory.graph_eqa.agentic_state import (
    AgentStateSnapshot,
    AttemptState,
    EvidenceState,
    FrontierState,
    PlaceState,
    RoomAssertionState,
    RoomEventState,
    RoomState,
    compile_agent_state,
    render_agent_state,
    rendered_state_allowlists,
)
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, NavHypothesis
from emet.memory.graph_eqa.world_evidence import RoomHypothesis

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "graph_eqa_grounded_replay.json"


def _replay() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _place(index: int) -> PlaceState:
    return PlaceState(
        place_id=f"place_{index:03d}",
        obs_adapter_id=index,
        labels=("table", "refrigerator", "long-context-label"),
        room_id="room_1",
        room_name="kitchen",
        xyz=(float(index), 1.0, 0.5),
        source="graph",
        path_cost=float(index),
        attempts=("navigate:failed:no_path",),
        approach_bearings=(0.0, 90.0),
        coverage="open",
        coverage_gain=3.0,
        information_gain=float(100 - index),
        revisit_change="view_revision=8",
        failure_risk=0.25,
    )


def _snapshot() -> AgentStateSnapshot:
    assertions = (
        RoomAssertionState(
            source="graph",
            room_name="kitchen",
            confidence=0.8,
            world_step=29,
            agent_round=3,
            pose_round=3,
            stale=False,
        ),
        RoomAssertionState(
            source="router_vlm",
            room_name="dining_room",
            confidence=0.65,
            world_step=21,
            agent_round=2,
            pose_round=2,
            stale=True,
        ),
    )
    return AgentStateSnapshot(
        schema_version=2,
        question="Where is the silver trash can?",
        question_id="11",
        session_id="replay",
        mode="answer",
        world_step=29,
        agent_round=3,
        round_index=3,
        max_rounds=8,
        nav_used=3,
        max_nav_steps=8,
        robot_pose_world=(1.0, 2.0, 0.0),
        current_room="kitchen",
        current_room_source="graph_current_pose",
        current_room_pose_round=3,
        current_room_world_step=29,
        current_room_stale=False,
        current_room_conflict=False,
        current_room_graph="kitchen",
        current_room_router="dining_room",
        in_target_area=True,
        target_rooms=("kitchen",),
        verified=False,
        last_capture_status="CONTENT_REFRESHED",
        pending_answer="Next to the refrigerator",
        pending_answer_obs_id=76,
        pending_answer_present=True,
        places=tuple(_place(index) for index in range(1, 18)),
        frontiers=(
            FrontierState(
                frontier_id="frontier_visible",
                obs_adapter_id=901,
                xyz=(4.0, 5.0, 0.0),
                status="active",
                cell_count=12,
                attachment_ids=("place_001",),
                parent_ids=(),
                information_gain=12.0,
                path_cost=2.0,
            ),
            FrontierState(
                frontier_id="frontier_omitted",
                obs_adapter_id=902,
                xyz=(8.0, 9.0, 0.0),
                status="active",
                cell_count=4,
                attachment_ids=(),
                parent_ids=(),
                information_gain=4.0,
                path_cost=5.0,
            ),
        ),
        rooms=(
            RoomState(
                room_id="room_1",
                room_name="kitchen",
                confidence=0.8,
                sources=("graph", "router_vlm"),
                member_place_ids=tuple(f"place_{index:03d}" for index in range(1, 10)),
                centroid_xy=(1.0, 2.0),
                assertions=assertions,
                conflict=True,
            ),
        ),
        room_assertions=assertions,
        room_events=tuple(
            RoomEventState(
                event_id=f"event_room_{index}",
                world_step=20 + index,
                agent_round=index,
                room_name="kitchen",
                kind="verify_absent",
                phrase="silver trash can",
                obs_adapter_id=index,
                view_id=f"view_{index}",
                note="bounded history row",
            )
            for index in range(1, 9)
        ),
        attempts=tuple(
            AttemptState(
                action_kind="navigate",
                outcome="failed",
                status_code="no_path",
                world_step=20 + index,
                question_id="11",
                target_kind="place",
                target_id=f"place_{index:03d}",
                obs_adapter_id=index,
                view_id=f"view_{index}",
                room_name="kitchen",
                note="bounded attempt row",
            )
            for index in range(1, 9)
        ),
        evidence=tuple(
            EvidenceState(
                event_id=f"event_{index:03d}",
                step=index,
                subject_kind="place",
                subject_id=f"place_{index:03d}",
                predicate="observed",
                polarity="positive",
                source="graph",
                confidence=0.8,
                view_id=f"view_{index}",
                labels=("table",),
                details=("note=bulk evidence",),
            )
            for index in range(30)
        ),
        recent_actions=(
            "r1 investigate obs=22 fail=NAV_LOOP_BLOCKED",
            "r2 explore_frontier ok",
        ),
        loop_flags=("obs=22 visits=1 status=NAV_LOOP_BLOCKED",),
    )


def test_section_budget_preserves_contract_rows_and_whole_action_rows():
    snapshot = _snapshot()
    text = render_agent_state(snapshot, max_chars=2600)
    allowlists = rendered_state_allowlists(snapshot, text)
    compact_snapshot = replace(
        snapshot,
        question=_replay()["question"],
        session_id="/tmp/replay/session/with/a/long/deterministic/identifier",
    )
    compact = render_agent_state(compact_snapshot, max_chars=1000)
    compact_allowlists = rendered_state_allowlists(compact_snapshot, compact)

    assert len(text) <= 2600
    assert text == render_agent_state(snapshot, max_chars=2600)
    assert "Budget: round=3/8 nav=3/8" in text
    assert "Current room: canonical=kitchen" in text
    assert "Verification: verified=False" in text
    assert "Pending answer evidence:" in text
    assert "NAV_LOOP_BLOCKED" in text
    assert "Places:" in text and allowlists["place_obs_ids"]
    assert "Frontiers:" in text and allowlists["frontier_ids"]
    assert "Room events (latest first):" in text
    assert "Global attempts (latest first):" in text
    assert "Bulk evidence (latest first):" in text
    assert "omitted=" in text
    assert "[state truncated" not in text
    for obs_id in allowlists["place_obs_ids"]:
        assert f"obs_adapter={obs_id} " in text
    assert len(compact) <= 1000
    assert "Budget: round=3/8 nav=3/8" in compact
    assert "Room: canonical=kitchen" in compact
    assert "Verified: verified=False" in compact
    assert "Pending:" in compact
    assert "Room events:" in compact
    assert "Attempts:" in compact
    assert "omitted=" in compact
    assert compact_allowlists["place_obs_ids"]
    assert compact_allowlists["frontier_ids"]


def test_rendered_action_allowlist_rejects_unshown_ids_and_redirects_to_visible():
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_decision_policy": "grounded_v2"}}
    agent.graph_memory = MagicMock()
    executor = AgenticEQAExecutor(agent, "Where is the can?", router=False)
    executor._hypotheses = [
        NavHypothesis("can", 22, np.array([0.0, 0.0, 0.0]), 2.0, "graph"),
        NavHypothesis("can", 76, np.array([1.0, 0.0, 0.0]), 1.0, "graph"),
    ]
    executor._last_agent_state_snapshot = replace(
        _snapshot(),
        visible_place_ids=("place_076",),
        visible_place_obs_ids=(76,),
        visible_frontier_ids=("frontier_visible",),
    )
    executor._last_rendered_action_allowlist = executor._rendered_action_allowlist()

    accepted, rejected = executor._validate_rendered_tool_calls(
        [
            ("investigate", {"obs_id": 22}),
            ("explore_frontier", {"frontier_id": "frontier_hidden"}),
            ("investigate", {"obs_id": 76}),
        ]
    )
    assert accepted == [("investigate", {"obs_id": 76})]
    assert {item["reason"] for item in rejected} == {
        "obs_id_not_rendered",
        "frontier_id_not_rendered",
    }

    executor.action_progress_mode = "shadow"
    accepted, rejected = executor._validate_rendered_tool_calls([("explore_frontier", {})])
    assert accepted == [("explore_frontier", {})]
    assert rejected == []

    executor.action_progress_mode = "enforce"
    accepted, rejected = executor._validate_rendered_tool_calls([("explore_frontier", {})])
    assert accepted == [("explore_frontier", {"frontier_id": "frontier_visible"})]
    assert rejected == []

    executor._last_agent_state_snapshot = replace(
        executor._last_agent_state_snapshot,
        visible_place_ids=(),
        visible_place_obs_ids=(),
        visible_frontier_ids=(),
    )
    executor._last_vlm_assess = {"need_more_views": True}
    tool, _args = executor._fallback_tool()
    assert tool == "submit_answer"

    executor._last_agent_state_snapshot = replace(
        executor._last_agent_state_snapshot,
        visible_place_ids=("place_076",),
        visible_place_obs_ids=(76,),
        visible_frontier_ids=("frontier_visible",),
    )
    executor._last_rendered_action_allowlist = executor._rendered_action_allowlist()
    executor.action_progress_mode = "off"
    executor.handle_tool = MagicMock(return_value={"ok": True})
    recovered = executor._recover_failed_router_motion(
        tool="investigate",
        out={"ok": False, "status": "OBS_NOT_IN_EVIDENCE", "obs_id": 22},
    )
    assert recovered is True
    executor.handle_tool.assert_called_once_with("investigate", {"obs_id": 76})

    executor.handle_tool.reset_mock()
    recovered = executor._recover_failed_router_motion(
        tool="investigate",
        out={"ok": False, "status": "NAV_LOOP_BLOCKED", "obs_id": 76},
    )
    assert recovered is True
    executor.handle_tool.assert_called_once_with(
        "explore_frontier",
        {"frontier_id": "frontier_visible", "toward": "Where is the can?"},
    )


def test_action_progress_mode_rejects_legacy_executor_policy():
    agent = SimpleNamespace(
        parameters={
            "eqa": {
                "agentic_decision_policy": "legacy",
                "action_progress_mode": "enforce",
            }
        },
        graph_memory=_agent_memory(),
        voxel_map=None,
    )
    with pytest.raises(
        ValueError,
        match="requires agentic_decision_policy=grounded_v2",
    ):
        AgenticEQAExecutor(agent, "Where is the can?", router=False)


def test_router_persists_exact_allowlist_and_nearby_caption_is_not_actionable():
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_decision_policy": "grounded_v2"}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.eqa_client = MagicMock(
        return_value=('{"current_room":"kitchen","tool_calls":[{"name":"investigate","arguments":{"obs_id":22}}]}')
    )
    agent.graph_memory.graph_room_at_robot.return_value = "kitchen"
    executor = AgenticEQAExecutor(
        agent,
        "Where is the can?",
        router=True,
        collect_trace=True,
        trace_meta={"qid": 11, "session_id": 1001},
    )
    executor._tools = []
    executor._tool_names = {"investigate", "explore_frontier"}
    executor._system_prompt = "test"
    executor._hypotheses = [
        NavHypothesis("nearby caption only", 22, np.array([0.0, 0.0, 0.0]), 2.0, "graph"),
        NavHypothesis("rendered place", 76, np.array([1.0, 0.0, 0.0]), 1.0, "graph"),
    ]

    def _rendered_state(current_executor):
        current_executor._last_agent_state_snapshot = replace(
            _snapshot(),
            visible_place_ids=("place_076",),
            visible_place_obs_ids=(76,),
            visible_frontier_ids=("frontier_visible",),
        )
        return "Places:\n- place_id=place_076 obs_adapter=76 room=kitchen"

    with (
        patch(
            "emet.memory.graph_eqa.agentic_router.build_state_message",
            side_effect=_rendered_state,
        ),
        patch.object(
            executor,
            "_room_visual_pack",
            return_value=(
                [],
                [{"obs_id": 22, "phrase": "nearby caption only"}],
                "Room context: nearby obs_id=22\n",
            ),
        ),
        patch.object(executor, "_robot_xyt_world", return_value=np.zeros(3)),
    ):
        calls, picked_by, meta = executor._route_tool_calls()

    assert picked_by == "fallback"
    assert calls == [("investigate", {"obs_id": 76})]
    assert meta["rejected_tool_calls"] == [{"tool": "investigate", "reason": "obs_id_not_rendered", "value": 22}]
    router_call_id = meta["router_call_id"]
    assert executor._router_action_allowlists[router_call_id]["place_obs_ids"] == (76,)
    assert 22 not in executor._router_action_allowlists[router_call_id]["place_obs_ids"]
    router_trace = next(
        row
        for row in executor._trace_rows
        if row.get("event") == "router_call" and row.get("router_call_id") == router_call_id
    )
    assert router_trace["action_allowlist"]["place_obs_ids"] == [76]


class _RoomMemory:
    def __init__(self) -> None:
        self.step = 41
        self.room_events: list[dict] = []
        self.world_evidence = SimpleNamespace(question_id=None, session_id="")

    def bind_episode_context(self, *, question_id, session_id) -> None:
        self.world_evidence.question_id = str(question_id)
        self.world_evidence.session_id = str(session_id)

    def _effective_timestep(self) -> int:
        return self.step

    @staticmethod
    def graph_room_at_robot(xyt) -> str:
        return "kitchen" if float(xyt[0]) < 1.0 else "living_room"

    @staticmethod
    def observation_room(obs_id: int) -> tuple[str, str]:
        assert obs_id == 76
        return "room_view", "dining_room"

    def record_room_event(self, **kwargs):
        self.room_events.append(kwargs)
        return kwargs


def test_room_is_recomputed_after_motion_and_events_use_view_room():
    memory = _RoomMemory()
    pose = np.array([0.0, 0.0, 0.0])
    robot = SimpleNamespace(get_base_pose=lambda: pose)
    agent = SimpleNamespace(
        parameters={"eqa": {"agentic_decision_policy": "grounded_v2"}},
        graph_memory=memory,
        robot=robot,
        _planning_base_xyt=lambda xyt: xyt,
    )
    executor = AgenticEQAExecutor(
        agent,
        "Is the object in the kitchen?",
        router=False,
        trace_meta={"qid": 11, "session_id": 7},
    )
    assert executor._refresh_graph_room_estimate() == "kitchen"
    executor._last_router_room_estimate = "kitchen"
    executor._router_room_stale = False

    pose[0] = 2.0
    memory.step = 42
    assert executor._refresh_room_after_motion() == "living_room"
    assert executor._last_room_estimate == "living_room"
    assert executor._current_room_source == "graph_current_pose"
    assert executor._room_estimate_stale is False
    assert executor._graph_room_stale is False
    assert executor._router_room_stale is True
    assert executor._in_target_area is False
    assert executor._room_world_step == 42

    executor._round = 2
    executor._record_room_timeline(
        kind="verify_absent",
        obs_id=76,
        phrase="silver trash can",
    )
    assert memory.room_events[-1]["room"] == "dining_room"
    assert memory.room_events[-1]["step"] == 42
    assert memory.room_events[-1]["agent_round"] == 3

    prior_pose_round = executor._room_pose_round
    memory.step = 43
    memory.graph_room_at_robot = lambda _xyt: "unknown"
    assert executor._refresh_room_after_motion() == "unknown"
    assert executor._last_room_estimate == "living_room"
    assert executor._room_estimate_stale is True
    assert executor._graph_room_stale is True
    assert executor._room_pose_round == prior_pose_round


def _agent_memory() -> GraphEQAMemory:
    memory = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={
            "eqa": {
                "graph_evidence_mode": "agent",
                "attempt_ledger": True,
            }
        },
    )
    memory.spatial_merge_m = 0.5
    return memory


def _restore_replay_observation_identity(
    memory: GraphEQAMemory,
    obs_id: int,
    observation: dict,
) -> None:
    world = memory.world_evidence
    view = world.view_for_obs(obs_id)
    assert view is not None and view.place_id is not None
    old_view_id = view.view_id
    old_place_id = view.place_id
    new_view_id = observation["view_id"]
    new_place_id = observation["place_id"]

    place = world.places.pop(old_place_id)
    world.places[new_place_id] = replace(place, place_id=new_place_id)
    if view.entity_id is not None and view.entity_id in world.entities:
        world.entities[view.entity_id] = replace(
            world.entities[view.entity_id],
            place_id=new_place_id,
        )
    world.views.pop(old_view_id)
    world.views[new_view_id] = replace(
        view,
        view_id=new_view_id,
        revision=observation["revision"],
        place_id=new_place_id,
    )
    world._obs_current_view[obs_id] = new_view_id
    world._obs_revision_to_view.pop((obs_id, view.revision), None)
    world._obs_revision_to_view[(obs_id, observation["revision"])] = new_view_id
    world.events = [
        replace(
            event,
            subject_id=(new_place_id if event.subject_id == old_place_id else event.subject_id),
            view_id=(new_view_id if event.view_id == old_view_id else event.view_id),
            place_id=(new_place_id if event.place_id == old_place_id else event.place_id),
        )
        for event in world.events
    ]


def test_typed_snapshot_has_conflicts_clocks_ids_and_referential_history():
    replay = _replay()
    case = replay["cases"]["dirty_obs76"]
    memory = _agent_memory()
    memory.bind_episode_context(
        question_id=replay["question_id"],
        session_id=case["session_id"],
    )
    obs = case["observation"]
    obs_id = memory.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        obs["xyz"],
        obs["labels"],
        viewer_xyz=[-0.9, 2.3, 0.0],
    )
    entity = next(iter(memory.world_evidence.entities.values()))
    room_id = "room_replay"
    memory.world_evidence.rooms[room_id] = RoomHypothesis(
        room_id=room_id,
        member_place_ids=(entity.place_id,),
        centroid_xy=(obs["xyz"][0], obs["xyz"][1]),
    )
    memory.world_evidence.places[entity.place_id] = replace(
        memory.world_evidence.places[entity.place_id],
        room_id=room_id,
    )
    memory.world_evidence.stamp_room(
        room_id,
        "dining_room",
        source="graph_labels",
        confidence=0.8,
        step=23,
    )
    memory.world_evidence.stamp_room(
        room_id,
        "kitchen",
        source="router_vlm",
        confidence=0.9,
        step=29,
        agent_round=3,
        pose_round=3,
    )
    memory.record_attempt(
        action_kind="navigate",
        outcome="ok",
        status_code="ok",
        obs_id=obs_id,
        step=29,
        source="eqa",
    )
    room_event = memory.record_room_event(
        room="kitchen",
        kind="verified",
        obs_id=obs_id,
        step=29,
        agent_round=3,
    )

    agent = SimpleNamespace(
        parameters={
            "eqa": {
                "agentic_decision_policy": "grounded_v2",
                "graph_evidence_mode": "agent",
                "room_history_mode": "agent",
                "attempt_ledger_mode": "agent",
            }
        },
        graph_memory=memory,
    )
    executor = AgenticEQAExecutor(
        agent,
        replay["question"],
        router=False,
        trace_meta={"qid": replay["question_id"], "session_id": case["session_id"]},
    )
    executor._round = 2
    executor._hypotheses = [NavHypothesis("dining table", obs_id, np.asarray(obs["xyz"]), 1.0, "graph")]
    executor._graph_room_estimate = "kitchen"
    executor._last_room_estimate = "kitchen"
    executor._last_router_room_estimate = "kitchen"
    executor._room_estimate_stale = False
    executor._router_room_stale = False
    executor._room_pose_round = 3
    executor._room_world_step = 29
    snapshot = compile_agent_state(executor)

    replay_room = next(item for item in snapshot.rooms if item.room_id == room_id)
    assert replay_room.conflict is True
    assert {item.source for item in replay_room.assertions} == {
        "graph_labels",
        "router_vlm",
    }
    assert snapshot.question_id == "11"
    assert snapshot.session_id == case["session_id"]
    assert snapshot.agent_round == 3
    assert snapshot.world_step != snapshot.agent_round
    assert snapshot.attempts[0].question_id == "11"
    assert snapshot.room_events[0].event_id == room_event["event_id"]
    event_ids = {event.event_id for event in memory.world_evidence.events}
    assert snapshot.room_events[0].event_id in event_ids
    assert snapshot.attempts[0].view_id in memory.world_evidence.views


def test_attempt_mode_cleanly_gates_grounded_action_history():
    memory = _agent_memory()
    memory.bind_episode_context(question_id=11, session_id="history-ablation")
    memory.record_attempt(
        action_kind="navigate",
        outcome="failed",
        status_code="no_path",
        note="blocked route",
        step=3,
        source="eqa",
    )
    common = {
        "graph_memory": memory,
        "graph_evidence_mode": "agent",
        "room_history_mode": "agent",
        "question": "What is beside the table?",
        "_question_id": "11",
        "_session_id": "history-ablation",
        "_round": 2,
        "_hypotheses": (),
        "_recent_actions": ("navigate obs=7 -> failed:no_path",),
        "_nav_loop_flags": ({"obs_id": 7, "visits": 2, "status": "no_path"},),
        "room_target_hints": False,
    }

    shadow = compile_agent_state(
        SimpleNamespace(**common, attempt_ledger_mode="shadow"),
    )
    shadow_text = render_agent_state(shadow, max_chars=6000)
    assert shadow.attempts == ()
    assert shadow.recent_actions == ()
    assert shadow.loop_flags == ()
    assert all(event.predicate != "navigate" for event in shadow.evidence)
    assert "failed:no_path" not in shadow_text
    assert "blocked route" not in shadow_text

    agent = compile_agent_state(
        SimpleNamespace(**common, attempt_ledger_mode="agent"),
    )
    agent_text = render_agent_state(agent, max_chars=6000)
    assert agent.attempts[0].status_code == "no_path"
    assert agent.recent_actions == ("navigate obs=7 -> failed:no_path",)
    assert agent.loop_flags == ("action=investigate visits=2 status=no_path adapter=7",)
    assert any(event.predicate == "navigate" for event in agent.evidence)
    assert "failed:no_path" in agent_text
    assert "blocked route" in agent_text


def test_static_progress_shadow_observes_and_enforce_removes_duplicate_candidate():
    memory = _agent_memory()
    memory.bind_episode_context(question_id=11, session_id="action-progress")
    obs_id = memory.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [1.0, 2.0, 0.5],
        ["kitchen island", "counter"],
    )
    agent = SimpleNamespace(
        parameters={
            "eqa": {
                "agentic_decision_policy": "grounded_v2",
                "graph_evidence_mode": "agent",
                "room_history_mode": "agent",
                "attempt_ledger_mode": "agent",
                "action_progress_mode": "enforce",
            }
        },
        graph_memory=memory,
        voxel_map=None,
    )
    executor = AgenticEQAExecutor(
        agent,
        "Where is the silver trash can?",
        router=False,
        trace_meta={"qid": 11, "session_id": "action-progress"},
    )
    executor._round = 1
    executor._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])
    executor._hypotheses = [
        NavHypothesis(
            "kitchen island",
            obs_id,
            np.array([1.0, 2.0, 0.5]),
            1.0,
            "graph",
        )
    ]
    first_frontier = executor._action_signature(
        "explore_frontier",
        {"frontier_id": "frontier_same", "toward": "the kitchen"},
    )
    reworded_frontier = executor._action_signature(
        "explore_frontier",
        {"frontier_id": "frontier_same", "toward": "the trash can"},
    )
    assert first_frontier.work_key == reworded_frontier.work_key
    assert first_frontier.equivalence_key == reworded_frontier.equivalence_key
    assert first_frontier.intent == "the kitchen"
    assert reworded_frontier.intent == "the trash can"
    assert first_frontier.work_intent == reworded_frontier.work_intent

    executor._place_inspect[obs_id] = PlaceInspectRecord(
        tried_approaches=list(range(4)),
    )
    executor._action_history = []
    for approach in range(4):
        signature = executor._action_signature(
            "investigate",
            {"obs_id": obs_id, "approach_index": approach},
        )
        token = executor._action_progress_token(signature)
        executor._action_history.append(
            ActionHistoryEntry(
                schema_version=1,
                round_index=approach + 1,
                selected_by="router",
                signature=signature,
                progress_before=token,
                progress_after=token,
                outcome_class="no_progress",
                status="no_path",
                ok=False,
            )
        )

    enforced = compile_agent_state(executor)
    enforced_text = render_agent_state(enforced)
    enforced_allowlist = rendered_state_allowlists(enforced, enforced_text)
    assert enforced.places == ()
    assert enforced_allowlist["place_obs_ids"] == ()
    assert enforced.blocked_actions[0].target_label == "kitchen island/counter"
    assert "Temporarily suppressed actions:" in enforced_text
    assert "all finite approach variants are temporarily ineligible" in enforced_text

    executor.action_progress_mode = "shadow"
    shadow = compile_agent_state(executor)
    shadow_text = render_agent_state(shadow)
    shadow_allowlist = rendered_state_allowlists(shadow, shadow_text)
    assert shadow_allowlist["place_obs_ids"] == (obs_id,)
    assert shadow.blocked_actions == ()
    assert shadow.gate_decisions[0].allowed is False
    assert "all finite approach variants are temporarily ineligible" not in shadow_text

    executor.action_progress_mode = "off"
    off = compile_agent_state(executor)
    assert render_agent_state(off) == shadow_text


def test_enforce_revalidates_same_frontier_before_each_batch_execution():
    memory = _agent_memory()
    memory.bind_episode_context(question_id=11, session_id="batch-revalidate")
    frontier = memory.world_evidence.update_frontier_tracks(
        [{"centroid_xyz": (1.0, 0.0, 0.0), "cells": [(1, 1), (1, 2)]}],
        step=1,
    )[0]
    agent = SimpleNamespace(
        parameters={
            "eqa": {
                "agentic_decision_policy": "grounded_v2",
                "action_progress_mode": "enforce",
            }
        },
        graph_memory=memory,
        voxel_map=None,
    )
    executor = AgenticEQAExecutor(agent, "Where is the can?", router=False)
    executor._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])
    executor._last_agent_state_snapshot = replace(
        _snapshot(),
        visible_frontier_ids=(frontier.frontier_id,),
    )
    executor._tool_explore_frontier = MagicMock(
        return_value={
            "ok": False,
            "status": "no_path",
            "frontier_id": frontier.frontier_id,
        }
    )

    first = executor.handle_tool(
        "explore_frontier",
        {"frontier_id": frontier.frontier_id},
    )
    second = executor.handle_tool(
        "explore_frontier",
        {"frontier_id": frontier.frontier_id},
    )

    assert first["status"] == "no_path"
    assert second["status"] == "ACTION_PROGRESS_SUPPRESSED"
    assert second["disposition"] == "would_suppress_duplicate"
    executor._tool_explore_frontier.assert_called_once()


def test_shadow_omitted_frontier_records_actual_pre_motion_progress():
    memory = _agent_memory()
    memory.bind_episode_context(question_id=11, session_id="frontier-pre-state")
    frontier = memory.world_evidence.update_frontier_tracks(
        [{"centroid_xyz": (1.0, 0.0, 0.0), "cells": [(1, 1), (1, 2)]}],
        step=1,
    )[0]
    pose = [np.array([0.0, 0.0, 0.0])]

    def navigate(*_args, **_kwargs):
        pose[0] = np.array([0.5, 0.0, 0.0])
        return NavOutcome.PROGRESS

    agent = SimpleNamespace(
        parameters={
            "eqa": {
                "agentic_decision_policy": "grounded_v2",
                "action_progress_mode": "shadow",
            }
        },
        graph_memory=memory,
        voxel_map=None,
        _vlm_frontier_choice=lambda *_args, **_kwargs: np.array(frontier.centroid_xyz),
        navigate_to_target_pose=navigate,
        _last_nav_attempt=None,
    )
    executor = AgenticEQAExecutor(agent, "Where is the can?", router=False)
    executor._robot_xyt_world = lambda: pose[0].copy()
    executor._tool_capture_and_update = MagicMock(
        return_value={"ok": False, "status": "NO_NEW_OBS"},
    )
    executor._tool_look_around = MagicMock(
        return_value={"ok": False, "capture": {"ok": False}},
    )
    executor._refresh_room_after_motion = MagicMock()
    executor._save_frontier_pick_panel = MagicMock(return_value=None)

    out = executor.handle_tool("explore_frontier", {"toward": "kitchen"})

    assert out["frontier_id"] == frontier.frontier_id
    entry = executor._action_history[-1]
    assert entry.signature.target.stable_id == frontier.frontier_id
    assert entry.progress_before.value("robot_pose_cell") == [0, 0]
    assert entry.progress_after.value("robot_pose_cell") == [2, 0]
    assert "motion" in entry.progress_reasons
    assert entry.outcome_class == "progress"

    agent.navigate_to_target_pose = lambda *_args, **_kwargs: NavOutcome.ABORTED_TIMEOUT
    timeout = executor.handle_tool(
        "explore_frontier",
        {"frontier_id": frontier.frontier_id},
    )
    assert timeout["status"] == "aborted_timeout"
    assert executor._action_history[-1].outcome_class == "transient"


def test_shadow_traces_terminal_verify_decision_without_blocking_dispatch():
    memory = _agent_memory()
    memory.bind_episode_context(question_id=11, session_id="verify-shadow")
    obs_id = memory.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [1.0, 2.0, 0.5],
        ["kitchen island"],
    )
    agent = SimpleNamespace(
        parameters={
            "eqa": {
                "agentic_decision_policy": "grounded_v2",
                "action_progress_mode": "shadow",
            }
        },
        graph_memory=memory,
        voxel_map=None,
    )
    executor = AgenticEQAExecutor(
        agent,
        "Where is the can?",
        router=False,
        collect_trace=True,
    )
    executor._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])
    executor._tool_verify_siglip = MagicMock(
        return_value={"ok": True, "status": "ABSENT", "obs_id": obs_id},
    )

    executor.handle_tool("verify_siglip", {"obs_id": obs_id, "phrase": "can"})
    executor.handle_tool("verify_siglip", {"obs_id": obs_id, "phrase": "can"})

    assert executor._tool_verify_siglip.call_count == 2
    dispatch_rows = [
        row
        for row in executor._trace_rows
        if row.get("event") == "action_gate_dispatch" and row.get("tool") == "verify_siglip"
    ]
    assert dispatch_rows[-1]["decision"]["allowed"] is False
    assert dispatch_rows[-1]["decision"]["disposition"] == "suppressed_terminal_view"


@pytest.mark.parametrize(
    "replay_case",
    _replay()["stalled_action_replays"],
    ids=lambda case: f"q{case['question_id']}-{case['source_run']}",
)
def test_q11_q12_stalled_replay_reopens_unused_approach_before_saturation(replay_case):
    question_id = replay_case["question_id"]
    question = replay_case["question"]
    observation = replay_case["observation"]
    stalled_attempt = replay_case["stalled_attempt"]
    memory = _agent_memory()
    memory.bind_episode_context(
        question_id=question_id,
        session_id=replay_case["session_id"],
    )
    obs_id = memory.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        observation["xyz"],
        observation["labels"],
    )
    _restore_replay_observation_identity(memory, obs_id, observation)
    agent = SimpleNamespace(
        parameters={
            "eqa": {
                "agentic_decision_policy": "grounded_v2",
                "graph_evidence_mode": "agent",
                "room_history_mode": "agent",
                "attempt_ledger_mode": "agent",
                "action_progress_mode": "enforce",
            }
        },
        graph_memory=memory,
        voxel_map=None,
    )
    executor = AgenticEQAExecutor(
        agent,
        question,
        router=False,
        trace_meta={
            "qid": question_id,
            "session_id": replay_case["session_id"],
        },
    )
    executor._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])
    executor._hypotheses = [
        NavHypothesis(
            observation["labels"][0],
            obs_id,
            np.array(observation["xyz"]),
            1.0,
            "graph",
        )
    ]
    first_signature = executor._action_signature(
        "investigate",
        {
            "obs_id": obs_id,
            "approach_index": stalled_attempt["approach_index"],
        },
    )
    assert first_signature.target.stable_id == observation["place_id"]
    token = executor._action_progress_token(first_signature)
    executor._action_history = [
        ActionHistoryEntry(
            schema_version=1,
            round_index=1,
            selected_by="router",
            signature=first_signature,
            progress_before=token,
            progress_after=token,
            outcome_class="no_progress",
            status=stalled_attempt["status"],
            ok=False,
        )
    ]
    executor._tried[obs_id] = f"{stalled_attempt['status']} verify={stalled_attempt['verify_status']}"
    executor._place_inspect[obs_id] = PlaceInspectRecord(
        investigate_count=1,
        tried_approaches=[stalled_attempt["approach_index"]],
        coverage=stalled_attempt["coverage"],
        local_frontier_cells=stalled_attempt.get("local_frontier_cells", 0),
    )

    alternate = compile_agent_state(executor)
    assert tuple(place.obs_adapter_id for place in alternate.places) == (obs_id,)
    assert alternate.gate_decisions[0].allowed is True
    assert alternate.gate_decisions[0].disposition == "allowed_alternate"

    current = executor._action_progress_token(first_signature)
    executor._action_history = [
        ActionHistoryEntry(
            schema_version=1,
            round_index=1,
            selected_by="router",
            signature=first_signature,
            progress_before=ProgressToken.build({"robot_pose_cell": (-1, 0)}),
            progress_after=current,
            outcome_class="progress",
            status="progress",
            ok=True,
            progress_reasons=("motion",),
        )
    ]
    continuation = compile_agent_state(executor)
    assert continuation.gate_decisions[0].allowed is True
    assert continuation.gate_decisions[0].disposition == "allowed_progress"
    assert continuation.gate_decisions[0].equivalence_key == first_signature.equivalence_key

    executor._place_inspect[obs_id].tried_approaches = [0, 1, 2, 3]
    executor._action_history = []
    for approach in range(4):
        signature = executor._action_signature(
            "investigate",
            {"obs_id": obs_id, "approach_index": approach},
        )
        unchanged = executor._action_progress_token(signature)
        executor._action_history.append(
            ActionHistoryEntry(
                schema_version=1,
                round_index=approach + 1,
                selected_by="router",
                signature=signature,
                progress_before=unchanged,
                progress_after=unchanged,
                outcome_class="no_progress",
                status=stalled_attempt["status"],
                ok=False,
            )
        )
    saturated = compile_agent_state(executor)
    assert saturated.places == ()
    assert saturated.gate_decisions[0].disposition == "would_suppress_saturated"

    view = memory.world_evidence.view_for_obs(obs_id)
    assert view is not None
    memory.world_evidence.views[view.view_id] = replace(
        view,
        revision=view.revision + 1,
    )
    reopened = compile_agent_state(executor)
    assert tuple(place.obs_adapter_id for place in reopened.places) == (obs_id,)
    assert reopened.gate_decisions[0].allowed is True
    assert reopened.gate_decisions[0].disposition == "allowed_progress"


def test_replay_fused_confirmation_survives_raw_absent_and_invalidates_on_revision():
    replay = _replay()
    dirty = replay["cases"]["dirty_obs76"]
    clean = replay["cases"]["clean_obs22"]

    clean_memory = _agent_memory()
    clean_memory.bind_episode_context(question_id=11, session_id=clean["session_id"])
    clean_obs = clean_memory.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        clean["observation"]["xyz"],
        clean["observation"]["labels"],
    )
    for row in clean["evidence"]:
        clean_memory.record_agentic_evidence(
            stage=row["stage"],
            outcome=row["outcome"],
            obs_id=clean_obs,
            phrase="silver trash can",
            confidence=0.9,
            source=row["stage"],
            score=row.get("score"),
            payload={
                "answerable": row.get("answerable"),
                "need_more_views": row.get("need_more_views"),
            },
        )
    assert (
        clean_memory.durable_confirmation_event_ids(
            obs_id=clean_obs,
            phrase="silver trash can",
        )
        == ()
    )

    memory = _agent_memory()
    memory.bind_episode_context(question_id=11, session_id=dirty["session_id"])
    obs = dirty["observation"]
    obs_id = memory.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        obs["xyz"],
        obs["labels"],
    )
    event_ids: list[str] = []
    for row in dirty["evidence"]:
        event_id = memory.record_agentic_evidence(
            stage=row["stage"],
            outcome=row["outcome"],
            obs_id=obs_id,
            phrase="silver trash can",
            confidence=1.0,
            source=row["stage"],
            score=row.get("score"),
            supporting_event_ids=tuple(event_ids),
            payload={
                "answerable": row.get("answerable"),
                "need_more_views": row.get("need_more_views"),
                "suggested_answer": row.get("suggested_answer"),
            },
        )
        assert event_id
        event_ids.append(event_id)
    raw_absent_id = memory.record_agentic_evidence(
        stage="siglip_proposal",
        outcome="absent",
        obs_id=obs_id,
        phrase="silver trash can",
        confidence=1.0,
        source="dense_patch",
        score=0.01,
    )
    durable = memory.durable_confirmation_event_ids(
        obs_id=obs_id,
        phrase="silver trash can",
    )
    assert event_ids[-1] in durable
    assert event_ids[1] in durable
    assert raw_absent_id not in durable
    persisted_ids = {event.event_id for event in memory.world_evidence.events}
    assert set(durable) <= persisted_ids
    assert all(event.question_id == "11" for event in memory.world_evidence.events)
    assert all(event.session_id == dirty["session_id"] for event in memory.world_evidence.events)

    evidence = AnswerEvidenceRecord(
        letter="D",
        source="vlm_suggested",
        answer_text=dirty["expected"]["answer"],
        obs_id=obs_id,
        obs_revision=memory.obs_revision(obs_id),
        view_id=memory.view_id_for_obs(obs_id),
        present=True,
        answerable=True,
        confidence=0.65,
        evidence_event_ids=durable,
    )
    decision = FinalAnswerDecision(
        answer=dirty["expected"]["answer"],
        source="vlm_suggested",
        confidence=0.65,
        evidence=evidence,
    )
    assert set(decision.to_dict()["evidence_event_ids"]) == set(durable)

    memory.bind_episode_context(question_id=11, session_id="next-session")
    assert (
        memory.durable_confirmation_event_ids(
            obs_id=obs_id,
            phrase="silver trash can",
        )
        == ()
    )
    memory.bind_episode_context(question_id=11, session_id=dirty["session_id"])
    memory.add_observation(
        np.full((4, 4, 3), 200, dtype=np.uint8),
        np.asarray(obs["xyz"]) + np.array([0.01, 0.0, 0.0]),
        obs["labels"],
    )
    assert memory.obs_revision(obs_id) == 2
    assert (
        memory.durable_confirmation_event_ids(
            obs_id=obs_id,
            phrase="silver trash can",
        )
        == ()
    )


def test_grounded_verified_submission_references_persisted_confirmation():
    replay = _replay()
    dirty = replay["cases"]["dirty_obs76"]
    memory = _agent_memory()
    obs = dirty["observation"]
    obs_id = memory.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        obs["xyz"],
        obs["labels"],
    )
    memory.eqa_client = MagicMock()
    agent = MagicMock()
    agent.parameters = {
        "eqa": {
            "agentic_decision_policy": "grounded_v2",
            "graph_evidence_mode": "agent",
            "room_history_mode": "agent",
            "attempt_ledger_mode": "agent",
        }
    }
    agent.graph_memory = memory
    agent.voxel_map = None
    executor = AgenticEQAExecutor(
        agent,
        replay["question"],
        router=False,
        trace_meta={"qid": replay["question_id"], "session_id": dirty["session_id"]},
    )
    executor._target_phrase = "silver trash can"
    hypothesis_id = executor._begin_policy_approach("graph", obs_id, executor._target_phrase)
    executor._policy_approached(hypothesis_id, obs_id)
    proposal_event_id = executor._persist_agentic_evidence(
        stage="siglip_proposal",
        outcome="absent",
        obs_id=obs_id,
        phrase=executor._target_phrase,
        confidence=0.1,
        source="dense_patch",
        score=dirty["evidence"][0]["score"],
    )

    assessment = SimpleNamespace(
        target="silver trash can",
        present=True,
        answerable=True,
        need_more_views=False,
        suggested_answer=dirty["expected"]["answer"],
        reason="visible next to refrigerator",
        raw="{}",
    )
    with (
        patch(
            "emet.eval.agentic_vlm_assess.assess_view_with_vlm",
            return_value=assessment,
        ),
        patch(
            "emet.eval.agentic_vlm_assess.build_inventory_brief",
            return_value="replay inventory",
        ),
    ):
        result = executor._run_vlm_view_assess(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            phrase=executor._target_phrase,
            obs_id=obs_id,
            proposal={
                "decision": "ABSENT",
                "evidence_event_id": proposal_event_id,
            },
        )

    assert result["verified"] is True
    assert executor._verified_evidence_event_ids
    persisted = {event.event_id for event in memory.world_evidence.events}
    assert set(executor._verified_evidence_event_ids) <= persisted
    fused = next(event for event in memory.world_evidence.events if event.predicate == "fused_confirmation")
    assert set(fused.payload["supporting_event_ids"]) <= persisted
    assert all(
        event.view_id == fused.view_id
        for event in memory.world_evidence.events
        if event.event_id in fused.payload["supporting_event_ids"]
    )
    assert executor._confirmed_answer_evidence is not None
    assert executor._confirmed_answer_evidence.evidence_event_ids == executor._verified_evidence_event_ids

    memory.query_answer = MagicMock(return_value=("", dirty["expected"]["answer"], True, None, None, []))
    submitted = executor._do_submit_answer(prefer_answer=dirty["expected"]["answer"])
    assert submitted["final_decision"]["evidence_event_ids"]
    assert set(submitted["final_decision"]["evidence_event_ids"]) <= persisted


def test_content_revision_clears_executor_verification_caches():
    memory = _agent_memory()
    xyz = np.array([1.0, 0.0, 0.5])
    obs_id = memory.add_observation(
        np.zeros((8, 8, 3), dtype=np.uint8),
        xyz,
        ["clock"],
    )
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_decision_policy": "grounded_v2"}}
    agent.graph_memory = memory
    agent.update = MagicMock(
        side_effect=lambda: memory.add_observation(
            np.full((8, 8, 3), 180, dtype=np.uint8),
            xyz + np.array([0.01, 0.0, 0.0]),
            ["clock"],
        )
    )
    executor = AgenticEQAExecutor(agent, "Where is the clock?", router=False)
    executor._refresh_hypotheses_from_graph = MagicMock()
    executor._verified = True
    executor._verified_obs_id = obs_id
    executor._verified_evidence_event_ids = ("event_old",)
    executor._vlm_assessed_obs_ids.add(obs_id)
    executor._assess_history[obs_id] = {"present": True}
    executor._confirmed_answer_evidence = AnswerEvidenceRecord(
        letter="A",
        source="vlm_suggested",
        obs_id=obs_id,
        present=True,
        answerable=True,
        evidence_event_ids=("event_old",),
    )

    out = executor._tool_capture_and_update()
    assert out["status"] == "CONTENT_REFRESHED"
    assert executor._verified is False
    assert executor._verified_obs_id is None
    assert executor._verified_evidence_event_ids == ()
    assert executor._confirmed_answer_evidence is None
    assert obs_id not in executor._vlm_assessed_obs_ids
    assert obs_id not in executor._assess_history


def test_episode_context_reset_keeps_old_event_metadata_immutable():
    memory = _agent_memory()
    obs_id = memory.add_observation(
        np.zeros((2, 2, 3), dtype=np.uint8),
        [0.0, 0.0, 0.0],
        ["chair"],
    )
    memory.bind_episode_context(question_id=11, session_id=1001)
    first = memory.record_agentic_evidence(
        stage="siglip_proposal",
        outcome="candidate",
        obs_id=obs_id,
        phrase="chair",
        confidence=0.6,
        source="siglip",
    )
    memory.bind_episode_context(question_id=12, session_id=1002)
    second = memory.record_agentic_evidence(
        stage="siglip_proposal",
        outcome="candidate",
        obs_id=obs_id,
        phrase="chair",
        confidence=0.6,
        source="siglip",
    )
    events = {event.event_id: event for event in memory.world_evidence.events}
    assert (events[first].question_id, events[first].session_id) == ("11", "1001")
    assert (events[second].question_id, events[second].session_id) == ("12", "1002")
    with pytest.raises(ValueError, match="incompatible supporting evidence"):
        memory.record_agentic_evidence(
            stage="fused_confirmation",
            outcome="confirmed",
            obs_id=obs_id,
            phrase="chair",
            confidence=1.0,
            source="agentic_policy",
            supporting_event_ids=(first,),
        )


def test_controller_forwards_numeric_trace_context_before_agentic_run():
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True}}
    agent.graph_memory = MagicMock()
    with (
        patch(
            "emet.memory.graph_eqa.agentic_eqa.agentic_verify_enabled",
            return_value=True,
        ),
        patch(
            "emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa",
            return_value=("answer", []),
        ) as run_agentic,
    ):
        result = GraphEQAController.run_eqa(
            agent,
            "Where is the chair?",
            trace_meta={"qid": 11, "session_id": 1001},
        )

    assert result == ("answer", [])
    agent.graph_memory.bind_episode_context.assert_called_once_with(
        question_id=11,
        session_id=1001,
    )
    assert run_agentic.call_args.kwargs["trace_meta"] == {
        "qid": 11,
        "session_id": 1001,
    }
