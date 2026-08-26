# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Locate-question routing on the shared agentic executor (no sim / VLM).

OVMM find only phrases ``Where is the X?`` — these tests must not depend on
``ovmm_phase`` or GT placement seeds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from emet.memory.graph_eqa.agentic_config import question_is_locate
from emet.memory.graph_eqa.agentic_eqa import (
    DEFAULT_INVESTIGATE_ANNULUS_OUTER_M,
    INVESTIGATE_ANNULUS_OUTER_M,
    NEAR_INVESTIGATE_M,
    AgenticEQAExecutor,
)
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, NavHypothesis


def _executor(*, question: str = "Where is the microwave?") -> AgenticEQAExecutor:
    agent = MagicMock()
    agent.parameters = {}
    agent.robot = MagicMock()
    return AgenticEQAExecutor(
        agent,
        question,
        max_rounds=4,
        router=False,
    )


def test_question_is_locate_for_ovmm_phrasing():
    assert question_is_locate("Where is the microwave?")
    assert question_is_locate("Where is the jar on the counter?")
    assert not question_is_locate("Where did you see the soap dispenser? A) Above the sink B) On the toilet tank")


def test_locate_question_prefers_nearby_investigate():
    ex = _executor(question="Where is the microwave?")
    assert ex._prefers_nearby_investigate() is True
    other = _executor(question="What color is the sofa? A) Red B) Blue")
    assert other._prefers_nearby_investigate() is False


def test_recall_nav_hypotheses_boosts_target_phrase_not_gt():
    ex = _executor(question="Where is the microwave?")
    ex._target_phrase = "microwave"
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = []
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm
    ex.agent.robot.get_emet_session.return_value = {
        "is_simulation": True,
        "sim_object_placements": {
            "microwave_gt": {"pos": [9.0, 9.0, 0.9], "cat": "microwave"},
        },
    }
    hyps = ex._recall_nav_hypotheses()
    assert all(int(h.obs_id) > -3_000_000 for h in hyps)
    kwargs = gm.hypothesize_nav_targets.call_args.kwargs
    boost = kwargs.get("boost_phrases") or []
    assert any("microwave" in str(p).lower() for p in boost)


def test_nearby_untried_investigate_hyp_prefers_close_card():
    ex = _executor(question="Where is the table?")
    ex._hypotheses = [
        NavHypothesis(
            phrase="table",
            obs_id=5,
            xyz=np.array([1.0, 0.1, 0.7]),
            score=1.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="table",
            obs_id=9,
            xyz=np.array([8.0, 8.0, 0.7]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    near = ex._nearby_untried_investigate_hyp(max_dist_m=NEAR_INVESTIGATE_M)
    assert near is not None
    assert int(near.obs_id) == 5


def test_fallback_locate_prefers_nearby_investigate():
    ex = _executor(question="Where is the cab?")
    ex._target_phrase = "cab"
    ex._hypotheses = [
        NavHypothesis(
            phrase="cab",
            obs_id=2,
            xyz=np.array([0.5, 0.0, 0.8]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert args["obs_id"] == 2


def test_fallback_close_look_prefers_nearby_investigate():
    ex = _executor(question="What time is it on the wall clock?")
    ex._close_look_required = True
    ex._hypotheses = [
        NavHypothesis(
            phrase="wall clock",
            obs_id=3,
            xyz=np.array([0.6, 0.0, 1.2]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert args["obs_id"] == 3


def test_record_recent_action_includes_nav_outcome():
    ex = _executor()
    ex._round = 1
    ex._record_recent_action(
        "investigate",
        {"obs_id": 4},
        {"ok": True, "obs_id": 4, "nav_outcome": "reached", "verify": {"status": "PRESENT"}},
    )
    assert ex._recent_actions
    assert "nav=reached" in ex._recent_actions[-1]


def test_hypothesize_boost_phrases_prepended():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    real = GraphEQAMemory.__new__(GraphEQAMemory)
    real._observations = []
    real._nodes = []
    real._relevant_objects = []
    real._confirmed_memory_phrases = lambda: []
    real.extract_relevant_objects = lambda _q: None
    real._siglip_match_for_phrase = lambda _p: None
    real._obs_is_frontier = lambda _oid: False
    real._obs_is_object_place = lambda _oid: True
    real._recall_rank_score = GraphEQAMemory._recall_rank_score.__get__(real, GraphEQAMemory)
    real._pack_diversified_hypotheses = GraphEQAMemory._pack_diversified_hypotheses.__get__(real, GraphEQAMemory)
    out = real.hypothesize_nav_targets(
        "Where is the jar?",
        max_k=4,
        boost_phrases=["jar", "counter"],
    )
    assert out == []


def test_prefer_explore_skipped_when_absent_at_non_target_fixture():
    ex = _executor(question="Where is the microwave?")
    ex._target_phrase = "microwave"
    ex._prefer_explore = False
    hyp = NavHypothesis(
        phrase="brick wall",
        obs_id=3,
        xyz=np.array([0.0, 0.0, 0.0]),
        score=1.0,
        source="graph",
    )
    ex._hypotheses = [hyp]
    ex._record_place_inspect(
        3,
        closest_m=0.5,
        verify_out={"status": "ABSENT"},
        approach_index=0,
    )
    assert ex._prefer_explore is False


def _graph_memory_stub() -> GraphEQAMemory:
    gm = GraphEQAMemory.__new__(GraphEQAMemory)
    gm.image_nav_min_approach_m = 0.35
    gm._robot_planar_xy = GraphEQAMemory._robot_planar_xy.__get__(gm, GraphEQAMemory)
    gm._orbit_approach_samples = GraphEQAMemory._orbit_approach_samples.__get__(gm, GraphEQAMemory)
    gm._standoff_waypoint_toward = GraphEQAMemory._standoff_waypoint_toward.__get__(gm, GraphEQAMemory)
    return gm


def test_investigate_target_xyz_synthetic_uses_standoff_not_raw_anchor():
    ex = _executor(question="Where is the counter?")
    oid = -2_000_000
    ex._hypotheses = [
        NavHypothesis(
            phrase="counter",
            obs_id=oid,
            xyz=np.array([2.0, 0.0, 0.9]),
            score=0.0,
            source="siglip",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    ex._voxel_planner = lambda: (None, None)  # type: ignore[method-assign]
    gm = _graph_memory_stub()
    gm._obs_nav_anchor = lambda _oid: None  # type: ignore[method-assign]
    gm._navigation_waypoint_for_obs = lambda _oid, _xyt: None  # type: ignore[method-assign]
    ex.agent.graph_memory = gm

    wp = ex._investigate_target_xyz(oid, 0)
    assert wp is not None
    dist_to_anchor = float(np.hypot(float(wp[0]) - 2.0, float(wp[1]) - 0.0))
    assert 0.1 < dist_to_anchor < 2.0


def test_investigate_target_xyz_falls_through_when_graph_waypoint_none():
    ex = _executor()
    ex._hypotheses = [
        NavHypothesis(
            phrase="table",
            obs_id=12,
            xyz=np.array([3.0, 1.0, 0.8]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    ex._voxel_planner = lambda: (None, None)  # type: ignore[method-assign]
    gm = _graph_memory_stub()
    gm._obs_nav_anchor = lambda oid: np.array([3.0, 1.0, 0.8]) if int(oid) == 12 else None
    gm._navigation_approach_waypoint_for_obs = lambda *_a, **_k: None
    gm._navigation_waypoint_for_obs = lambda *_a, **_k: None
    ex.agent.graph_memory = gm

    wp = ex._investigate_target_xyz(12, 0)
    assert wp is not None
    assert float(np.hypot(float(wp[0]) - 3.0, float(wp[1]) - 1.0)) > 0.1


def test_synthetic_arrival_theta_faces_object_anchor():
    ex = _executor(question="Where is the table?")
    oid = -2_000_000
    ex._hypotheses = [
        NavHypothesis(
            phrase="table",
            obs_id=oid,
            xyz=np.array([2.0, 0.0, 0.9]),
            score=0.0,
            source="siglip",
        ),
    ]
    target = np.array([1.5, 0.0, 1.0])
    look_x, look_y = ex._investigate_arrival_look_at_xy(oid, target)
    assert look_x == 2.0
    assert look_y == 0.0
    theta = float(np.arctan2(look_y - target[1], look_x - target[0]))
    assert abs(theta) < 1e-6


def test_investigate_no_waypoint_marks_approach_tried():
    ex = _executor(question="Where is the table?")
    oid = 41
    hyp = NavHypothesis(
        phrase="table",
        obs_id=oid,
        xyz=np.array([1.0, 0.0, 0.9]),
        score=0.0,
        source="graph",
    )
    ex._hypotheses = [hyp]
    ex.agent.graph_memory = MagicMock()
    ex.agent.navigate_to_target_pose = MagicMock()
    with patch.object(ex, "_investigate_target_xyz", return_value=None):
        out = ex._tool_investigate(oid)
    assert out["status"] == "NO_WAYPOINT"
    assert ex._tried.get(oid) == "no waypoint"
    assert ex._place_inspect[oid].tried_approaches
    assert oid in ex._unreachable_obs_ids
    assert ex._hypothesis_nav_blocked(oid) is True


def test_investigate_annulus_outer_m_scoped():
    assert _executor(question="Where is the cab?")._investigate_annulus_outer_m() == INVESTIGATE_ANNULUS_OUTER_M
    close_look = _executor(question="What time is it on the clock?")
    close_look._close_look_required = True
    assert close_look._investigate_annulus_outer_m() == INVESTIGATE_ANNULUS_OUTER_M
    assert (
        _executor(question="What color is the sofa? A) Red B) Blue")._investigate_annulus_outer_m()
        == DEFAULT_INVESTIGATE_ANNULUS_OUTER_M
    )


def test_investigate_target_xyz_forwards_tight_outer_radius_to_graph_annulus():
    ex = _executor(question="Where is the cab?")
    oid = 12
    ex._hypotheses = [
        NavHypothesis(
            phrase="cab",
            obs_id=oid,
            xyz=np.array([3.0, 1.0, 0.8]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    ex._voxel_planner = lambda: (None, None)  # type: ignore[method-assign]
    gm = _graph_memory_stub()
    gm._obs_nav_anchor = lambda oid_: np.array([3.0, 1.0, 0.8]) if int(oid_) == oid else None
    spy = MagicMock(return_value=np.array([3.2, 1.1, 1.0]))
    gm._navigation_approach_waypoint_for_obs = spy
    gm._navigation_waypoint_for_obs = lambda *_a, **_k: None
    ex.agent.graph_memory = gm

    wp = ex._investigate_target_xyz(oid, 0)
    assert wp is not None
    assert spy.call_args is not None
    assert spy.call_args.kwargs.get("radius_outer_m") == INVESTIGATE_ANNULUS_OUTER_M
