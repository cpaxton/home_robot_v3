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

from emet.controller.habitat_nav import NavOutcome
from emet.mapping.close_map import CloseDistanceMap
from emet.memory.graph_eqa import (
    AgenticEQAExecutor,
    GraphEQAMemory,
    NavHypothesis,
    question_is_locate,
)
from emet.memory.graph_eqa.agentic_eqa import (
    DEFAULT_INVESTIGATE_ANNULUS_OUTER_M,
    INVESTIGATE_ANNULUS_OUTER_M,
    NEAR_INVESTIGATE_M,
)


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
    assert not question_is_locate("Where is the sink? A) kitchen B) bath")
    assert not question_is_locate("Where is the clock? A) Above the sink B) On the wall")


def test_locate_question_prefers_nearby_investigate():
    ex = _executor(question="Where is the microwave?")
    assert ex._prefers_nearby_investigate() is True
    assert ex._hold_detections_before_explore() is True
    other = _executor(question="What color is the sofa? A) Red B) Blue")
    assert other._prefers_nearby_investigate() is False
    assert other._hold_detections_before_explore() is False
    where_mcq = _executor(question="Where is the sink? A) kitchen B) bath")
    assert where_mcq._prefers_nearby_investigate() is False
    assert where_mcq._hold_detections_before_explore() is False
    clock_mcq = _executor(question="What time is it? A) 3 B) 4")
    clock_mcq._close_look_required = True
    assert clock_mcq._prefers_nearby_investigate() is True
    assert clock_mcq._hold_detections_before_explore() is True


def test_recall_nav_hypotheses_boosts_target_phrase_not_gt():
    ex = _executor(question="Where is the microwave?")
    ex._target_phrase = "microwave"
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = []
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm
    ex.agent.voxel_map = None
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


def test_recall_prepends_voxel_localize_over_graph_tv():
    """Gated DynaMem localize_text is an investigate card even when graph has junk nodes."""
    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "red cylinder"
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = [
        NavHypothesis(
            phrase="tv",
            obs_id=6,
            xyz=np.array([1.0, 2.0, 0.0]),
            score=1.0,
            source="graph",
        ),
    ]
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm

    class _Voxel:
        _last_localize_stats = {"query": "red cylinder", "max_cosine": 0.22, "yoloe_hit": True}

        def localize_text(self, text, debug=False, return_debug=False):
            q = str(text or "").strip().lower()
            if "red cylinder" not in q:
                return None
            return np.array([0.08, -0.55, 0.6])

    ex.agent.voxel_map = _Voxel()
    hyps = ex._recall_nav_hypotheses()
    assert hyps[0].source == "voxel"
    assert np.allclose(hyps[0].xyz, [0.08, -0.55, 0.6])
    assert int(hyps[0].obs_id) < 0
    assert hyps[0].confidence == 1.0
    assert hyps[0].yoloe_hit is True
    assert any(h.source == "graph" and int(h.obs_id) == 6 for h in hyps)
    cylinder = [h for h in hyps if h.source == "voxel" and "red cylinder" in str(h.phrase).lower()]
    assert cylinder, "first-hit 3-gram must not skip the object phrase pin"


def test_voxel_localize_skips_fixture_wrap_phrases():
    """A table 3-gram must not become the cylinder card (nopin S0 scored 0/0 that way)."""
    ex = _executor(question="Where is the red cylinder on the table?")
    # Heuristic extract picks the 3-gram first; voxel cards use the object 2-gram.
    ex._target_phrase = "red cylinder table"
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = []
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm

    class _Voxel:
        def __init__(self) -> None:
            self.hits = {
                "red cylinder table": np.array([0.04, -0.55, 0.6]),
                "red cylinder": np.array([0.08, -0.55, 0.6]),
                "blue cube": np.array([-0.02, -0.55, 0.6]),
            }
            self.queries: list[str] = []
            self._last_localize_stats: dict[str, object] = {}

        def localize_text(self, text, debug=False, return_debug=False):
            q = str(text or "").strip().lower()
            self.queries.append(q)
            pt = self.hits.get(q)
            self._last_localize_stats = {
                "query": text,
                "max_cosine": 0.22 if pt is not None else 0.05,
                "yoloe_hit": pt is not None,
            }
            return None if pt is None else pt

    vm = _Voxel()
    ex.agent.voxel_map = vm
    boost = ex._target_boost_phrases()
    assert boost[0] == "red cylinder"
    hyps = ex._voxel_localize_hypotheses()
    phrases = {str(h.phrase).lower() for h in hyps}
    assert "red cylinder" in phrases
    assert "red cylinder table" not in phrases
    assert "red cylinder table" not in vm.queries
    xyz = next(h.xyz for h in hyps if str(h.phrase).lower() == "red cylinder")
    assert np.allclose(xyz, [0.08, -0.55, 0.6])
    assert ex._voxel_score_xyz is not None
    assert np.allclose(ex._voxel_score_xyz, [0.08, -0.55, 0.6])
    assert ex._voxel_score_phrase == "red cylinder"


def test_voxel_hit_alias_pins_object_phrase():
    """Unigram target still pins ``red cylinder`` so scoring does not live-query."""
    from emet.mapping.voxel_localize import pinned_localize_xyz

    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "cylinder"
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = []
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm

    class _Voxel:
        def __init__(self) -> None:
            self.hits = {"cylinder": np.array([0.08, -0.55, 0.6])}
            self._last_localize_stats: dict[str, object] = {}

        def localize_text(self, text, debug=False, return_debug=False):
            q = str(text or "").strip().lower()
            pt = self.hits.get(q)
            self._last_localize_stats = {
                "query": text,
                "max_cosine": 0.22 if pt is not None else 0.05,
                "yoloe_hit": pt is not None,
            }
            return None if pt is None else pt

    vm = _Voxel()
    ex.agent.voxel_map = vm
    hyps = ex._voxel_localize_hypotheses()
    assert hyps
    xyz, stats = pinned_localize_xyz(vm, "red cylinder")
    assert xyz is not None
    assert np.allclose(xyz, [0.08, -0.55, 0.6])
    assert stats["from_pin"] is True
    assert ex._voxel_score_phrase == "red cylinder"


def test_fixture_wrap_hit_is_not_findobj_xyz():
    """Table-blob localize must not be scored as the cylinder."""
    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "red cylinder table"
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = []
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm

    class _Voxel:
        _last_localize_stats: dict[str, object] = {}

        def localize_text(self, text, debug=False, return_debug=False):
            q = str(text or "").strip().lower()
            self._last_localize_stats = {"query": text, "max_cosine": 0.22, "yoloe_hit": True}
            if q == "red cylinder table":
                return np.array([0.04, -0.55, 0.6])
            return None

    ex.agent.voxel_map = _Voxel()
    hyps = ex._voxel_localize_hypotheses()
    assert hyps == []
    assert ex._voxel_score_xyz is None


def test_inspect_graph_returns_query_catalog_without_moving():
    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "red cylinder"
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = []
    gm.get_nodes.return_value = []
    gm.image_description_client = None
    gm.memory_summary_enabled = False
    ex.agent.graph_memory = gm

    class _Voxel:
        _last_localize_stats = {"query": "red cylinder", "max_cosine": 0.22, "yoloe_hit": True}

        def localize_text(self, text, debug=False, return_debug=False):
            return np.array([0.08, -0.55, 0.6])

    voxel = _Voxel()
    cm = CloseDistanceMap(grid_size=(32, 32), origin_xy=(16.0, 16.0), resolution_m=0.1)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (0.08 - 0.30, -0.55, 0.4)
    pose[:3, 2] = (1.0, 0.0, 0.0)
    assert cm.update_from_view(pose, np.array([[0.08, -0.55, 0.4]], dtype=np.float64)) >= 1
    voxel.close_map = cm
    ex.agent.voxel_map = voxel
    gm.hypothesize_nav_targets.return_value = [
        NavHypothesis(
            phrase="tv",
            obs_id=6,
            xyz=np.array([0.0, 1.5, 0.0]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    out = ex._tool_inspect_graph()
    assert out["ok"] is True
    assert out["moved"] is False
    assert out["n_detections"] == 1
    det = out["detections"][0]
    assert det["kind"] == "proposal"
    assert int(det["obs_id"]) < 0
    assert det["confidence"] == 1.0
    assert det["yoloe_hit"] is True
    assert det["close_map"]["resolved"] is True
    assert det["close_map"]["aimed"] is True
    assert det["close_map"]["min_cam_m"] is not None
    assert out["n_views"] == 1
    assert int(out["views"][0]["obs_id"]) == 6
    assert all(int(row["obs_id"]) != 6 for row in out["detections"])


def test_verify_rejects_detection_handle_when_no_camera_frame():
    ex = _executor(question="Where is the red cylinder?")
    ex.agent.graph_memory = MagicMock()
    ex._latest_obs_id = lambda: None  # type: ignore[method-assign]
    out = ex._tool_verify_siglip("red cylinder", -3_000_000)
    assert out["ok"] is False
    assert out["status"] == "NOT_A_VIEW"


def test_stalled_voxel_detection_stays_investigable():
    ex = _executor(question="Where is the red cylinder on the table?")
    hyp = NavHypothesis(
        phrase="red cylinder",
        obs_id=-3_000_000,
        xyz=np.array([0.08, -0.55, 0.6]),
        score=400.0,
        source="voxel",
        confidence=1.0,
    )
    ex._hypotheses = [hyp]
    ex._tried[-3_000_000] = "STALLED_NAV_LOOP verify=REQUIRES_FRESH_VIEW"
    assert ex._obs_already_verified(-3_000_000) is False
    assert ex._hypothesis_nav_blocked(-3_000_000) is False
    nxt = ex._next_untried_hypothesis()
    assert nxt is not None
    assert int(nxt.obs_id) == -3_000_000


def test_siglip_seed_is_not_a_voxel_detection():
    """Ungated SigLIP -2M seeds must not block explore the way localize_text does."""
    ex = _executor(question="Where is the red cylinder on the table?")
    seed = NavHypothesis(
        phrase="siglip red cylinder",
        obs_id=-2_000_000,
        xyz=np.array([0.08, -0.55, 0.6]),
        score=0.0,
        source="siglip",
    )
    ex._hypotheses = [seed]
    gm = MagicMock()
    gm.get_nodes.return_value = [MagicMock(is_frontier=True, is_viewpoint=False)]
    ex.agent.graph_memory = gm
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    assert ex._hypothesis_is_detection(seed) is False
    assert ex._unused_detection_hypothesis() is None
    out = ex._tool_explore_frontier()
    assert out.get("status") != "DETECTIONS_REMAIN"


def test_stalled_voxel_investigate_scores_current_view_without_look_around():
    """Arrival yaw faces the object — do not pan before assess."""
    ex = _executor(question="Where is the red cylinder on the table?")
    oid = -3_000_000
    hyp = NavHypothesis(
        phrase="red cylinder",
        obs_id=oid,
        xyz=np.array([0.08, -0.55, 0.6]),
        score=400.0,
        source="voxel",
        confidence=1.0,
    )
    ex._hypotheses = [hyp]
    ex.agent.graph_memory = MagicMock()
    ex.agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
    ex.agent._last_nav_attempt = None
    events: list[str] = []
    verified: list[dict] = []

    def _look(**_kwargs: object) -> dict:
        events.append("look")
        return {"ok": True}

    def _handle(name: str, args: dict) -> dict:
        if name == "verify_siglip":
            events.append("verify")
            verified.append(dict(args))
            return {"ok": True, "status": "CANDIDATE"}
        return {"ok": True}

    ex.handle_tool = _handle  # type: ignore[method-assign]
    ex._tool_look_around = _look  # type: ignore[method-assign]
    ex._tool_capture_and_update = lambda: {"ok": True}  # type: ignore[method-assign]
    ex._latest_obs_id = lambda: 6  # type: ignore[method-assign]
    ex._robot_xyt = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    ex._investigate_target_xyz = lambda *_a, **_k: np.array([0.5, 0.0, 1.0])  # type: ignore[method-assign]
    rec = MagicMock()
    rec.as_dict.return_value = {}
    rec.coverage = "unknown"
    rec.local_frontier_cells = 0
    rec.card_bits.return_value = "investigated=1"
    rec.approaches_left = 3
    ex._apply_close_map_after_approach = lambda *_a, **_k: rec  # type: ignore[method-assign]
    ex._record_place_inspect = lambda *_a, **_k: rec  # type: ignore[method-assign]
    ex._refresh_place_coverage = lambda *_a, **_k: rec  # type: ignore[method-assign]
    ex._maybe_retract_claim_after_station = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._stamp_room_after_investigate = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._pin_eqa_look_obs = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._refresh_room_after_motion = lambda: None  # type: ignore[method-assign]
    ex._attach_gt = lambda *_a, **_k: None  # type: ignore[method-assign]
    out = ex._tool_investigate(oid)
    assert out["ok"] is True
    assert "verify" in events
    assert events.index("verify") == 0
    assert verified
    assert int(verified[0]["obs_id"]) == 6


def test_fallback_need_more_investigates_voxel_detection_not_frontier():
    """A miss on this RGB must not dump a remaining voxel detection for explore."""
    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "red cylinder"
    gm = MagicMock()
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm
    ex._last_vlm_assess = {"need_more_views": True, "present": False}
    ex._n_nav = 1
    ex._n_explore = 0
    ex._hypotheses = [
        NavHypothesis(
            phrase="red cylinder",
            obs_id=-3_000_000,
            xyz=np.array([20.0, 20.0, 0.6]),
            score=400.0,
            source="voxel",
            confidence=1.0,
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert args["obs_id"] == -3_000_000


def test_fallback_prefers_voxel_detection_over_graph_at_camera_pose():
    """Mapping-view graph XYZ is the camera, not the object (dynagraph method)."""
    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "red cylinder"
    gm = MagicMock()
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm
    ex._hypotheses = [
        NavHypothesis(
            phrase="tv",
            obs_id=6,
            xyz=np.array([0.0, 1.5, 0.0]),
            score=1.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="red cylinder",
            obs_id=-3_000_000,
            xyz=np.array([0.08, -0.55, 0.6]),
            score=400.0,
            source="voxel",
            confidence=1.0,
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    near = ex._nearby_untried_investigate_hyp(max_dist_m=NEAR_INVESTIGATE_M)
    assert near is not None
    assert int(near.obs_id) == -3_000_000
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert args["obs_id"] == -3_000_000
    nxt = ex._next_untried_hypothesis()
    assert nxt is not None
    assert int(nxt.obs_id) == -3_000_000


def test_explore_frontier_refuses_while_detection_unused():
    """explore_frontier is coverage, not a substitute for localize_text XYZ."""
    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "red cylinder"
    gm = MagicMock()
    gm.get_nodes.return_value = [MagicMock(is_frontier=True, is_viewpoint=False)]
    ex.agent.graph_memory = gm
    ex._hypotheses = [
        NavHypothesis(
            phrase="red cylinder",
            obs_id=-3_000_000,
            xyz=np.array([0.08, -0.55, 0.6]),
            score=400.0,
            source="voxel",
            confidence=1.0,
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    out = ex._tool_explore_frontier()
    assert out["ok"] is False
    assert out["status"] == "DETECTIONS_REMAIN"
    assert int(out["obs_id"]) == -3_000_000


def test_mcq_allows_explore_while_detection_unused():
    """HM-EQA must still change rooms; voxel keyword hits are not an OVMM pin."""
    ex = _executor(question="Where is the sink? A) kitchen B) bath")
    view, _cyl = _table_hyps_at_spawn()
    det = NavHypothesis(
        phrase="sink",
        obs_id=-3_000_000,
        xyz=np.array([0.08, -0.55, 0.6]),
        score=400.0,
        source="voxel",
        confidence=1.0,
    )
    gm = MagicMock()
    gm.get_nodes.return_value = [MagicMock(is_frontier=True, is_viewpoint=False)]
    ex.agent.graph_memory = gm
    ex._hypotheses = [view, det]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    out = ex._tool_explore_frontier()
    assert out.get("status") != "DETECTIONS_REMAIN"
    name, _args = ex._fallback_tool()
    assert name == "explore_frontier"


def test_close_look_mcq_holds_detection_before_explore():
    """Clock/count MCQ still needs the close view; do not wander first."""
    ex = _executor(question="What time is it? A) 3 B) 4")
    ex._close_look_required = True
    gm = MagicMock()
    gm.get_nodes.return_value = [MagicMock(is_frontier=True, is_viewpoint=False)]
    ex.agent.graph_memory = gm
    _view, det = _table_hyps_at_spawn()
    det = NavHypothesis(
        phrase="clock",
        obs_id=-3_000_000,
        xyz=np.array([0.08, -0.55, 0.6]),
        score=400.0,
        source="voxel",
        confidence=1.0,
    )
    ex._hypotheses = [det]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    out = ex._tool_explore_frontier()
    assert out["ok"] is False
    assert out["status"] == "DETECTIONS_REMAIN"
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert int(args["obs_id"]) == -3_000_000


def test_mcq_still_refuses_camera_pose_view_when_detection_unused():
    """Camera pose is never the object — same pack for locate and MCQ."""
    ex = _executor(question="Where is the sink? A) kitchen B) bath")
    view, det = _table_hyps_at_spawn()
    ex.agent.graph_memory = MagicMock()
    ex._hypotheses = [view, det]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    out = ex._tool_investigate(6)
    assert out["ok"] is False
    assert out["status"] == "CAMERA_POSE_PLACE"
    assert int(out["redirect_obs_id"]) == -3_000_000


def _table_hyps_at_spawn() -> tuple[NavHypothesis, NavHypothesis]:
    view = NavHypothesis(
        phrase="tv",
        obs_id=6,
        xyz=np.array([0.0, 1.5, 0.0]),
        score=1.0,
        source="graph",
    )
    det = NavHypothesis(
        phrase="red cylinder",
        obs_id=-3_000_000,
        xyz=np.array([0.08, -0.55, 0.6]),
        score=400.0,
        source="voxel",
        confidence=1.0,
    )
    return view, det


def test_investigate_refuses_camera_pose_view_when_detection_unused():
    """Last smoke: VLM picked graph obs_id=6 at the mapping pose."""
    ex = _executor(question="Where is the red cylinder on the table?")
    ex._target_phrase = "red cylinder"
    view, det = _table_hyps_at_spawn()
    ex.agent.graph_memory = MagicMock()
    ex._hypotheses = [view, det]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    out = ex._tool_investigate(6)
    assert out["ok"] is False
    assert out["status"] == "CAMERA_POSE_PLACE"
    assert int(out["redirect_obs_id"]) == -3_000_000


def test_close_map_stay_skips_unapproached_and_camera_pose():
    ex = _executor(question="Where is the red cylinder on the table?")
    view, det = _table_hyps_at_spawn()
    ex._hypotheses = [view, det]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    ex._close_map_unresolved_stay = lambda _oid: True  # type: ignore[method-assign]
    assert ex._close_map_stay_hypothesis() is None
    ex._close_map_attempts[6] = 1
    assert ex._close_map_stay_hypothesis() is None
    ex._close_map_attempts[-3_000_000] = 1
    stay = ex._close_map_stay_hypothesis()
    assert stay is not None
    assert int(stay.obs_id) == -3_000_000


def test_recover_camera_pose_place_investigates_detection():
    ex = _executor(question="Where is the red cylinder on the table?")
    view, det = _table_hyps_at_spawn()
    ex._hypotheses = [view, det]
    ex._robot_xyt_world = lambda: np.array([0.0, 1.5, 0.0])  # type: ignore[method-assign]
    called: list[tuple[str, dict]] = []

    def _handle(name: str, args: dict) -> dict:
        called.append((name, dict(args)))
        return {"ok": True}

    ex.handle_tool = _handle  # type: ignore[method-assign]
    ok = ex._recover_failed_router_motion(
        tool="investigate",
        out={"ok": False, "status": "CAMERA_POSE_PLACE", "obs_id": 6},
    )
    assert ok is True
    assert called == [("investigate", {"obs_id": -3_000_000})]


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


def test_prefer_explore_once_beats_nearby_locate_cards():
    """After a close ABSENT look, grow coverage even if unused graph views sit nearby.

    Stretch find8 never called explore_frontier during find because locate hold
    plus ~150 mapping views within 3.5 m always won.
    """
    ex = _executor(question="Where is the jar on the counter?")
    ex._target_phrase = "jar"
    ex._prefer_explore = True
    ex._prefer_explore_reason = "absent"
    ex._n_consecutive_explore = 0
    gm = MagicMock()
    gm.get_nodes.return_value = [MagicMock(is_frontier=True, is_viewpoint=False)]
    ex.agent.graph_memory = gm
    ex._hypotheses = [
        NavHypothesis(
            phrase="cabinet",
            obs_id=12,
            xyz=np.array([0.4, 0.1, 0.5]),
            score=1.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="unexplored frontier",
            obs_id=99,
            xyz=np.array([3.0, 0.0, 0.0]),
            score=0.2,
            source="frontier",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    near = ex._nearby_untried_investigate_hyp(max_dist_m=NEAR_INVESTIGATE_M)
    assert near is not None
    assert int(near.obs_id) == 12
    name, _args = ex._fallback_tool()
    assert name == "explore_frontier"
    ex._n_consecutive_explore = 1
    name2, args2 = ex._fallback_tool()
    assert name2 == "investigate"
    assert int(args2["obs_id"]) == 12


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


def test_count_find_obs_ids_uses_shared_helper_not_spawn_visual():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["stove", "kitchen cabinets"])
    mem.add_observation(rgb, np.array([0.16, -2.54, 0.05]), ["bed"])
    mem._relevant_objects = ["bedside tables", "bedroom white bedding"]
    mem._relevant_phrases = ["bedside tables", "bedroom white bedding"]
    mem.set_visual_find_fn(lambda phrase, max_n: [(0.9, 1)])
    q = "How many bedside tables are there in the bedroom? A) One B) Two C) Three D) None. Answer:"
    mem._question = q
    ex = _executor(question=q)
    ex.agent.graph_memory = mem
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    ids = ex._count_find_obs_ids()
    assert 2 in ids
    assert 1 not in ids
    mem.last_eqa_obs_ids = [1]
    assert ex._count_find_unattached_obs_ids() == [oid for oid in ids if oid != 1]
    assert ex._downgrade_unattached_count_none("Two", True) is False
    assert mem.last_eqa_action_obs_id == 2


def test_inspect_graph_views_prefer_object_place_over_camera_pose():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(
        rgb,
        np.array([0.0, 0.0, 0.5]),
        ["microwave"],
        identity_key="near",
    )
    mem.add_observation(
        rgb,
        np.array([4.0, 1.0, 0.5]),
        ["microwave"],
        identity_key="far",
    )
    mem._relevant_objects = ["microwave"]
    mem._relevant_phrases = ["microwave"]
    ex = _executor(question="Where is the microwave?")
    ex.agent.graph_memory = mem
    ex.agent.voxel_map = None
    ex._target_phrase = "microwave"
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    out = ex._tool_inspect_graph()
    view_ids = [int(r["obs_id"]) for r in out["views"]]
    assert 2 in view_ids
    if 1 in view_ids:
        assert view_ids.index(2) < view_ids.index(1)
