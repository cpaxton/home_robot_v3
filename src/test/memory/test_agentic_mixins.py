# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU-only guards for the AgenticEQAExecutor facade (size + public methods)."""

from pathlib import Path
from unittest.mock import MagicMock

from emet.memory.graph_eqa import agentic_eqa
from emet.memory.graph_eqa.agentic.executor_init import TOOL_HANDLERS
from emet.memory.graph_eqa.agentic_eqa import (
    NAV_SAME_OBS_LOOP_LIMIT,
    PLACE_APPROACH_SAMPLES,
    AgenticEQAExecutor,
    AgenticEQAResult,
    AgenticSession,
    AgenticState,
    EvidencePhase,
)

# Keep the facade ingestible: loading a 7k-line module crashes the agent host.
_FACADE_MAX_LINES = 250
_MODULE_MAX_LINES = 800


def test_facade_still_reexports_split_symbols() -> None:
    assert AgenticEQAResult.__name__ == "AgenticEQAResult"
    assert AgenticState is EvidencePhase
    assert NAV_SAME_OBS_LOOP_LIMIT >= 1
    assert PLACE_APPROACH_SAMPLES >= 1
    assert hasattr(AgenticEQAExecutor, "__init__")
    assert hasattr(AgenticEQAExecutor, "handle_tool")
    assert hasattr(AgenticEQAExecutor, "_prefers_nearby_investigate")
    assert hasattr(AgenticEQAExecutor, "_tool_explore_frontier")
    assert hasattr(AgenticEQAExecutor, "_tool_investigate")
    assert hasattr(AgenticEQAExecutor, "_place_anchor_xy")
    assert hasattr(AgenticEQAExecutor, "_decide_close_look")
    assert hasattr(AgenticEQAExecutor, "_action_gate_decision")
    assert hasattr(AgenticEQAExecutor, "_save_frontier_pick_panel")
    assert hasattr(AgenticEQAExecutor, "run")
    assert hasattr(AgenticEQAExecutor, "_route_tool_calls")
    assert "_FIXTURE_LABEL_TOKENS" in AgenticEQAExecutor.__dict__ or hasattr(
        AgenticEQAExecutor, "_FIXTURE_LABEL_TOKENS"
    )


def test_tool_handlers_cover_eqa_names() -> None:
    for name in (
        "inspect_graph",
        "explore_frontier",
        "investigate",
        "navigate_to_obs",
        "look_around",
        "capture_and_update",
        "verify_siglip",
        "submit_answer",
        "finish",
    ):
        assert name in TOOL_HANDLERS
    assert TOOL_HANDLERS["investigate"] is TOOL_HANDLERS["navigate_to_obs"]


def test_executor_init_owns_session() -> None:
    agent = MagicMock()
    agent.parameters = {}
    agent.robot = MagicMock()
    ex = AgenticEQAExecutor(agent, "where is the mug?", router=False)
    assert isinstance(ex.session, AgenticSession)
    assert ex.session is object.__getattribute__(ex, "session")
    ex._tried = {3: "x"}
    assert ex.session._tried == {3: "x"}
    assert ex._tried == {3: "x"}


def test_facade_stays_small() -> None:
    n = sum(1 for _ in Path(agentic_eqa.__file__).open(encoding="utf-8"))
    assert n <= _FACADE_MAX_LINES, f"agentic_eqa.py is {n} lines; keep it a thin facade"


def test_investigate_and_place_modules_stay_under_limit() -> None:
    from emet.memory.graph_eqa.agentic import investigate, place

    for mod in (investigate, place):
        n = sum(1 for _ in Path(mod.__file__).open(encoding="utf-8"))
        assert n <= _MODULE_MAX_LINES, f"{Path(mod.__file__).name} is {n} lines; split further"


def test_compat_shims_share_tool_handlers() -> None:
    from emet.memory.graph_eqa.agentic_init import TOOL_HANDLERS as old

    assert old is TOOL_HANDLERS
    assert old["investigate"] is TOOL_HANDLERS["navigate_to_obs"]


def test_voxel_planner_uses_get_voxel_map_when_attr_has_no_localize() -> None:
    class _Occupancy:
        pass

    class _Semantic:
        def localize_text(self, *args, **kwargs):
            return None

    semantic = _Semantic()

    class _Agent:
        voxel_map = _Occupancy()
        planner = object()
        parameters = {}
        robot = MagicMock()

        def get_voxel_map(self):
            return semantic

    ex = AgenticEQAExecutor(_Agent(), "where is the mug?", router=False)
    vm, planner = ex._voxel_planner()
    assert vm is semantic
    assert planner is _Agent.planner
