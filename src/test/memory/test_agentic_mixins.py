# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU-only guards for the agentic_eqa.py mixin split (facade re-exports + MRO)."""

from pathlib import Path

from emet.memory.graph_eqa import agentic_eqa
from emet.memory.graph_eqa.agentic_action import AgenticActionMixin
from emet.memory.graph_eqa.agentic_answer import AgenticAnswerMixin
from emet.memory.graph_eqa.agentic_assess import AgenticAssessMixin
from emet.memory.graph_eqa.agentic_capture import AgenticCaptureMixin
from emet.memory.graph_eqa.agentic_eqa import (
    NAV_SAME_OBS_LOOP_LIMIT,
    PLACE_APPROACH_SAMPLES,
    AgenticEQAExecutor,
    AgenticEQAResult,
)
from emet.memory.graph_eqa.agentic_explore import AgenticExploreMixin
from emet.memory.graph_eqa.agentic_init import AgenticInitMixin
from emet.memory.graph_eqa.agentic_investigate import AgenticInvestigateMixin
from emet.memory.graph_eqa.agentic_place import AgenticPlaceMixin
from emet.memory.graph_eqa.agentic_router import AgenticRouterMixin
from emet.memory.graph_eqa.agentic_run import AgenticRunMixin
from emet.memory.graph_eqa.agentic_verify import AgenticVerifyMixin

# Keep the facade ingestible: loading a 7k-line module crashes the agent host.
_FACADE_MAX_LINES = 250
_MIXIN_MAX_LINES = 800


def test_executor_mro_includes_action_and_explore_mixins() -> None:
    mro = AgenticEQAExecutor.__mro__
    expected = (
        AgenticInitMixin,
        AgenticRunMixin,
        AgenticRouterMixin,
        AgenticAnswerMixin,
        AgenticVerifyMixin,
        AgenticAssessMixin,
        AgenticCaptureMixin,
        AgenticInvestigateMixin,
        AgenticPlaceMixin,
        AgenticExploreMixin,
        AgenticActionMixin,
    )
    for mixin in expected:
        assert mixin in mro
    indexes = [mro.index(mixin) for mixin in expected]
    assert indexes == sorted(indexes)


def test_facade_still_reexports_split_symbols() -> None:
    assert AgenticEQAResult.__name__ == "AgenticEQAResult"
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


def test_facade_stays_small() -> None:
    n = sum(1 for _ in Path(agentic_eqa.__file__).open(encoding="utf-8"))
    assert n <= _FACADE_MAX_LINES, f"agentic_eqa.py is {n} lines; keep it a thin facade"


def test_investigate_and_place_mixins_stay_under_limit() -> None:
    from emet.memory.graph_eqa import agentic_investigate, agentic_place

    for mod in (agentic_investigate, agentic_place):
        n = sum(1 for _ in Path(mod.__file__).open(encoding="utf-8"))
        assert n <= _MIXIN_MAX_LINES, f"{Path(mod.__file__).name} is {n} lines; split further"
