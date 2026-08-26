# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU-only guards for the graph_memory.py mixin split (facade re-exports + MRO)."""

from pathlib import Path

from emet.memory.graph_eqa import graph_memory
from emet.memory.graph_eqa.graph_answer import GraphAnswerMixin
from emet.memory.graph_eqa.graph_eqa_obs import GraphEqaObsMixin
from emet.memory.graph_eqa.graph_hypotheses import GraphHypothesisMixin
from emet.memory.graph_eqa.graph_init import GraphInitMixin
from emet.memory.graph_eqa.graph_memory import (
    SIGLIP_PRESENT_THRESHOLD,
    GraphEQAMemory,
    GraphNode,
    NavHypothesis,
    _near,
    replace,
)
from emet.memory.graph_eqa.graph_mutate import GraphMutateMixin
from emet.memory.graph_eqa.graph_nav import GraphNavMixin
from emet.memory.graph_eqa.graph_prompt import GraphPromptMixin
from emet.memory.graph_eqa.graph_rooms import GraphRoomsMixin

# Keep the facade ingestible: loading a 7k-line module crashes the agent host.
_FACADE_MAX_LINES = 250


def test_memory_mro_includes_split_mixins() -> None:
    mro = GraphEQAMemory.__mro__
    expected = (
        GraphInitMixin,
        GraphMutateMixin,
        GraphRoomsMixin,
        GraphEqaObsMixin,
        GraphHypothesisMixin,
        GraphPromptMixin,
        GraphNavMixin,
        GraphAnswerMixin,
    )
    for mixin in expected:
        assert mixin in mro
    indexes = [mro.index(mixin) for mixin in expected]
    assert indexes == sorted(indexes)


def test_facade_still_reexports_split_symbols() -> None:
    assert GraphNode.__name__ == "GraphNode"
    assert NavHypothesis.__name__ == "NavHypothesis"
    assert SIGLIP_PRESENT_THRESHOLD > 0
    assert callable(_near)
    assert replace is not None
    assert hasattr(GraphEQAMemory, "add_observation")
    assert hasattr(GraphEQAMemory, "query_answer")
    assert hasattr(GraphEQAMemory, "hypothesize_nav_targets")
    assert hasattr(GraphEQAMemory, "stamp_vlm_room_at_robot")
    assert hasattr(GraphEQAMemory, "sync_frontier_nodes")


def test_graph_memory_facade_stays_small() -> None:
    n = sum(1 for _ in Path(graph_memory.__file__).open(encoding="utf-8"))
    assert n <= _FACADE_MAX_LINES, f"graph_memory.py is {n} lines; keep it a thin facade"
