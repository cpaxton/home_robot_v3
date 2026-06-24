# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

from emet.eval.dynamic_exploration_runner import count_object_nodes


@dataclass
class _FakeNode:
    labels: list[str] = field(default_factory=list)
    is_viewpoint: bool = False


class _FakeMemory:
    def __init__(self, nodes: list[_FakeNode]) -> None:
        self._nodes = nodes

    def get_nodes(self) -> list[_FakeNode]:
        return list(self._nodes)


def test_count_object_nodes_no_hint_excludes_viewpoints():
    mem = _FakeMemory(
        [
            _FakeNode(labels=["mug"]),
            _FakeNode(labels=["apple"], is_viewpoint=True),
            _FakeNode(labels=["vase"]),
        ]
    )
    assert count_object_nodes(mem) == 2


def test_count_object_nodes_label_hint_matches_any_label():
    mem = _FakeMemory(
        [
            _FakeNode(labels=["red_mug", "kitchen"]),
            _FakeNode(labels=["apple"]),
            _FakeNode(labels=["obj_main_variant"]),
        ]
    )
    assert count_object_nodes(mem, label_hint="mug") == 1
    assert count_object_nodes(mem, label_hint="obj_main") == 1
    assert count_object_nodes(mem, label_hint="OBJ_MAIN") == 1


def test_count_object_nodes_label_hint_no_match():
    mem = _FakeMemory([_FakeNode(labels=["apple"]), _FakeNode(labels=["vase"])])
    assert count_object_nodes(mem, label_hint="mug") == 0


def test_count_object_nodes_none_memory():
    assert count_object_nodes(None) == 0
