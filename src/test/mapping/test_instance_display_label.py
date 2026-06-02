"""Labels for Rerun / scene graph use detector and graph class names."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from emet.mapping.instance.instance import (
    Instance,
    graph_label_for_instance_xyz,
    instance_display_label,
)
from emet.visualization.rerun import format_dynagraph_node_label, graph_node_primary_label


class _FakeDetector:
    class_list = ["sink", "microwave", "cabinet"]


@dataclass
class _FakeGraphNode:
    node_id: int
    xyz: tuple[float, float, float]
    labels: list[str] = field(default_factory=list)
    is_viewpoint: bool = False


class _FakeGraphMemory:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes(self):
        return list(self._nodes)


def _instance_with_category(cid: int) -> Instance:
    return Instance(global_id=7, category_id=cid)


def test_instance_display_label_uses_detector_class_list():
    inst = _instance_with_category(0)
    assert instance_display_label(inst, detection_model=_FakeDetector()) == "sink"


def test_instance_display_label_prefers_graph_neighbor():
    inst = _instance_with_category(99)
    inst.point_cloud = torch.tensor([[1.0, 2.0, 0.5], [1.1, 2.1, 0.5]])
    graph = _FakeGraphMemory(
        [
            _FakeGraphNode(1, (1.05, 2.05, 0.5), labels=["kitchen_sink"]),
            _FakeGraphNode(2, (5.0, 5.0, 0.0), labels=["table"]),
        ]
    )
    assert instance_display_label(inst, detection_model=None, graph_memory=graph) == "kitchen_sink"


def test_graph_node_primary_label_skips_obj_placeholder():
    node = _FakeGraphNode(3, (0, 0, 0), labels=["obj_12", "faucet"])
    assert graph_node_primary_label(node) == "faucet"
    assert "faucet" in format_dynagraph_node_label(node)


def test_graph_label_for_instance_xyz_match():
    graph = _FakeGraphMemory([_FakeGraphNode(1, (0.0, 0.0, 0.0), labels=["chair"])])
    assert graph_label_for_instance_xyz(np.array([0.1, 0.05, 0.0]), graph) == "chair"
    assert graph_label_for_instance_xyz(np.array([3.0, 3.0, 0.0]), graph) is None
