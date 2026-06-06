# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

from types import SimpleNamespace

from emet.visualization.rerun import _color_for_graph_label, _dynagraph_node_color


def test_same_class_same_color():
    a = _color_for_graph_label("sink")
    b = _color_for_graph_label("sink")
    assert a == b


def test_different_classes_differ():
    a = _color_for_graph_label("sink")
    b = _color_for_graph_label("microwave")
    assert a != b


def test_dynagraph_node_uses_label_class():
    node = SimpleNamespace(
        node_id=1,
        labels=["Refrigerator"],
        obs_id=1,
        is_viewpoint=False,
        bbox_xyxy=None,
        xyz=[0, 0, 0],
    )
    c = _dynagraph_node_color(node, None)
    assert c == _color_for_graph_label("refrigerator")
