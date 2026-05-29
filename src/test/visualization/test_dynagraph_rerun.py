# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.visualization.rerun import (
    _color_for_graph_label,
    _mosaic_labeled_images,
    _obs_rgb_by_id,
    _safe_rerun_path_component,
    build_dynagraph_gallery_markdown,
    dynagraph_crop_entity_path,
    format_dynagraph_node_label,
)


def test_format_dynagraph_node_label_includes_obs_id():
    n = SimpleNamespace(node_id=2, labels=["mug"], obs_id=7, support_count=1)
    assert format_dynagraph_node_label(n) == "mug [#2 img 7]"


def test_safe_rerun_path_component():
    assert _safe_rerun_path_component("wooden table") == "wooden_table"
    assert "/" not in _safe_rerun_path_component("a/b")


def test_obs_rgb_by_id_from_graph_memory():
    gm = SimpleNamespace(
        get_observations=lambda: [
            SimpleNamespace(obs_id=1, rgb=np.zeros((4, 4, 3), dtype=np.uint8)),
            SimpleNamespace(obs_id=3, rgb=np.ones((2, 2, 3), dtype=np.uint8)),
        ]
    )
    m = _obs_rgb_by_id(gm)
    assert set(m.keys()) == {1, 3}
    assert m[1].shape == (4, 4, 3)


def test_dynagraph_crop_entity_path():
    n = SimpleNamespace(node_id=2, labels=["wooden table"])
    assert dynagraph_crop_entity_path(n) == "world/dynagraph/crops/002_wooden_table"


def test_build_dynagraph_gallery_markdown_links():
    nodes = [
        SimpleNamespace(
            node_id=1,
            labels=["mug"],
            obs_id=3,
            xyz=[1.0, 2.0, 0.5],
            support_count=1,
            description=None,
        ),
    ]
    md = build_dynagraph_gallery_markdown(nodes, has_crop_images=True)
    assert "recording://world/dynagraph/crops/001_mug" in md
    assert "| 1 | [mug]" in md
    assert "**img** 3" in md


def test_color_for_graph_label_stable():
    assert _color_for_graph_label("mug") == _color_for_graph_label("mug")
    assert _color_for_graph_label("mug") != _color_for_graph_label("cup")


def test_mosaic_labeled_images_nonempty():
    entries = [
        ("#1 img1 mug", np.full((8, 12, 3), 200, dtype=np.uint8)),
        ("#2 img2 cup", np.full((8, 12, 3), 50, dtype=np.uint8)),
    ]
    mosaic = _mosaic_labeled_images(entries, cols=2, thumb_max=32)
    assert mosaic is not None
    assert mosaic.ndim == 3 and mosaic.shape[2] == 3
