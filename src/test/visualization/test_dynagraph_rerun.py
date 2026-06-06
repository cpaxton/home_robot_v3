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
    crop_rgb_bbox_xyxy,
    dynagraph_crop_entity_path,
    dynagraph_node_rgb_crop,
    format_dynagraph_node_label,
    node_has_detection_crop,
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
            bbox_xyxy=(10, 10, 40, 40),
        ),
    ]
    md = build_dynagraph_gallery_markdown(nodes, has_crop_images=True)
    assert "recording://world/dynagraph/crops/001_mug" in md
    assert "| 1 | [mug]" in md
    assert "**img** 3" in md


def test_color_for_graph_label_stable():
    assert _color_for_graph_label("mug") == _color_for_graph_label("mug")
    assert _color_for_graph_label("mug") != _color_for_graph_label("cup")


def test_crop_rgb_bbox_xyxy_crops_instance_region():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[2:6, 3:9] = 255
    crop = crop_rgb_bbox_xyxy(rgb, (3, 2, 9, 6), padding_frac=0.0)
    assert crop.shape[0] == 4 and crop.shape[1] == 6
    assert crop.max() == 255


def test_dynagraph_node_rgb_crop_uses_bbox_when_present():
    frame = np.arange(60, dtype=np.uint8).reshape(5, 4, 3)
    node = SimpleNamespace(obs_id=1, bbox_xyxy=(1, 1, 3, 3))
    obs_rgb = {1: frame}
    crop = dynagraph_node_rgb_crop(node, obs_rgb)
    assert crop.shape[0] == 2 and crop.shape[1] == 2


def test_dynagraph_node_rgb_crop_none_without_bbox():
    frame = np.ones((4, 4, 3), dtype=np.uint8) * 7
    node = SimpleNamespace(obs_id=2, bbox_xyxy=None)
    assert dynagraph_node_rgb_crop(node, {2: frame}) is None


def test_node_has_detection_crop_rejects_full_frame_bbox():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    node = SimpleNamespace(obs_id=1, bbox_xyxy=(0, 0, 100, 100))
    assert not node_has_detection_crop(node, {1: frame})
    node2 = SimpleNamespace(obs_id=1, bbox_xyxy=(10, 10, 40, 50))
    assert node_has_detection_crop(node2, {1: frame})


def test_mosaic_labeled_images_nonempty():
    entries = [
        ("#1 img1 mug", np.full((8, 12, 3), 200, dtype=np.uint8)),
        ("#2 img2 cup", np.full((8, 12, 3), 50, dtype=np.uint8)),
    ]
    mosaic = _mosaic_labeled_images(entries, cols=2, thumb_max=32)
    assert mosaic is not None
    assert mosaic.ndim == 3 and mosaic.shape[2] == 3
