# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""describe_scene prefers VLM / memory — not low-conf YoloE dumps."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from emet.controller.controller_dynamem import _DESCRIBE_SCENE_YOLOE_LABELS, DynamemController


def _agent(**params: object) -> DynamemController:
    agent = DynamemController.__new__(DynamemController)
    agent.parameters = dict(params) if params else {}
    agent.robot = SimpleNamespace(
        get_observation=lambda: SimpleNamespace(
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            depth=None,
        )
    )
    agent.detection_model = None
    agent.get_voxel_map = lambda: None
    return agent


def test_describe_scene_combines_memory_and_vlm():
    agent = _agent(detection={"describe_use_detector_fallback": True})
    vm = SimpleNamespace(image_description_client=MagicMock(), eqa_client=None)
    agent.get_voxel_map = lambda: vm
    agent._describe_scene_vlm = MagicMock(return_value="From my head camera: a table and a chair.")
    agent._describe_scene_from_memory = MagicMock(return_value="From my map/scene graph I also know about: cup.")
    agent._describe_scene_yoloe = MagicMock(return_value="From my head camera I can make out: inhaler.")
    text = agent.describe_head_camera_scene_text()
    # Caption first, then graph grounding.
    assert text.index("table and a chair") < text.index("cup")
    assert "cup" in text
    agent._describe_scene_from_memory.assert_called_once()
    agent._describe_scene_vlm.assert_called_once()
    agent._describe_scene_yoloe.assert_not_called()


def test_describe_scene_uses_memory_when_no_vlm():
    agent = _agent()
    agent._describe_scene_vlm = MagicMock(return_value=None)
    agent._describe_scene_from_memory = MagicMock(return_value="From my map/scene graph I also know about: sofa.")
    text = agent.describe_head_camera_scene_text(graph_memory=object())
    assert "sofa" in text
    assert "scene graph" in text.lower() or "map" in text.lower()


def test_describe_scene_fallback_without_vlm_or_memory():
    agent = _agent(detection={"describe_use_detector_fallback": False})
    agent._describe_scene_vlm = MagicMock(return_value=None)
    agent._describe_scene_from_memory = MagicMock(return_value=None)
    agent.announce_action = MagicMock()
    agent.look_around = MagicMock()
    agent.rotate_in_place = MagicMock()
    agent.run_exploration = MagicMock()
    text = agent.describe_head_camera_scene_text()
    # "what can you see" must not auto-scan — caption/current view only.
    agent.look_around.assert_not_called()
    agent.rotate_in_place.assert_not_called()
    agent.run_exploration.assert_not_called()
    assert "photo" in text.lower() or "front" in text.lower()
    assert "--eqa" not in text
    assert "YoloE" not in text


def test_describe_scene_skips_explore_when_memory_works():
    agent = _agent()
    agent._describe_scene_vlm = MagicMock(return_value=None)
    agent._describe_scene_from_memory = MagicMock(return_value="From my map/scene graph I also know about: sofa.")
    agent.look_around = MagicMock()
    agent.run_exploration = MagicMock()
    text = agent.describe_head_camera_scene_text()
    assert "sofa" in text
    agent.look_around.assert_not_called()
    agent.run_exploration.assert_not_called()


def test_pick_interesting_scene_image_prefers_graph_crop():
    agent = _agent()
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[10:30, 20:40] = 200
    obs = SimpleNamespace(obs_id=1, rgb=rgb)
    node = SimpleNamespace(
        is_viewpoint=False,
        is_frontier=False,
        obs_id=1,
        bbox_xyxy=(20, 10, 40, 30),
        labels=["mug"],
        support_count=3,
    )
    gm = SimpleNamespace(get_nodes=lambda: [node], get_observations=lambda: [obs])
    live = np.ones((32, 32, 3), dtype=np.uint8) * 10
    image, label = agent.pick_interesting_scene_image(graph_memory=gm, live_rgb=live)
    assert label == "mug"
    assert image is not None
    assert image.shape[0] < 40 and image.shape[1] < 60
    assert int(image.mean()) > 100


def test_pick_interesting_scene_image_falls_back_to_live():
    agent = _agent()
    live = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    image, label = agent.pick_interesting_scene_image(graph_memory=None, live_rgb=live)
    assert label is None
    assert image is not None
    np.testing.assert_array_equal(image, live)


def test_describe_scene_yoloe_fallback_only_when_enabled():
    agent = _agent(
        detection={
            "describe_use_detector_fallback": True,
            "describe_confidence_threshold": 0.30,
            "describe_use_curated_vocab": True,
        }
    )
    dm = MagicMock()
    dm.class_list = ["inhaler", "bathroom stall", "adapter"]
    table_i = list(_DESCRIBE_SCENE_YOLOE_LABELS).index("table")
    dm.predict.return_value = (
        None,
        None,
        {
            "instance_classes": np.array([table_i]),
            "instance_scores": np.array([0.55]),
        },
    )
    agent.detection_model = dm
    agent._describe_scene_vlm = MagicMock(return_value=None)
    agent._describe_scene_from_memory = MagicMock(return_value=None)
    # Bypass isinstance checks by calling yoloe helper directly (unit-tested path).
    text = agent._describe_scene_yoloe(np.zeros((32, 32, 3), dtype=np.uint8), None, dm)
    assert "table" in text
    assert "inhaler" not in text
