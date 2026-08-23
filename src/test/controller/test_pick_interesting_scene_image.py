# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Crop usability gate and pick_interesting_scene_image fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.agent.camera_debug import rgb_frame_is_usable
from emet.controller.controller_dynamem import DynamemController


def test_rgb_frame_is_usable_rejects_white_and_black():
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    ok = np.zeros((32, 32, 3), dtype=np.uint8)
    ok[8:24, 8:24] = (40, 120, 200)
    assert not rgb_frame_is_usable(white)
    assert not rgb_frame_is_usable(black)
    assert rgb_frame_is_usable(ok)


def test_pick_interesting_rejects_white_crop_falls_back_to_live():
    agent = DynamemController.__new__(DynamemController)
    live = np.zeros((48, 64, 3), dtype=np.uint8)
    live[10:40, 10:50] = (30, 90, 160)

    # Entire observation white so any padded crop stays unusable.
    white_obs = np.full((80, 80, 3), 255, dtype=np.uint8)
    node = SimpleNamespace(
        is_viewpoint=False,
        is_frontier=False,
        bbox_xyxy=(10, 10, 30, 30),
        obs_id=0,
        support_count=3,
        labels=["chair"],
        node_id=1,
    )
    obs = SimpleNamespace(obs_id=0, rgb=white_obs)
    gm = SimpleNamespace(
        get_nodes=lambda: [node],
        get_observations=lambda: [obs],
    )
    img, label = agent.pick_interesting_scene_image(graph_memory=gm, live_rgb=live)
    assert label is None
    assert img is not None
    assert int(img.mean()) < 200


def test_pick_interesting_skips_generic_object_label():
    agent = DynamemController.__new__(DynamemController)
    live = np.zeros((48, 64, 3), dtype=np.uint8)
    live[10:40, 10:50] = (30, 90, 160)
    obs_rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    obs_rgb[10:34, 10:34] = (50, 140, 80)
    node = SimpleNamespace(
        is_viewpoint=False,
        is_frontier=False,
        bbox_xyxy=(10, 10, 34, 34),
        obs_id=0,
        support_count=2,
        labels=["object"],
        node_id=2,
    )
    obs = SimpleNamespace(obs_id=0, rgb=obs_rgb)
    gm = SimpleNamespace(get_nodes=lambda: [node], get_observations=lambda: [obs])
    img, label = agent.pick_interesting_scene_image(graph_memory=gm, live_rgb=live)
    assert label is None
    assert img is not None
    np.testing.assert_array_equal(img, live)


def test_pick_interesting_uses_named_usable_crop():
    agent = DynamemController.__new__(DynamemController)
    live = np.zeros((48, 64, 3), dtype=np.uint8)
    live[:] = 10
    obs_rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    obs_rgb[10:34, 10:34] = (50, 140, 80)
    node = SimpleNamespace(
        is_viewpoint=False,
        is_frontier=False,
        bbox_xyxy=(10, 10, 34, 34),
        obs_id=0,
        support_count=2,
        labels=["mug"],
        node_id=3,
    )
    obs = SimpleNamespace(obs_id=0, rgb=obs_rgb)
    gm = SimpleNamespace(get_nodes=lambda: [node], get_observations=lambda: [obs])
    img, label = agent.pick_interesting_scene_image(graph_memory=gm, live_rgb=live)
    assert label == "mug"
    assert img is not None
    assert img.shape[0] < live.shape[0] or img.shape[1] < live.shape[1]


def test_agent_skill_tools_include_core_set():
    from emet.agent.tools import get_tools

    names = {t.name for t in get_tools({})}
    for required in (
        "describe_scene",
        "explore",
        "find_objects",
        "scan_environment",
        "send_image",
        "send_map_snapshot",
        "query_memory",
    ):
        assert required in names
