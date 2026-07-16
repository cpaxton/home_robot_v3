# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for scene-aware Dynagraph label filtering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from emet.memory.graph_eqa.dynamem_graph_hooks import update_graph_memory_from_dynamem_observation
from emet.memory.graph_eqa.graph_label_filter import (
    filter_graph_labels,
    is_graph_label_allowed,
    resolve_graph_scene_profile,
)
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphObjectFusion
from emet.memory.graph_eqa.graph_observation_pipeline import apply_instance_items_to_graph


def test_kitchen_denies_bathroom_stall_allows_range_hood():
    assert not is_graph_label_allowed("bathroom stall", scene_profile="kitchen")
    assert not is_graph_label_allowed("Toilet Paper Dispenser", scene_profile="kitchen")
    assert is_graph_label_allowed("range hood", scene_profile="kitchen")
    assert is_graph_label_allowed("cabinet", scene_profile="kitchen")
    # Habitat homes keep bathroom labels.
    assert is_graph_label_allowed("bathroom stall", scene_profile="indoor")


def test_filter_graph_labels_kitchen():
    out = filter_graph_labels(
        ["range hood", "bathroom stall", "object", "cabinet"],
        scene_profile="kitchen",
    )
    assert out == ["range hood", "cabinet"]


def test_resolve_profile_from_robocasa_session():
    robot = MagicMock()
    robot.get_emet_session.return_value = {"environment": {"kind": "robocasa"}}
    assert resolve_graph_scene_profile(robot=robot) == "kitchen"


def test_resolve_profile_explicit_none():
    params = {"graph_eqa_label_filter": {"scene_profile": "none"}}
    assert resolve_graph_scene_profile(parameters=params) == "none"
    assert is_graph_label_allowed("bathroom stall", scene_profile="none")


def test_apply_instance_items_drops_kitchen_junk():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.parameters = {"graph_eqa_label_filter": {"scene_profile": "kitchen"}}
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    apply_instance_items_to_graph(
        mem,
        rgb,
        [
            ("bathroom stall", np.array([1.0, 0.0, 0.5]), (0, 0, 2, 2)),
            ("range hood", np.array([2.0, 0.0, 1.0]), (2, 2, 4, 4)),
        ],
        dedup_skips=lambda _l, _x: False,
        scene_profile="kitchen",
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 1
    assert "range hood" in " ".join(objs[0].labels).lower()


def test_vlm_hook_filters_kitchen_bathroom_labels():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.55,
        embedding_min_cosine=0.0,
        fallback_spatial_merge_xy_m=0.55,
        require_label_match=False,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.parameters = {"graph_eqa_label_filter": {"scene_profile": "kitchen"}}
    mem.spatial_merge_m = 0.0

    sensor_builder = MagicMock()
    sensor_builder.labels_and_description_from_observation.return_value = (
        ["bathroom stall", "range hood"],
        "kitchen",
    )
    sensor_builder.world_xyz_for_observation.return_value = np.array([1.0, 0.5, 0.9])

    robot = MagicMock()
    robot.get_emet_session.return_value = {"environment": {"kind": "robocasa"}}
    obs = MagicMock()
    obs.rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    obs.camera_pose = np.eye(4, dtype=np.float64)
    obs.camera_pose[2, 3] = 1.2
    obs.semantic = None
    obs.depth = None
    obs.gps = None
    obs.compass = None
    obs.navigation_origin_xyt = None

    vm = SimpleNamespace(observations=[], image_descriptions=[])
    update_graph_memory_from_dynamem_observation(
        graph_memory=mem,
        robot=robot,
        voxel_map=vm,
        detection_model=None,
        sensor_builder=sensor_builder,
        use_instance_graph=False,
        use_sensor_perception=True,
        dedup_skips=lambda _l, _x: False,
        obs=obs,
        frame_step=1,
        graph_object_fusion=fusion,
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 1
    blob = " ".join(objs[0].labels).lower()
    assert "range hood" in blob
    assert "bathroom" not in blob
