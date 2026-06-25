# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphObjectFusion
from emet.memory.graph_eqa.dynamem_graph_hooks import update_graph_memory_from_dynamem_observation


def _fake_obs(rgb_shape=(4, 4, 3)):
    obs = MagicMock()
    obs.rgb = np.zeros(rgb_shape, dtype=np.uint8)
    obs.camera_pose = np.eye(4, dtype=np.float64)
    obs.camera_pose[2, 3] = 1.2
    obs.semantic = None
    obs.depth = None
    obs.gps = None
    obs.compass = None
    obs.navigation_origin_xyt = None
    return obs


def test_vlm_labels_merge_through_fusion_when_enabled():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.55,
        embedding_min_cosine=0.0,
        fallback_spatial_merge_xy_m=0.55,
        require_label_match=False,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0

    sensor_builder = MagicMock()
    sensor_builder.labels_and_description_from_observation.return_value = (
        ["mug", "coffee cup", "plate"],
        "scene",
    )
    sensor_builder.world_xyz_for_observation.return_value = np.array([1.0, 0.5, 0.9])

    vm = MagicMock()
    vm.observations = []
    vm.image_descriptions = []

    obs = _fake_obs()
    for _ in range(3):
        update_graph_memory_from_dynamem_observation(
            graph_memory=mem,
            robot=MagicMock(),
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
    assert objs[0].support_count >= 3
