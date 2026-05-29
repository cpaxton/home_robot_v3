# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for observation → GraphEQAMemory hooks (instance detections + VLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from emet.memory.graph_eqa.dynamem_graph_hooks import update_graph_memory_from_dynamem_observation
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory


def _fake_frame_with_instance():
    h, w = 8, 8
    inst = torch.full((h, w), -1, dtype=torch.long)
    inst[0:4, 0:4] = 0  # 16 pixels (default min_points=10)
    depth = torch.ones((h, w), dtype=torch.float32) * 1.0
    fw = torch.zeros((h, w, 3), dtype=torch.float32)
    fw[0:4, 0:4, 0] = 2.5
    fw[0:4, 0:4, 1] = -1.0
    fw[0:4, 0:4, 2] = 0.9
    classes = torch.tensor([3], dtype=torch.long)

    class Frame:
        pass

    f = Frame()
    f.instance = inst
    f.depth = depth
    f.full_world_xyz = fw
    f.instance_classes = classes
    return f


def test_instance_detections_added_when_sensor_perception_also_enabled():
    """Regression: both flags true must still feed YoloE/instance nodes into the graph."""
    gm = GraphEQAMemory(parameters={}, defer_llm_clients=True)
    vm = MagicMock()
    vm.observations = [_fake_frame_with_instance()]
    vm.min_depth = 0.1
    vm.max_depth = 5.0
    vm.image_descriptions = []

    det = MagicMock()
    det.class_list = ["mug", "cup", "bowl", "plate"]

    obs = MagicMock()
    obs.rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    obs.camera_pose = np.eye(4)
    obs.camera_pose[:3, 3] = [1.0, 2.0, 3.0]

    sensor_builder = MagicMock()
    sensor_builder.labels_and_description_from_observation.return_value = (["table"], "a table")
    sensor_builder.world_xyz_for_observation.return_value = np.array([1.0, 2.0, 3.0])

    update_graph_memory_from_dynamem_observation(
        graph_memory=gm,
        robot=MagicMock(get_base_pose=MagicMock(return_value=np.zeros(3))),
        voxel_map=vm,
        detection_model=det,
        sensor_builder=sensor_builder,
        use_instance_graph=True,
        use_sensor_perception=True,
        dedup_skips=None,
        obs=obs,
        frame_step=1,
    )

    nodes = gm.get_nodes()
    labels_flat = [lb for n in nodes for lb in n.labels]
    assert "mug" in labels_flat or "plate" in labels_flat, labels_flat
    assert "table" in labels_flat, labels_flat
