# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for LazyGraph commit helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.lazy_graph_commit import (
    commit_graph_from_arrival_obs,
    record_lazy_graph_viewpoint,
)


class _FakeObs:
    def __init__(self) -> None:
        self.camera_pose = np.eye(4, dtype=np.float64)
        self.camera_pose[:3, 3] = [1.0, 2.0, 0.5]
        self.rgb = np.zeros((8, 8, 3), dtype=np.uint8)


def test_record_lazy_graph_viewpoint_adds_navigation_sample():
    gm = GraphEQAMemory(parameters={"graph_eqa_record_navigation": True})
    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    obs = _FakeObs()

    record_lazy_graph_viewpoint(graph_memory=gm, robot=robot, obs=obs, frame_step=3)

    samples = gm.get_navigation_samples()
    assert len(samples) == 1


def test_commit_graph_from_arrival_obs_uses_qwen_labels_only():
    gm = GraphEQAMemory(parameters={"dynagraph_merge_xy_m": 0.0})
    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    obs = _FakeObs()

    sensor_builder = MagicMock()
    sensor_builder.labels_and_description_from_observation.return_value = (["red mug"], "a mug")
    sensor_builder.world_xyz_for_observation.return_value = np.array([1.1, 2.2, 0.6], dtype=np.float64)

    obs_id = commit_graph_from_arrival_obs(
        graph_memory=gm,
        robot=robot,
        sensor_builder=sensor_builder,
        obs=obs,
        query_text="mug",
        localize_source="voxel",
        object_xyz=np.array([1.1, 2.2, 0.6]),
        frame_step=1,
    )

    assert obs_id == 1
    sensor_builder.labels_and_description_from_observation.assert_called_once()
    assert sensor_builder.labels_and_description_from_observation.call_args.kwargs.get("voxel_labels") is None
    object_nodes = [n for n in gm.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(object_nodes) == 1
    assert object_nodes[0].labels == ["red mug"]


def test_normalize_memory_backend_accepts_lazy_graph():
    from emet.config.embodied_agent_config import normalize_memory_backend

    assert normalize_memory_backend("lazy-graph") == "lazy_graph"
