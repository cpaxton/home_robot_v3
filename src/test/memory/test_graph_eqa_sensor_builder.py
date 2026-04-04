# Copyright (c) Hello Robot, Inc. All rights reserved.

import numpy as np

from emet.core.interfaces import Observations
from emet.memory.graph_eqa import (
    GraphEQAMemory,
    SensorGraphBuilder,
    compare_graph_to_placements_report,
    format_scene_graph_pretty,
    parse_comma_separated_labels,
    world_xyz_median_from_depth,
)


def test_parse_comma_separated_labels():
    assert parse_comma_separated_labels("a, b, c") == ["a", "b", "c"]
    assert parse_comma_separated_labels("sink\ntable") == ["sink", "table"]


def test_world_xyz_median_from_depth():
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    depth = np.full((4, 4), 1.0, dtype=np.float32)
    K = np.array([[100.0, 0, 2.0], [0, 100.0, 2.0], [0, 0, 1.0]])
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        camera_K=K,
        camera_pose=pose,
    )
    xyz = world_xyz_median_from_depth(obs)
    assert xyz.shape == (3,)
    assert np.all(np.isfinite(xyz))


def test_sensor_graph_builder_fallback_labels():
    b = SensorGraphBuilder(cpu_only=True)
    pose = np.eye(4)
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=pose,
    )
    assert b.labels_from_observation(obs, voxel_labels=["apple"]) == ["apple"]


def test_sensor_graph_builder_mock_vl():
    b = SensorGraphBuilder(
        cpu_only=False,
        perception_client=lambda x: "chair, floor, window",
    )
    pose = np.eye(4)
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=pose,
    )
    labs = b.labels_from_observation(obs, voxel_labels=None)
    assert "chair" in labs


def test_format_scene_graph_pretty():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    mem.add_observation(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.1]),
        ["obj_a"],
    )
    s = format_scene_graph_pretty(mem)
    assert "Scene graph" in s
    assert "obj_a" in s


def test_compare_graph_to_placements_report():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    mem.add_observation(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.array([0.1, -0.5, 0.4]),
        ["apple"],
    )
    nodes = mem.get_nodes()
    gt = {
        "apple_main": {"cat": "apple", "pos": np.array([0.12, -0.48, 0.41]), "quat": np.zeros(4)},
    }
    r = compare_graph_to_placements_report(nodes, gt, max_dist_xy=2.0)
    assert "apple" in r or "GT" in r


def test_graph_memory_defer_add_observation_without_gemini():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.5]),
        ["table"],
    )
    assert len(mem.get_nodes()) == 1
