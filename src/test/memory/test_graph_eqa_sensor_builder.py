# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.

import numpy as np

from emet.core.interfaces import Observations
from emet.memory.graph_eqa import (
    GraphEQAMemory,
    SensorGraphBuilder,
    compare_graph_to_placements_report,
    format_scene_graph_pretty,
    labels_from_extract_response,
    parse_comma_separated_labels,
    parse_graph_object_json,
    short_labels_from_voxel_descriptions,
    world_xyz_median_from_depth,
)


def test_parse_comma_separated_labels():
    assert parse_comma_separated_labels("a, b, c") == ["a", "b", "c"]
    assert parse_comma_separated_labels("sink\ntable") == ["sink", "table"]


def test_parse_comma_separated_labels_does_not_split_on_periods():
    """Periods must not become comma separators (old bug: many CoT micro-labels)."""
    out = parse_comma_separated_labels("a.b.c,d")
    assert out == ["a.b.c", "d"]


def test_parse_graph_object_json_plain_and_fenced():
    d = parse_graph_object_json('{"objects":[{"name":"red cylinder"}]}')
    assert d == {"objects": [{"name": "red cylinder"}]}
    raw = 'Here:\n```json\n{"labels":["a","b"]}\n```\n'
    d2 = parse_graph_object_json(raw)
    assert d2 == {"labels": ["a", "b"]}
    assert parse_graph_object_json("not json {") is None


def test_parse_graph_object_json_strips_qwen_thinking_block():
    raw = '<think>planning</think>\n{"labels":["cup"]}\n'
    assert parse_graph_object_json(raw) == {"labels": ["cup"]}


def test_parse_graph_object_json_balanced_when_trailing_extra_brace():
    """First `{`..last `}` would include junk; brace-balanced slice should win."""
    raw = '{"labels":["x"]} and then } noise'
    assert parse_graph_object_json(raw) == {"labels": ["x"]}


def test_parse_graph_object_json_accepts_trailing_comma():
    d = parse_graph_object_json('{"labels":["a","b",],}')
    assert d == {"labels": ["a", "b"]}


def test_labels_from_extract_response_root_string_array():
    assert labels_from_extract_response('["cup", "bowl"]') == ["cup", "bowl"]


def test_labels_from_extract_response_root_object_array():
    assert labels_from_extract_response('[{"name":"x"},{"label":"y"}]') == ["x", "y"]


def test_labels_from_extract_response_items_key():
    assert labels_from_extract_response('{"items":[{"name":"sofa"}]}') == ["sofa"]


def test_labels_from_extract_response_objects_and_labels_keys():
    assert labels_from_extract_response('{"objects":[{"name":"cup"}]}') == ["cup"]
    assert labels_from_extract_response('{"labels":["x","y"]}') == ["x", "y"]
    assert labels_from_extract_response("no json") is None


def test_labels_from_extract_response_rejects_cot_like_names():
    assert labels_from_extract_response('{"objects":[{"name":"The user wants a list"},{"name":"real mug"}]}') == [
        "real mug"
    ]


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


def test_extract_vlm_opt_out_skips_label_client(monkeypatch):
    """``EMET_GRAPH_EQA_EXTRACT_VLM=0`` uses voxel labels only (no vision generate)."""
    monkeypatch.setenv("EMET_GRAPH_EQA_EXTRACT_VLM", "0")
    monkeypatch.setenv("EMET_EQA_AGENTIC_VERIFY", "1")
    called = {"n": 0}

    def _client(_cmds):
        called["n"] += 1
        return '{"objects":[{"name":"chair"}]}'

    b = SensorGraphBuilder(cpu_only=False, perception_client=_client)
    pose = np.eye(4)
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=pose,
    )
    labs, desc = b.labels_and_description_from_observation(obs, voxel_labels=["sofa"])
    assert called["n"] == 0
    assert labs == ["sofa"]
    assert desc is None


def test_agentic_verify_still_runs_vlm_label_extract(monkeypatch):
    """Agentic verify must not disable graph label VLM (otherwise n_object stays 0)."""
    from emet.memory.graph_eqa.sensor_graph_builder import _skip_vlm_label_extract

    monkeypatch.setenv("EMET_EQA_AGENTIC_VERIFY", "1")
    monkeypatch.delenv("EMET_GRAPH_EQA_EXTRACT_VLM", raising=False)
    assert _skip_vlm_label_extract(None) is False
    called = {"n": 0}

    def _client(_cmds):
        called["n"] += 1
        return '{"objects":[{"name":"wall clock"}]}'

    b = SensorGraphBuilder(cpu_only=False, perception_client=_client)
    pose = np.eye(4)
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=pose,
    )
    labs, _desc = b.labels_and_description_from_observation(obs)
    assert called["n"] == 1
    assert labs == ["wall clock"]


def test_short_labels_from_voxel_descriptions_splits_noise():
    long_line = "a" * 100 + ", cup"
    labs = short_labels_from_voxel_descriptions([long_line])
    assert any("cup" in x for x in labs)


def test_labels_and_description_json_with_long_raw_description():
    padding = "z" * 220
    payload = '{"objects":[{"name":"chair"},{"name":"table"}]}'
    b = SensorGraphBuilder(
        cpu_only=False,
        perception_client=lambda x: f"```json\n{payload}\n```\n{padding}",
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
    labs, desc = b.labels_and_description_from_observation(obs, voxel_labels=None)
    assert labs == ["chair", "table"]
    assert desc is not None and len(desc) > 200


def test_labels_and_description_cot_without_json_falls_back_to_object():
    cot = (
        "The user wants a list of visible distinct objects, 1, **Analyze the image:** "
        "dark scene. 2, **Identify:** table."
    )
    b = SensorGraphBuilder(cpu_only=False, perception_client=lambda cmd: cot)
    pose = np.eye(4)
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=pose,
    )
    labs, desc = b.labels_and_description_from_observation(obs, voxel_labels=None)
    assert labs == ["object"]
    assert desc is not None
    assert "The user wants" in desc
    assert not any("**Analyze" in lab for lab in labs)


def test_sensor_graph_builder_mock_vl():
    b = SensorGraphBuilder(
        cpu_only=False,
        perception_client=lambda x: '{"objects":[{"name":"chair"},{"name":"floor"},{"name":"window"}]}',
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


def test_format_scene_graph_pretty_navigation_samples_section():
    mem = GraphEQAMemory(eqa_client=lambda x: "", image_description_client=lambda x: "")
    mem.record_navigation_sample(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.array([1.0, 2.0, 0.05]),
        base_xyz=np.array([1.01, 2.02, 0.0]),
    )
    s = format_scene_graph_pretty(mem, title="Scene graph (export)")
    assert "Navigation samples" in s
    assert "anchor=" in s


def test_format_scene_graph_pretty_lists_all_labels():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    mem.add_observation(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.1]),
        ["a", "b", "c", "d", "e"],
    )
    s = format_scene_graph_pretty(mem)
    assert "a, b, c, d, e" in s


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
