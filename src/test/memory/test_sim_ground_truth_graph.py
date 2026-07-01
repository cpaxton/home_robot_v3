# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for sim ground-truth graph upsert and deduplication."""

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, is_ground_truth_node
from emet.memory.graph_eqa.sim_ground_truth_graph import (
    build_ground_truth_graph_from_session,
    deduplicate_placements,
    ground_truth_alignment_report,
    upsert_graph_memory_from_placements,
)
from emet.simulation.sim_object_placements import DEFAULT_TABLE_SCENE_PLACEMENTS, placements_to_session_dict


def _rgb() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _default_table_placements() -> dict[str, dict]:
    session = placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS)
    assert session is not None
    return {k: {"cat": v["cat"], "pos": np.asarray(v["pos"], dtype=np.float64)} for k, v in session.items()}


def test_upsert_gt_deduplicates_by_body_name():
    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = _default_table_placements()
    n1 = upsert_graph_memory_from_placements(mem, _rgb(), placements)
    assert n1 == len(placements)
    n_nodes_1 = len(mem.get_nodes())

    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    assert len(mem.get_nodes()) == n_nodes_1
    gt_nodes = [n for n in mem.get_nodes() if is_ground_truth_node(n)]
    assert len(gt_nodes) == n_nodes_1
    # Second upsert with same poses deduplicates without duplicating nodes (support_count may stay 1).
    assert all(int(n.support_count) == 1 for n in gt_nodes)


def test_maintain_skips_ground_truth_nodes():
    mem = GraphEQAMemory(
        parameters={"dynagraph_staleness_horizon": 1},
        defer_llm_clients=True,
    )
    placements = _default_table_placements()
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    mem.set_graph_timestep(0)
    mem.maintain(100)
    assert len(mem.get_nodes()) == len(placements)


def test_deduplicate_placements_merges_same_cat_nearby():
    raw = {
        "obj_a": {"cat": "mug", "pos": np.array([0.0, 0.0, 1.0])},
        "obj_b": {"cat": "mug", "pos": np.array([0.01, 0.0, 1.0])},
        "obj_c": {"cat": "bowl", "pos": np.array([0.5, 0.0, 1.0])},
    }
    out = deduplicate_placements(raw, merge_xy_m=0.05)
    assert len(out) == 2
    assert "obj_a" in out
    assert "obj_c" in out


def test_ground_truth_alignment_report_perception_only():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = _default_table_placements()
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    mem.add_observation(_rgb(), np.array([0.0, -0.5, 0.6]), ["mystery object"])
    all_report = ground_truth_alignment_report(mem, placements)
    perc_report = ground_truth_alignment_report(mem, placements, perception_nodes_only=True)
    assert "node 1" in all_report
    assert "mystery" in perc_report.lower() or "NO GT match" in perc_report


def test_build_ground_truth_graph_from_session_returns_stable_count():
    mem = GraphEQAMemory(defer_llm_clients=True)
    session = {"sim_object_placements": placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS)}
    n1, gt = build_ground_truth_graph_from_session(mem, _rgb(), session)
    assert gt is not None
    assert n1 >= 3
    n2, _ = build_ground_truth_graph_from_session(mem, _rgb(), session)
    assert n2 == n1
    assert len(mem.get_nodes()) == n1


def test_gt_graph_stores_extent_half_from_bounds():
    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = {
        "sink": {
            "cat": "sink",
            "pos": np.array([1.0, 0.0, 0.8]),
            "bounds": np.array([[0.8, -0.2, 0.75], [1.2, 0.2, 0.85]]),
        },
    }
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    node = mem.get_nodes()[0]
    assert node.extent_half is not None
    np.testing.assert_allclose(node.extent_half, [0.2, 0.2, 0.05], atol=1e-6)


def test_projected_aabb_2d_in_image():
    from emet.memory.graph_eqa.sim_ground_truth_graph import projected_aabb_2d

    camera_pose = np.eye(4, dtype=np.float64)
    camera_K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    bounds = np.array([[0.1, 0.1, 0.5], [0.5, 0.5, 1.0]])
    bbox = projected_aabb_2d(bounds, camera_pose, camera_K, image_hw=(64, 64))
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 < x1 and y0 < y1


def test_associate_ground_truth_to_frame_instances_synthetic():
    from types import SimpleNamespace

    from emet.memory.graph_eqa.sim_ground_truth_graph import (
        associate_ground_truth_to_frame_instances,
        projected_aabb_2d,
    )

    camera_pose = np.eye(4, dtype=np.float64)
    camera_K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    bounds = np.array([[0.1, 0.1, 0.5], [0.5, 0.5, 1.0]])
    bbox = projected_aabb_2d(bounds, camera_pose, camera_K, image_hw=(64, 64))
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    inst = np.zeros((64, 64), dtype=np.int64)
    inst[y0 + 1 : y1, x0 + 1 : x1] = 1
    frame = SimpleNamespace(
        camera_pose=camera_pose,
        camera_K=camera_K,
        instance=inst,
        depth=np.ones((64, 64), dtype=np.float32) * 0.8,
        rgb=_rgb(),
        instance_classes=np.array([0]),
        instance_scores=np.array([0.9]),
    )
    placements = {
        "obj1": {
            "cat": "cube",
            "pos": np.array([0.3, 0.3, 0.75]),
            "bounds": bounds,
        },
    }
    assocs = associate_ground_truth_to_frame_instances(placements, frame, min_iou=0.01)
    assert len(assocs) == 1
    assert assocs[0]["body_key"] == "obj1"
    assert assocs[0]["instance_id"] == 1
    assert assocs[0]["iou"] > 0.01


def test_gt_metrics_helpers():
    from emet.memory.graph_eqa.sim_ground_truth_graph import (
        gt_graph_completeness,
        gt_localization_errors,
        instance_gt_association_recall,
    )

    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = _default_table_placements()
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    assert gt_graph_completeness(mem, placements) == 1.0
    errors = gt_localization_errors(mem, placements)
    assert len(errors) == len(placements)
    assert all(v["err_xy_m"] < 1e-6 for v in errors.values())
    assert instance_gt_association_recall(mem, placements) == 0.0
    mem.attach_detection_to_ground_truth_node("object1", _rgb(), detection_label="cube")
    assert instance_gt_association_recall(mem, placements) > 0.0


def test_placements_to_json_dict():
    from emet.memory.graph_eqa.sim_ground_truth_graph import placements_to_json_dict

    placements = {
        "obj1": {
            "cat": "mug",
            "pos": np.array([1.0, 2.0, 0.5]),
            "bounds": np.array([[0.9, 1.9, 0.4], [1.1, 2.1, 0.6]]),
        },
    }
    out = placements_to_json_dict(placements)
    assert out["obj1"]["cat"] == "mug"
    assert len(out["obj1"]["bounds"]) == 2


def test_ground_truth_observation_hook_records_viewpoints():
    from types import SimpleNamespace

    from emet.memory.graph_eqa.dynamem_graph_hooks import update_graph_memory_ground_truth_from_observation

    mem = GraphEQAMemory(defer_llm_clients=True)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 1.5]
    obs = SimpleNamespace(rgb=_rgb(), camera_pose=pose)
    robot = SimpleNamespace(get_base_pose=lambda: np.array([1.0, 2.0, 0.0]))
    update_graph_memory_ground_truth_from_observation(
        graph_memory=mem,
        robot=robot,
        obs=obs,
        frame_step=3,
    )
    assert len(mem.get_nodes()) == 0
    assert len(mem.get_observations()) == 0
    nav = mem.get_navigation_samples()
    assert len(nav) == 1
    np.testing.assert_allclose(nav[0].xyz, [1.0, 2.0, 1.5])
    np.testing.assert_allclose(nav[0].base_xyz, [1.0, 2.0, 0.0])


def test_save_memory_gt_associations_roundtrip(tmp_path):
    from emet.memory.format import FrameBlob, MemoryManifest, MemoryState, load_memory, save_memory

    state = MemoryState(
        manifest=MemoryManifest(backend="graph_eqa", ground_truth_mode=True),
        frames=[
            FrameBlob(
                camera_pose=np.eye(4),
                gt_associations=[{"body_key": "obj1", "iou": 0.5, "instance_id": 1}],
            )
        ],
    )
    save_memory(state, str(tmp_path))
    loaded = load_memory(str(tmp_path))
    assert loaded.frames[0].gt_associations is not None
    assert loaded.frames[0].gt_associations[0]["body_key"] == "obj1"


def test_attach_detection_to_ground_truth_node():
    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = _default_table_placements()
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    rgb2 = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert mem.attach_detection_to_ground_truth_node("object1", rgb2, detection_label="cube")
    node = next(n for n in mem.get_nodes() if n.description and "object1" in n.description)
    assert "|det:cube" in (node.description or "")
    obs = next(o for o in mem._observations if o.obs_id == node.obs_id)
    assert obs.rgb[0, 0, 0] == 128
