# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.calibrate import (
    grid_search_fusion_config,
    load_calibration_frames_jsonl,
    replay_frames_with_fusion,
    score_fused_nodes_vs_gt,
)
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_object_fusion.fusion import (
    GraphDetectionCandidate,
    GraphObjectFusion,
    bounds_3d_iou,
    cosine_similarity_np,
)
from emet.memory.graph_eqa.graph_object_fusion.evaluate import (
    associate_detections_to_gt,
    score_detections_vs_gt,
)
from emet.memory.graph_eqa.mujoco_align import score_nodes_vs_gt


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


def test_bounds_3d_iou_disjoint():
    assert bounds_3d_iou(
        {"min": [0, 0, 0], "max": [1, 1, 1]},
        {"min": [2, 0, 0], "max": [3, 1, 1]},
    ) == 0.0


def test_cosine_identical():
    v = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    assert cosine_similarity_np(v, v) > 0.99


def test_fusion_merges_duplicate_detections():
    cfg = GraphObjectFusionConfig(enabled=True, spatial_merge_xy_m=0.5, embedding_min_cosine=0.0)
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    b = {"min": [0, 0, 0], "max": [0.2, 0.2, 0.2]}
    c1 = GraphDetectionCandidate(label="cup", xyz=np.array([1.0, 0.0, 0.5]), bounds_3d=b)
    c2 = GraphDetectionCandidate(label="cup", xyz=np.array([1.05, 0.02, 0.52]), bounds_3d=b)
    fusion.apply_detection(mem, rgb, c1)
    fusion.apply_detection(mem, rgb, c2)
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 1
    assert objs[0].support_count >= 2


def test_spatial_recall_without_label_match():
    gt = {
        "objects": [
            {
                "id": "mug_main",
                "label": "mug",
                "pos_world": [1.0, 2.0, 0.9],
            }
        ]
    }
    frames = [
        {
            "step": 1,
            "detections": [
                {"label": "cup", "xyz": [1.02, 2.01, 0.91]},
                {"label": "plate", "xyz": [5.0, 5.0, 0.5]},
            ],
        }
    ]
    metrics = score_detections_vs_gt(gt, frames, match_xy_m=0.55)
    assert metrics["spatial_recall"] == 1.0
    assert metrics["label_recall"] == 0.0
    rows = associate_detections_to_gt(gt, frames, match_xy_m=0.55)
    assert rows[0].matched
    assert rows[0].det_label == "cup"
    assert not rows[0].label_match


def test_replay_and_score_fixture():
    gt = json.loads((_fixtures_dir() / "gt_robocasa_seed0_snippet.json").read_text(encoding="utf-8"))
    frames = load_calibration_frames_jsonl(_fixtures_dir() / "calibration_frames_snippet.jsonl")
    raw = score_detections_vs_gt(gt, frames)
    assert raw["spatial_recall"] >= 0.5
    cfg = GraphObjectFusionConfig(enabled=True, spatial_merge_xy_m=0.55, embedding_min_cosine=0.0)
    mem = replay_frames_with_fusion(frames, cfg)
    metrics = score_fused_nodes_vs_gt(mem, gt, frames=frames)
    assert metrics["spatial_recall"] >= 0.5
    assert metrics["node_count"] <= 3.0


def test_grid_search_fixture():
    gt = json.loads((_fixtures_dir() / "gt_robocasa_seed0_snippet.json").read_text(encoding="utf-8"))
    frames = load_calibration_frames_jsonl(_fixtures_dir() / "calibration_frames_snippet.jsonl")
    best, report, _ = grid_search_fusion_config(
        frames,
        gt,
        spatial_values=(0.4, 0.55),
        embed_values=(0.0,),
        iou_values=(0.05,),
        min_recall=0.4,
    )
    assert best.enabled
    assert report["best"]["spatial_recall"] >= 0.4
    assert "label_recall" in report["best"]


def test_score_nodes_vs_gt_label_optional():
    from emet.memory.graph_eqa.graph_memory import GraphNode

    gt_objs = [{"label": "mug", "pos_world": [1.0, 2.0, 0.9]}]
    node = GraphNode(node_id=1, labels=["cup"], xyz=np.array([1.02, 2.01, 0.91]), obs_id=1)
    gated = score_nodes_vs_gt([node], gt_objs, require_label_match=True)
    spatial = score_nodes_vs_gt([node], gt_objs, require_label_match=False)
    assert gated["gt_recall"] == 0.0
    assert spatial["gt_recall"] == 1.0
