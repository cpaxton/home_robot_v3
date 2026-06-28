# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.calibrate import (
    load_calibration_frames_jsonl,
    replay_frames_with_fusion,
    score_fused_nodes_vs_gt,
)
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig, load_graph_object_fusion_config
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate, GraphObjectFusion


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


def _default_fusion_config() -> GraphObjectFusionConfig:
    return load_graph_object_fusion_config()


def test_stationary_noisy_dedup():
    gt = json.loads((_fixtures_dir() / "gt_robocasa_seed0_snippet.json").read_text(encoding="utf-8"))
    frames = load_calibration_frames_jsonl(_fixtures_dir() / "calibration_frames_stationary_noisy.jsonl")
    cfg = _default_fusion_config()
    mem = replay_frames_with_fusion(frames, cfg)
    metrics = score_fused_nodes_vs_gt(mem, gt, frames=frames, match_xy_m=cfg.match_xy_m)
    n_gt = len(gt["objects"])
    assert metrics["spatial_recall"] >= 0.9
    assert metrics["node_count"] <= float(n_gt) + 1.0
    assert metrics["duplication_penalty"] <= 1.0


def test_long_explore_noisy_dedup():
    gt = json.loads((_fixtures_dir() / "gt_long_explore_noisy.json").read_text(encoding="utf-8"))
    frames = load_calibration_frames_jsonl(_fixtures_dir() / "calibration_frames_long_explore_noisy.jsonl")
    cfg = _default_fusion_config()
    mem = replay_frames_with_fusion(frames, cfg)
    metrics = score_fused_nodes_vs_gt(mem, gt, frames=frames, match_xy_m=cfg.match_xy_m)
    n_gt = len(gt["objects"])
    n_raw = sum(len(fr.get("detections", [])) for fr in frames)
    assert metrics["spatial_recall"] >= 0.85
    assert metrics["node_count"] <= float(n_gt) * 1.3
    assert metrics["node_count"] / max(1.0, float(n_raw)) <= 0.7


def test_snippet_regression_unchanged():
    gt = json.loads((_fixtures_dir() / "gt_robocasa_seed0_snippet.json").read_text(encoding="utf-8"))
    frames = load_calibration_frames_jsonl(_fixtures_dir() / "calibration_frames_snippet.jsonl")
    cfg = _default_fusion_config()
    mem = replay_frames_with_fusion(frames, cfg)
    metrics = score_fused_nodes_vs_gt(mem, gt, frames=frames, match_xy_m=cfg.match_xy_m)
    assert metrics["spatial_recall"] >= 0.5
    assert metrics["node_count"] <= 3.0


def test_fallback_tier_unit():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.42,
        min_centroid_dist_m=0.55,
        bounds_3d_iou_min=0.08,
        embedding_min_cosine=0.99,
        fallback_spatial_merge_xy_m=0.45,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    b = {"min": [0, 0, 0], "max": [0.2, 0.2, 0.2]}
    c1 = GraphDetectionCandidate(
        label="cup",
        xyz=np.array([1.0, 0.0, 0.5]),
        bounds_3d=b,
        embedding=emb_a,
    )
    c2 = GraphDetectionCandidate(
        label="mug",
        xyz=np.array([1.04, 0.03, 0.52]),
        bounds_3d={"min": [2, 2, 2], "max": [2.2, 2.2, 2.2]},
        embedding=emb_b,
    )
    fusion.apply_detection(mem, rgb, c1)
    fusion.apply_detection(mem, rgb, c2)
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint]
    assert len(objs) == 1
    assert objs[0].support_count >= 2
