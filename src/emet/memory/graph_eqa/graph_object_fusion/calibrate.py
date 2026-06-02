# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Offline grid search for GraphObjectFusion thresholds vs sim GT."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate, GraphObjectFusion
from emet.memory.graph_eqa.mujoco_align import _norm_label, score_nodes_vs_gt
from emet.simulation.mujoco_gt_objects import load_gt_scene_json


def load_calibration_frames_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def detection_dict_to_candidate(d: dict[str, Any]) -> GraphDetectionCandidate:
    emb = d.get("embedding")
    if emb is not None:
        emb = np.asarray(emb, dtype=np.float32)
    return GraphDetectionCandidate(
        label=str(d.get("label", "object")),
        xyz=np.asarray(d.get("xyz", [0, 0, 0]), dtype=np.float64),
        bbox_xyxy=tuple(d["bbox_xyxy"]) if d.get("bbox_xyxy") is not None else None,
        bounds_3d=d.get("bounds_3d"),
        embedding=emb,
    )


def replay_frames_with_fusion(
    frames: list[dict[str, Any]],
    config: GraphObjectFusionConfig,
) -> GraphEQAMemory:
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    fusion = GraphObjectFusion(config)
    rgb_stub = np.zeros((8, 8, 3), dtype=np.uint8)
    for fr in frames:
        step = int(fr.get("step", 0))
        if step > 0:
            mem.set_graph_timestep(step)
        for d in fr.get("detections", []):
            cand = detection_dict_to_candidate(d)
            fusion.apply_detection(mem, rgb_stub, cand, viewer_xyz=None)
    return mem


def score_fused_nodes_vs_gt(
    mem: GraphEQAMemory,
    gt: dict[str, Any],
    *,
    match_xy_m: float | None = None,
) -> dict[str, float]:
    mxy = match_xy_m if match_xy_m is not None else 0.55
    nodes = [n for n in mem.get_nodes() if not n.is_viewpoint]
    gt_objects = gt.get("objects", [])
    return score_nodes_vs_gt(nodes, gt_objects, match_xy_m=mxy)


def grid_search_fusion_config(
    frames: list[dict[str, Any]],
    gt: dict[str, Any],
    *,
    spatial_values: tuple[float, ...] = (0.30, 0.38, 0.42, 0.48, 0.55),
    embed_values: tuple[float, ...] = (0.55, 0.62, 0.70, 0.78),
    iou_values: tuple[float, ...] = (0.05, 0.08, 0.12, 0.18),
    min_recall: float = 0.85,
) -> tuple[GraphObjectFusionConfig, dict[str, Any], list[dict[str, Any]]]:
    """Return best config, best metrics, and full grid results."""
    base = GraphObjectFusionConfig(enabled=True)
    results: list[dict[str, Any]] = []
    best_cfg: GraphObjectFusionConfig | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, int] | None = None

    for sx, ec, iou in product(spatial_values, embed_values, iou_values):
        cfg = replace(
            base,
            spatial_merge_xy_m=float(sx),
            embedding_min_cosine=float(ec),
            bounds_3d_iou_min=float(iou),
        )
        mem = replay_frames_with_fusion(frames, cfg)
        metrics = score_fused_nodes_vs_gt(mem, gt, match_xy_m=cfg.match_xy_m)
        row = {
            "spatial_merge_xy_m": sx,
            "embedding_min_cosine": ec,
            "bounds_3d_iou_min": iou,
            **metrics,
        }
        results.append(row)
        if metrics["gt_recall"] < min_recall:
            continue
        key = (
            -metrics["duplication_penalty"],
            -metrics["node_count"],
            metrics["node_precision"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = cfg
            best_metrics = metrics

    if best_cfg is None:
        best_cfg = replace(base, spatial_merge_xy_m=0.42, embedding_min_cosine=0.62, bounds_3d_iou_min=0.08)
        mem = replay_frames_with_fusion(frames, best_cfg)
        best_metrics = score_fused_nodes_vs_gt(mem, gt)

    report = {
        "best": {**(best_metrics or {}), **asdict(best_cfg)},
        "grid": results,
    }
    return best_cfg, report, results


def write_fusion_config_yaml(path: str | Path, config: GraphObjectFusionConfig) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = {"graph_object_fusion": asdict(config)}
    import yaml

    dest.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return dest
