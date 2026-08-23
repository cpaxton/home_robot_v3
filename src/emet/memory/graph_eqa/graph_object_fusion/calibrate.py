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
from emet.memory.graph_eqa.graph_object_fusion.evaluate import score_fused_graph_vs_gt
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate, GraphObjectFusion


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
        identity_key=d.get("identity_key"),
        countable_instance=bool(d.get("countable_instance", d.get("instance_id") is not None)),
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
        if config.bounds_3d_iou_merge_min > 0.0:
            fusion.consolidate_high_iou_nodes(mem)
    return mem


def score_fused_nodes_vs_gt(
    mem: GraphEQAMemory,
    gt: dict[str, Any],
    *,
    match_xy_m: float | None = None,
    frames: list[dict[str, Any]] | None = None,
    bounds_iou_min: float = 0.08,
) -> dict[str, float]:
    mxy = match_xy_m if match_xy_m is not None else 0.55
    n_raw = sum(len(fr.get("detections", [])) for fr in frames) if frames else None
    metrics = score_fused_graph_vs_gt(
        mem,
        gt,
        match_xy_m=mxy,
        bounds_iou_min=bounds_iou_min,
        n_raw_detections=n_raw,
    )
    n_nodes = float(metrics.get("n_fused_nodes") or 0)
    return {
        "spatial_recall": float(metrics["spatial_recall"]),
        "label_recall": float(metrics["label_recall"]),
        "bounds3d_recall": float(metrics["bounds3d_recall"]),
        "gt_recall": float(metrics["spatial_recall"]),
        "duplication_penalty": float(metrics.get("duplication_penalty", 0.0)),
        "node_count": n_nodes,
        "node_precision": float(metrics["spatial_recall"]),
    }


def grid_search_fusion_config(
    frames: list[dict[str, Any]],
    gt: dict[str, Any],
    *,
    spatial_values: tuple[float, ...] = (0.30, 0.38, 0.42, 0.48, 0.55),
    embed_values: tuple[float, ...] = (0.55, 0.62, 0.70, 0.78),
    iou_values: tuple[float, ...] = (0.05, 0.08, 0.12, 0.18),
    min_recall: float = 0.85,
    min_label_recall: float | None = None,
) -> tuple[GraphObjectFusionConfig, dict[str, Any], list[dict[str, Any]]]:
    """Return best config, best metrics, and full grid results (objective: spatial_recall)."""
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
        metrics = score_fused_nodes_vs_gt(
            mem,
            gt,
            match_xy_m=cfg.match_xy_m,
            frames=frames,
            bounds_iou_min=float(iou),
        )
        row = {
            "spatial_merge_xy_m": sx,
            "embedding_min_cosine": ec,
            "bounds_3d_iou_min": iou,
            **metrics,
        }
        results.append(row)
        if metrics["spatial_recall"] < min_recall:
            continue
        if min_label_recall is not None and metrics["label_recall"] < min_label_recall:
            continue
        key = (
            metrics["spatial_recall"],
            metrics["bounds3d_recall"],
            -metrics["duplication_penalty"],
            -metrics["node_count"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = cfg
            best_metrics = metrics

    if best_cfg is None:
        best_cfg = replace(base, spatial_merge_xy_m=0.42, embedding_min_cosine=0.62, bounds_3d_iou_min=0.08)
        mem = replay_frames_with_fusion(frames, best_cfg)
        best_metrics = score_fused_nodes_vs_gt(mem, gt, frames=frames)

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
