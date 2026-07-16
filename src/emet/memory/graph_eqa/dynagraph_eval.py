# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Unified Dynagraph episode evaluation (explore, graph, fusion, GT, EQA)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from emet.memory.floor_metrics import FLOOR_METRICS_JSON, load_floor_metrics
from emet.memory.format import GRAPH_FILENAME, load_memory


def resolve_episode_dir(episode_dir: str | Path) -> Path:
    """Accept export root or nested ``graph/`` memory directory."""
    p = Path(episode_dir)
    if (p / GRAPH_FILENAME).is_file() or (p / "manifest.json").is_file():
        return p
    nested = p / "graph"
    if (nested / GRAPH_FILENAME).is_file() or (nested / "manifest.json").is_file():
        return nested
    return p


def _is_viewpoint_node(node: Any) -> bool:
    labels = getattr(node, "labels", None) or []
    if any(str(l).lower().startswith("view img") for l in labels):
        return True
    desc = getattr(node, "description", None)
    return isinstance(desc, str) and desc.lower().startswith("viewpoint")


def graph_stats_from_memory_state(state: Any) -> dict[str, Any]:
    nodes = state.graph.nodes if state.graph else []
    object_nodes = [n for n in nodes if not _is_viewpoint_node(n)]
    viewpoint_nodes = [n for n in nodes if _is_viewpoint_node(n)]
    edges = state.graph.edges if state.graph else []
    return {
        "node_count": float(len(object_nodes)),
        "viewpoint_count": float(len(viewpoint_nodes)),
        "edge_count": float(len(edges)),
        "total_node_count": float(len(nodes)),
    }


def graph_stats_from_report_text(text: str) -> dict[str, float]:
    m = re.search(r"Nodes\s*\((\d+)\)", text, re.I)
    node_count = float(m.group(1)) if m else 0.0
    return {"node_count": node_count, "edge_count": 0.0, "viewpoint_count": 0.0}


def explore_metrics(episode_dir: Path) -> dict[str, Any]:
    fm_path = episode_dir / FLOOR_METRICS_JSON
    if not fm_path.is_file():
        parent = episode_dir.parent / FLOOR_METRICS_JSON
        if parent.is_file():
            fm_path = parent
        else:
            return {}
    fm = load_floor_metrics(fm_path.parent)
    explored = float(fm.get("explored_area_m2", 0.0))
    spawn = fm.get("spawn_floor_map") or {}
    scene_walkable = float(spawn.get("scene_walkable_area_m2", 0.0) or 0.0)
    frac = explored / scene_walkable if scene_walkable > 0 else None
    return {
        "explored_area_m2": explored,
        "explored_cell_count": float(fm.get("explored_cell_count", 0)),
        "scene_walkable_area_m2": scene_walkable,
        "explored_fraction": frac,
    }


def _frames_from_detections_json(episode_dir: Path) -> list[dict[str, Any]]:
    frames_dir = episode_dir / "frames"
    if not frames_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for p in sorted(frames_dir.glob("detections_*.json")):
        tag = p.stem.replace("detections_", "")
        try:
            step = int(tag)
        except ValueError:
            step = len(rows) + 1
        dets = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(dets, list):
            rows.append({"step": step, "detections": dets})
    return rows


def fusion_metrics(
    episode_dir: Path,
    gt: dict[str, Any] | None = None,
    *,
    match_xy_m: float = 0.55,
    bounds_iou_min: float = 0.08,
) -> dict[str, Any]:
    from emet.memory.graph_eqa.graph_object_fusion.evaluate import (
        score_detections_vs_gt,
        score_fused_graph_vs_gt,
    )

    frames_path = episode_dir / "calibration_frames.jsonl"
    if frames_path.is_file():
        from emet.memory.graph_eqa.graph_object_fusion.calibrate import load_calibration_frames_jsonl

        frames = load_calibration_frames_jsonl(frames_path)
    else:
        frames = _frames_from_detections_json(episode_dir)

    if not frames:
        return {"n_detections": 0.0, "n_frames": 0.0}

    n_dets = sum(len(fr.get("detections", [])) for fr in frames)
    out: dict[str, Any] = {"n_detections": float(n_dets), "n_frames": float(len(frames))}

    if gt is None:
        gt_path = episode_dir / "gt.json"
        if not gt_path.is_file():
            gt_path = episode_dir.parent / "gt.json"
        if gt_path.is_file():
            from emet.simulation.mujoco_gt_objects import load_gt_scene_json

            gt = load_gt_scene_json(gt_path)

    if gt is not None:
        raw = score_detections_vs_gt(gt, frames, match_xy_m=match_xy_m, bounds_iou_min=bounds_iou_min)
        out["raw"] = {
            k: raw[k]
            for k in (
                "spatial_recall",
                "label_recall",
                "bounds3d_recall",
                "mean_xy_err_m",
                "duplication_penalty",
            )
            if k in raw
        }
        if (episode_dir / GRAPH_FILENAME).is_file():
            from emet.memory.backend import get_memory_backend
            from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

            mem = GraphEQAMemory(defer_llm_clients=True)
            backend = get_memory_backend("graph_eqa", graph_memory=mem, voxel_map=None)
            backend.load(str(episode_dir))
            fused = score_fused_graph_vs_gt(
                mem,
                gt,
                match_xy_m=match_xy_m,
                bounds_iou_min=bounds_iou_min,
                n_raw_detections=int(n_dets),
            )
            out["fused"] = {
                k: fused[k]
                for k in (
                    "spatial_recall",
                    "label_recall",
                    "bounds3d_recall",
                    "n_fused_nodes",
                    "duplication_penalty",
                )
                if k in fused
            }
    return out


def gt_metrics(episode_dir: Path) -> dict[str, Any]:
    from emet.memory.format import SIM_GT_PLACEMENTS_FILENAME
    from emet.memory.graph_eqa.sim_ground_truth_graph import (
        gt_graph_completeness,
        gt_localization_errors,
        instance_gt_association_recall,
        projected_association_recall,
        read_sim_object_placements,
    )

    placements_path = episode_dir / SIM_GT_PLACEMENTS_FILENAME
    if not placements_path.is_file():
        return {}

    placements = read_sim_object_placements(
        {"sim_object_placements": json.loads(placements_path.read_text(encoding="utf-8"))}
    )
    state = load_memory(str(episode_dir))
    from emet.memory.backend import get_memory_backend
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    mem = GraphEQAMemory(defer_llm_clients=True)
    backend = get_memory_backend("graph_eqa", graph_memory=mem, voxel_map=None)
    backend.load(str(episode_dir))

    loc_errors = gt_localization_errors(mem, placements)
    mean_xy = float(np.mean([v["err_xy_m"] for v in loc_errors.values()])) if loc_errors else None

    return {
        "n_placements": float(len(placements)),
        "gt_graph_completeness": gt_graph_completeness(mem, placements),
        "instance_gt_association_recall": instance_gt_association_recall(mem, placements),
        "projected_association_recall": projected_association_recall(state.frames, placements),
        "localization_mean_err_xy_m": mean_xy,
    }


def load_eqa_results(episode_dir: Path) -> list[dict[str, Any]]:
    p = episode_dir / "eqa_results.json"
    if not p.is_file():
        p = episode_dir.parent / "eqa_results.json"
    if not p.is_file():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "questions" in raw:
        return list(raw["questions"])
    return []


def compute_dynagraph_eval(
    episode_dir: str | Path,
    *,
    match_xy_m: float = 0.55,
    bounds_iou_min: float = 0.08,
    questions_path: str | Path | None = None,
    question_env: str | None = None,
) -> dict[str, Any]:
    episode_dir = resolve_episode_dir(episode_dir)
    state = load_memory(str(episode_dir))

    graph = graph_stats_from_memory_state(state)
    report_path = episode_dir / "scene_graph_report.txt"
    if graph["node_count"] == 0.0 and report_path.is_file():
        graph.update(graph_stats_from_report_text(report_path.read_text(encoding="utf-8", errors="replace")))

    graph_health: dict[str, Any] = {}
    graph_json = episode_dir / GRAPH_FILENAME
    if graph_json.is_file():
        from emet.memory.graph_eqa.graph_stats import (
            classify_graph_failure,
            graph_health_from_checkpoint_nodes,
        )

        raw = json.loads(graph_json.read_text(encoding="utf-8"))
        n_obs = len(raw.get("observations") or []) if raw.get("observations") is not None else None
        graph_health = graph_health_from_checkpoint_nodes(
            list(raw.get("nodes") or []),
            n_obs=n_obs,
        )
        graph_health["failure_class"] = classify_graph_failure(graph_health)

    explore = explore_metrics(episode_dir)
    fusion = fusion_metrics(episode_dir, match_xy_m=match_xy_m, bounds_iou_min=bounds_iou_min)
    gt = gt_metrics(episode_dir)

    eqa_section: dict[str, Any] = {}
    eqa_rows = load_eqa_results(episode_dir)
    if eqa_rows:
        from emet.memory.graph_eqa.question_bank import score_eqa_results

        eqa_section = score_eqa_results(eqa_rows, episode_dir=episode_dir)
    elif questions_path:
        from emet.memory.graph_eqa.question_bank import load_question_bank

        bank = load_question_bank(questions_path, env_filter=question_env)
        eqa_section = {"question_bank_loaded": len(bank), "note": "no eqa_results.json in episode"}

    return {
        "episode_dir": str(episode_dir),
        "explore": explore,
        "graph": graph,
        "graph_health": graph_health,
        "fusion": fusion,
        "gt": gt,
        "eqa": eqa_section,
        "n_frames": float(len(state.frames)),
        "ground_truth_mode": bool(getattr(state.manifest, "ground_truth_mode", False)),
    }
