# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Label-agnostic calibration eval: instance detections vs sim GT (geometry first)."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode
from emet.memory.graph_eqa.graph_object_fusion.fusion import bounds_3d_iou
from emet.memory.graph_eqa.mujoco_align import _label_matches


@dataclass
class AssociationRow:
    """One GT object ↔ best detection association (geometry-first scoring)."""

    body_key: str
    gt_label: str
    det_label: str | None
    dist_xy_m: float | None
    dist_3d_m: float | None
    label_match: bool
    bounds_3d_iou: float | None
    step: int | None
    matched: bool


def _gt_objects_list(gt: dict[str, Any]) -> list[dict[str, Any]]:
    objs = gt.get("objects", [])
    return [o for o in objs if isinstance(o, dict)]


def _gt_pos(gt_obj: dict[str, Any]) -> np.ndarray:
    return np.asarray(gt_obj.get("pos_world", gt_obj.get("pos", [0, 0, 0])), dtype=np.float64).reshape(3)


def _gt_bounds(gt_obj: dict[str, Any]) -> dict[str, list[float]] | None:
    b = gt_obj.get("bounds_3d")
    if b is None or not isinstance(b, dict):
        return None
    if "min" in b and "max" in b:
        return b
    return None


def _det_xyz(det: dict[str, Any]) -> np.ndarray:
    return np.asarray(det.get("xyz", [0, 0, 0]), dtype=np.float64).reshape(3)


def _iter_frame_detections(frames: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for fr in frames:
        step = int(fr.get("step", 0))
        for d in fr.get("detections", []):
            if isinstance(d, dict):
                out.append((step, d))
    return out


def associate_detections_to_gt(
    gt: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    match_xy_m: float = 0.55,
    bounds_iou_min: float = 0.08,
) -> list[AssociationRow]:
    """
    For each GT object, pick the closest detection centroid in XY within ``match_xy_m``.

    Greedy per-GT (one best detection per body); does not require label agreement.
    """
    gt_objs = _gt_objects_list(gt)
    dets = _iter_frame_detections(frames)
    rows: list[AssociationRow] = []

    for go in gt_objs:
        body_key = str(go.get("id", go.get("label", "?")))
        gt_label = str(go.get("label", body_key))
        gpos = _gt_pos(go)
        gb = _gt_bounds(go)

        best: tuple[float, float, float, str, int, float | None] | None = None
        for step, det in dets:
            xyz = _det_xyz(det)
            dxy = float(np.linalg.norm(xyz[:2] - gpos[:2]))
            if dxy > match_xy_m:
                continue
            d3 = float(np.linalg.norm(xyz - gpos))
            det_label = str(det.get("label", "object"))
            biou: float | None = None
            db = det.get("bounds_3d")
            if gb is not None and isinstance(db, dict):
                biou = bounds_3d_iou(gb, db)
            cand = (dxy, d3, -float(biou or 0.0), det_label, step, biou)
            if best is None or cand < best:
                best = cand

        if best is None:
            rows.append(
                AssociationRow(
                    body_key=body_key,
                    gt_label=gt_label,
                    det_label=None,
                    dist_xy_m=None,
                    dist_3d_m=None,
                    label_match=False,
                    bounds_3d_iou=None,
                    step=None,
                    matched=False,
                )
            )
        else:
            dxy, d3, _, det_label, step, biou = best
            rows.append(
                AssociationRow(
                    body_key=body_key,
                    gt_label=gt_label,
                    det_label=det_label,
                    dist_xy_m=float(dxy),
                    dist_3d_m=float(d3),
                    label_match=_label_matches(det_label, gt_label),
                    bounds_3d_iou=biou,
                    step=int(step),
                    matched=True,
                )
            )
    return rows


def _nodes_to_detection_dicts(nodes: list[GraphNode]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in nodes:
        if getattr(n, "is_viewpoint", False):
            continue
        out.append(
            {
                "label": n.labels[0] if n.labels else "object",
                "xyz": [float(x) for x in np.asarray(n.xyz, dtype=np.float64).reshape(3)],
                "bounds_3d": n.bounds_3d,
            }
        )
    return out


def _metrics_from_associations(
    associations: list[AssociationRow],
    *,
    bounds_iou_min: float,
    n_detections: int,
    n_fused_nodes: int | None = None,
) -> dict[str, Any]:
    n_gt = max(1, len(associations))
    spatial_matched = [a for a in associations if a.matched]
    label_matched = [a for a in spatial_matched if a.label_match]
    bounds_matched = [
        a
        for a in spatial_matched
        if a.bounds_3d_iou is not None and float(a.bounds_3d_iou) >= bounds_iou_min
    ]

    xy_errs = [float(a.dist_xy_m) for a in spatial_matched if a.dist_xy_m is not None]
    mean_xy = float(np.mean(xy_errs)) if xy_errs else None
    p50_xy = float(np.median(xy_errs)) if xy_errs else None

    confusion: Counter[tuple[str, str]] = Counter()
    for a in spatial_matched:
        if not a.label_match and a.det_label:
            confusion[(a.gt_label, a.det_label)] += 1

    dup = 0.0
    if n_fused_nodes is not None:
        dup = max(0.0, float(n_fused_nodes) - float(len(spatial_matched)))

    return {
        "n_gt": float(len(associations)),
        "n_detections": float(n_detections),
        "n_fused_nodes": float(n_fused_nodes) if n_fused_nodes is not None else None,
        "spatial_recall": float(len(spatial_matched)) / float(n_gt),
        "label_recall": float(len(label_matched)) / float(n_gt),
        "bounds3d_recall": float(len(bounds_matched)) / float(n_gt),
        "mean_xy_err_m": mean_xy,
        "p50_xy_err_m": p50_xy,
        "duplication_penalty": dup,
        "taxonomy_confusion": [
            {"gt_label": g, "det_label": d, "count": int(c)} for (g, d), c in confusion.most_common()
        ],
        "associations": [asdict(a) for a in associations],
    }


def score_detections_vs_gt(
    gt: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    match_xy_m: float = 0.55,
    bounds_iou_min: float = 0.08,
) -> dict[str, Any]:
    """Score raw calibration frames (no fusion replay).

    Args:
        gt: Sim GT scene JSON with ``objects[]``.
        frames: Calibration JSONL rows or equivalent ``{step, detections}`` list.
        match_xy_m: Planar association radius (m).
        bounds_iou_min: 3D bounds IoU threshold for ``bounds3d_recall``.

    Returns:
        Metrics dict (``spatial_recall``, ``label_recall``, ``associations``, …).
    """
    associations = associate_detections_to_gt(
        gt,
        frames,
        match_xy_m=match_xy_m,
        bounds_iou_min=bounds_iou_min,
    )
    n_dets = sum(len(fr.get("detections", [])) for fr in frames)
    return _metrics_from_associations(
        associations,
        bounds_iou_min=bounds_iou_min,
        n_detections=n_dets,
    )


def score_fused_graph_vs_gt(
    mem: GraphEQAMemory,
    gt: dict[str, Any],
    *,
    match_xy_m: float = 0.55,
    bounds_iou_min: float = 0.08,
    n_raw_detections: int | None = None,
) -> dict[str, Any]:
    """Score fused graph nodes using the same spatial association as raw detections.

    Args:
        mem: ``GraphEQAMemory`` after fusion replay or live explore.
        gt: Sim GT scene JSON.
        match_xy_m: Planar association radius (m).
        bounds_iou_min: 3D bounds IoU threshold.
        n_raw_detections: Optional raw detection count (for duplication diagnostics).

    Returns:
        Metrics dict including ``n_fused_nodes`` and ``duplication_penalty``.
    """
    nodes = [n for n in mem.get_nodes() if not getattr(n, "is_viewpoint", False)]
    pseudo_frames = [{"step": 0, "detections": _nodes_to_detection_dicts(nodes)}]
    associations = associate_detections_to_gt(
        gt,
        pseudo_frames,
        match_xy_m=match_xy_m,
        bounds_iou_min=bounds_iou_min,
    )
    n_dets = n_raw_detections if n_raw_detections is not None else len(nodes)
    return _metrics_from_associations(
        associations,
        bounds_iou_min=bounds_iou_min,
        n_detections=n_dets,
        n_fused_nodes=len(nodes),
    )


def format_calibration_eval_report(metrics: dict[str, Any]) -> str:
    """Human-readable summary for stdout."""
    lines = [
        "### Calibration eval (geometry-first)",
        "",
        f"- spatial_recall: {metrics.get('spatial_recall', 0):.3f}",
        f"- label_recall:   {metrics.get('label_recall', 0):.3f}  _(taxonomy diagnostic)_",
        f"- bounds3d_recall: {metrics.get('bounds3d_recall', 0):.3f}",
    ]
    if metrics.get("mean_xy_err_m") is not None:
        lines.append(f"- mean_xy_err_m: {metrics['mean_xy_err_m']:.3f}")
    if metrics.get("p50_xy_err_m") is not None:
        lines.append(f"- p50_xy_err_m:  {metrics['p50_xy_err_m']:.3f}")
    if metrics.get("n_fused_nodes") is not None:
        lines.append(
            f"- fused nodes: {int(metrics['n_fused_nodes'])} "
            f"(dup penalty {metrics.get('duplication_penalty', 0):.0f})"
        )
    lines.append("")
    lines.append("#### Per-GT associations")
    for a in metrics.get("associations", []):
        if not a.get("matched"):
            lines.append(f"- **{a['body_key']}** ({a['gt_label']}): _no detection within radius_")
            continue
        lm = "yes" if a.get("label_match") else "no"
        biou = a.get("bounds_3d_iou")
        biou_s = f"{biou:.3f}" if biou is not None else "n/a"
        lines.append(
            f"- **{a['body_key']}** ({a['gt_label']}) → det `{a['det_label']}` "
            f"xy={a['dist_xy_m']:.2f}m label_match={lm} bounds_iou={biou_s} step={a.get('step')}"
        )
    conf = metrics.get("taxonomy_confusion") or []
    if conf:
        lines.append("")
        lines.append("#### Taxonomy confusion (spatial match, label mismatch)")
        for row in conf[:12]:
            lines.append(f"- {row['gt_label']} → {row['det_label']} (×{row['count']})")
    return "\n".join(lines)
