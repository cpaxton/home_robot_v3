# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Map voxel Frame instance masks + YoloE class ids to graph labels and world XYZ centroids."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M = 0.4

logger = logging.getLogger(__name__)


def frame_rgb_hwc_uint8(frame: Any) -> np.ndarray | None:
    """Head RGB from a voxel ``Frame`` as H×W×3 uint8 (matches instance mask resolution)."""
    rgb = getattr(frame, "rgb", None)
    if rgb is None:
        return None
    t = rgb if isinstance(rgb, torch.Tensor) else torch.as_tensor(rgb)
    if t.ndim == 3 and int(t.shape[0]) == 3:
        t = t.permute(1, 2, 0)
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[:, :, :3]
    if float(arr.max()) <= 1.0 + 1e-6:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def frame_world_xyz_hw3(frame: Any) -> torch.Tensor | None:
    """
    Dense H×W×3 world XYZ aligned with ``frame.depth`` / ``frame.instance``.

    ``SparseVoxelMapDynamem.add`` stores ``full_world_xyz`` as (H*W, 3) from depth unprojection;
    without reshaping, graph instance extraction returns no detections.
    """
    depth = getattr(frame, "depth", None)
    fw = getattr(frame, "full_world_xyz", None)
    if depth is None or fw is None:
        return None
    if not isinstance(depth, torch.Tensor):
        depth = torch.as_tensor(depth)
    if not isinstance(fw, torch.Tensor):
        fw = torch.as_tensor(fw)
    h, w = int(depth.shape[0]), int(depth.shape[1])
    if fw.dim() == 3 and int(fw.shape[0]) == h and int(fw.shape[1]) == w:
        return fw.to(dtype=torch.float32)
    if fw.dim() == 2 and int(fw.shape[-1]) == 3:
        if int(fw.shape[0]) == h * w:
            return fw.reshape(h, w, 3).to(dtype=torch.float32)
    if int(fw.numel()) == h * w * 3:
        return fw.reshape(h, w, 3).to(dtype=torch.float32)
    return None


def label_for_detection_category(detection_model: Any | None, category_id: int) -> str:
    """Resolve a detector category index to a short string (YoloE: ``class_list`` / vocabulary)."""
    cid = int(category_id)
    if detection_model is None:
        return f"object_{cid}"
    cl = getattr(detection_model, "class_list", None)
    if cl is not None and 0 <= cid < len(cl):
        return str(cl[cid])
    vocab = getattr(detection_model, "_current_vocabulary", None)
    if isinstance(vocab, dict) and cid in vocab:
        return str(vocab[cid])
    return f"class_{cid}"


def frame_instances_to_detections(
    frame: Any,
    *,
    min_depth: float,
    max_depth: float,
    detection_model: Any | None = None,
    min_points: int = 10,
    background_instance_labels: tuple[int, ...] = (-1,),
) -> list[dict[str, Any]]:
    """
    Per-instance structured rows for graph nodes + JSON cache.

    Each dict: ``instance_id``, ``category_id``, ``label_short``, ``xyz`` (3 floats).

    Expects ``overlay_masks`` convention: pixel values are mask indices ``0 .. K-1``; background is ``-1``.
    """
    inst = getattr(frame, "instance", None)
    if inst is None:
        return []
    depth = getattr(frame, "depth", None)
    classes = getattr(frame, "instance_classes", None)
    if depth is None:
        return []
    fw = frame_world_xyz_hw3(frame)
    if fw is None:
        raw = getattr(frame, "full_world_xyz", None)
        logger.warning(
            "graph instances: cannot reshape full_world_xyz (depth %s, fw %s)",
            getattr(depth, "shape", None),
            getattr(raw, "shape", None),
        )
        return []

    if not isinstance(inst, torch.Tensor):
        inst = torch.as_tensor(inst)
    if not isinstance(depth, torch.Tensor):
        depth = torch.as_tensor(depth)
    if classes is not None and not isinstance(classes, torch.Tensor):
        classes = torch.as_tensor(classes)

    inst = inst.to(device=fw.device, dtype=torch.long)
    depth = depth.to(device=fw.device, dtype=torch.float32)
    h, w = int(depth.shape[0]), int(depth.shape[1])
    if inst.shape[0] != h or inst.shape[1] != w:
        logger.warning(
            "graph instances: instance mask %s != depth %s",
            tuple(inst.shape),
            (h, w),
        )
        return []

    bg = set(background_instance_labels)
    valid_depth = (depth > min_depth) & (depth < max_depth) & torch.isfinite(depth)
    finite_xyz = torch.isfinite(fw).all(dim=-1)
    valid = valid_depth & finite_xyz

    uniq = torch.unique(inst)
    out: list[dict[str, Any]] = []
    for uid_t in uniq:
        uid = int(uid_t.item())
        if uid in bg:
            continue
        mask = (inst == uid) & valid
        if int(mask.sum().item()) < min_points:
            continue
        pts = fw[mask]
        xyz = pts.median(dim=0).values.detach().cpu().numpy().astype(np.float64)
        pts_np = pts.detach().cpu().numpy().astype(np.float64)
        mn = pts_np.min(axis=0)
        mx = pts_np.max(axis=0)
        bounds_3d = {
            "min": mn.tolist(),
            "max": mx.tolist(),
            "center": (0.5 * (mn + mx)).tolist(),
            "size": (mx - mn).tolist(),
        }
        ys, xs = torch.where(mask)
        bbox_xyxy = (
            int(xs.min().item()),
            int(ys.min().item()),
            int(xs.max().item()) + 1,
            int(ys.max().item()) + 1,
        )

        category_id = -1
        if classes is not None and uid < classes.shape[0]:
            category_id = int(classes[uid].item())
        label = label_for_detection_category(detection_model, category_id)
        out.append(
            {
                "instance_id": uid,
                "category_id": category_id,
                "label_short": label,
                "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "bbox_xyxy": bbox_xyxy,
                "bounds_3d": bounds_3d,
            }
        )

    return out


def frame_instances_to_labels_xyz(
    frame: Any,
    *,
    min_depth: float,
    max_depth: float,
    detection_model: Any | None = None,
    min_points: int = 10,
    background_instance_labels: tuple[int, ...] = (-1,),
) -> list[tuple[str, np.ndarray]]:
    """For each non-background instance id, median world XYZ and short label (see ``frame_instances_to_detections``)."""
    dets = frame_instances_to_detections(
        frame,
        min_depth=min_depth,
        max_depth=max_depth,
        detection_model=detection_model,
        min_points=min_points,
        background_instance_labels=background_instance_labels,
    )
    return [
        (
            d["label_short"],
            np.asarray(d["xyz"], dtype=np.float64),
            tuple(d["bbox_xyxy"]),
        )
        for d in dets
    ]


def instance_items_from_instance_memory(
    voxel_map: Any,
    detection_model: Any | None,
) -> list[tuple[str, np.ndarray, tuple[int, int, int, int]]]:
    """Fallback: build graph rows from InstanceMemory when Frame XYZ layout blocked extraction."""
    get_instances = getattr(voxel_map, "get_instances", None)
    if get_instances is None:
        return []
    items: list[tuple[str, np.ndarray, tuple[int, int, int, int]]] = []
    for inst in get_instances():
        view = inst.get_best_view() if hasattr(inst, "get_best_view") else None
        if view is None:
            continue
        bbox_t = getattr(view, "bbox", None)
        if bbox_t is None:
            continue
        bb = np.asarray(bbox_t.detach().cpu().numpy() if isinstance(bbox_t, torch.Tensor) else bbox_t).reshape(-1)
        if bb.size < 4:
            continue
        bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        try:
            xyz = np.asarray(inst.get_center().detach().cpu().numpy(), dtype=np.float64).reshape(3)
        except Exception:
            continue
        cid = int(view.category_id) if getattr(view, "category_id", None) is not None else -1
        label = label_for_detection_category(detection_model, cid)
        items.append((label, xyz, bbox))
    return items
