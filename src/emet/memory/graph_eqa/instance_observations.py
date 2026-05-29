# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Map voxel Frame instance masks + YoloE class ids to graph labels and world XYZ centroids."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M = 0.4


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
    fw = getattr(frame, "full_world_xyz", None)
    depth = getattr(frame, "depth", None)
    classes = getattr(frame, "instance_classes", None)
    if fw is None or depth is None:
        return []

    if not isinstance(inst, torch.Tensor):
        inst = torch.as_tensor(inst)
    if not isinstance(fw, torch.Tensor):
        fw = torch.as_tensor(fw)
    if not isinstance(depth, torch.Tensor):
        depth = torch.as_tensor(depth)
    if classes is not None and not isinstance(classes, torch.Tensor):
        classes = torch.as_tensor(classes)

    inst = inst.to(device=fw.device, dtype=torch.long)
    fw = fw.to(dtype=torch.float32)
    depth = depth.to(device=fw.device, dtype=torch.float32)

    if fw.dim() != 3 or fw.shape[-1] != 3:
        return []
    h, w = int(depth.shape[0]), int(depth.shape[1])
    if inst.shape[0] != h or inst.shape[1] != w:
        return []
    if fw.shape[0] != h or fw.shape[1] != w:
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
