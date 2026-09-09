# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""VLM-selected surface evidence from RGB-D, without a category detector.

A depth-connected surface is not a full semantic segmentation or grasp plan.
It localizes the selected visible surface; manipulation must reacquire/validate
the target immediately before execution.
"""

import math
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import label

from emet.eval.agentic_vlm_assess import _call_eqa_client, _parse_json_object
from emet.memory.graph_eqa.ingest.instance_observations import frame_instances_to_detections, frame_rgb_hwc_uint8


def region_depth_mask(depth, box, point, *, min_depth, max_depth):
    """Finite, depth-connected support around a VLM-selected interior pixel.

    Coordinates are normalized integers in [0,1000], in x/y order. Invalid
    points, multiple depth layers and tiny support fail closed; background is
    never replaced with a frame median or a camera pose.
    """
    values = list(box) + list(point)
    if len(box) != 4 or len(point) != 2 or any(type(v) is not int or not 0 <= v <= 1000 for v in values):
        raise ValueError("invalid normalized region coordinates")
    h, w = depth.shape
    x0, y0 = math.floor(box[0] * w / 1000), math.floor(box[1] * h / 1000)
    x1, y1 = math.ceil(box[2] * w / 1000), math.ceil(box[3] * h / 1000)
    px, py = min(w - 1, point[0] * w // 1000), min(h - 1, point[1] * h // 1000)
    if not (0 <= x0 <= px < x1 <= w and 0 <= y0 <= py < y1 <= h):
        raise ValueError("selected point is outside target region")
    seed = depth[py, px]
    if not np.isfinite(seed) or not min_depth < seed < max_depth:
        raise ValueError("selected surface has invalid depth")
    roi = depth[y0:y1, x0:x1]
    valid = np.isfinite(roi) & (roi > min_depth) & (roi < max_depth)
    # Metric depth support, not a semantic confidence threshold.
    valid &= np.abs(roi - seed) <= 0.10
    components, _ = label(valid)
    component = components[py - y0, px - x0]
    support = components == component
    if not component or support.sum() < 25:
        raise ValueError("insufficient connected depth support")
    mask = np.full(depth.shape, -1, dtype=np.int32)
    mask[y0:y1, x0:x1][support] = 0
    return mask


def select_vlm_region(rgb, query, description, *, client):
    prompt = (
        f"Locate the visible object referred to by {description or query!r}. Target category hint: {query!r}. "
        "Use pixels, not the hint, as evidence. Return a tight bounding box around the target and an interior "
        "point on its visible physical surface, not a hole, occluder or support furniture. Coordinates are "
        "integers normalized to 0..1000, x then y. Verify requested attributes and relationships. "
        "If absent, ambiguous, or the relationship cannot be established, abstain. "
        'Return JSON only: {"verified":true,"box":[x_min,y_min,x_max,y_max],"point":[x,y],"reason":"..."}. '
        'For abstention return {"verified":false,"reason":"..."}.'
    )
    system = "Ground a robot target in the provided image. Do not invent missing visual evidence."
    raw = _call_eqa_client(client, [prompt, Image.fromarray(rgb)], system_prompt=system) if client else ""
    parsed = _parse_json_object(raw)
    verification = {
        "source": "vlm_region",
        "prompt": prompt,
        "system_prompt": system,
        "raw": raw,
        "image_order": ["rgb_file"],
        "valid": False,
    }
    return parsed, verification


def ground_vlm_region(frame, query, description, *, client, min_depth, max_depth):
    rgb = frame_rgb_hwc_uint8(frame)
    depth = frame.depth.detach().cpu().numpy() if hasattr(frame.depth, "detach") else np.asarray(frame.depth)
    detected = SimpleNamespace(
        rgb=rgb,
        depth=depth,
        full_world_xyz=frame.full_world_xyz,
        instance=np.full(depth.shape, -1, dtype=np.int32),
        instance_classes=[0],
        instance_scores=[1.0],
    )
    parsed, verification = select_vlm_region(rgb, query, description, client=client)
    if parsed.get("verified") is not True:
        verification["reason"] = "VLM abstained or returned invalid output"
        return detected, [], [], verification
    try:
        detected.instance = region_depth_mask(
            depth, parsed.get("box", []), parsed.get("point", []), min_depth=min_depth, max_depth=max_depth
        )
    except (ValueError, TypeError) as exc:
        verification["reason"] = str(exc)
        return detected, [], [], verification
    detections = frame_instances_to_detections(
        detected,
        min_depth=min_depth,
        max_depth=max_depth,
        detection_model=SimpleNamespace(class_list=[query]),
    )
    verification.update(
        valid=bool(detections),
        geometry_source="vlm_selected_depth_surface",
        region=parsed,
        score_semantics="binary VLM acceptance, not calibrated detector confidence",
    )
    return detected, detections, [0] if detections else [], verification
