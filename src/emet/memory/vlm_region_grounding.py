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
from PIL import Image, ImageDraw
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


def region_annotation(rgb, region):
    image = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(image)
    h, w = rgb.shape[:2]
    box, point = region.get("box"), region.get("point")
    if isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box):
        coordinates = np.asarray(box) * [w, h, w, h] / 1000
        if coordinates[2] >= coordinates[0] and coordinates[3] >= coordinates[1]:
            draw.rectangle(tuple(coordinates), outline="yellow", width=3)
    if isinstance(point, list) and len(point) == 2 and all(type(v) is int for v in point):
        x, y = point[0] * w / 1000, point[1] * h / 1000
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="red", width=3)
    return image


def select_vlm_region(rgb, query, description, *, client, correction=None, box_only=False):
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
    if box_only:
        prompt = (
            f"Locate {description or query!r} in the image. Category hint: {query!r}. "
            "Return a bounding box around the visible target; coordinates are integers normalized "
            "to 0..1000 in x,y order. Check requested attributes and relationships from pixels. "
            "Abstain if absent, ambiguous, or the relationship cannot be established. "
            'Return JSON {"verified":true,"box":[x_min,y_min,x_max,y_max],"reason":"..."} '
            'or {"verified":false,"reason":"..."}. Do not predict a 3D position or surface point.'
        )
    images = [Image.fromarray(rgb)]
    if correction is not None:
        prompt += (
            f" Your previous selection was {correction['region']!r}. Geometry check: {correction['error']}. "
            "The second image marks that selection in yellow with a red point. Correct the box and interior "
            "point using the original image, or abstain. This feedback does not establish object presence."
        )
        images.append(region_annotation(rgb, correction["region"]))
    raw = _call_eqa_client(client, [prompt, *images], system_prompt=system) if client else ""
    parsed = _parse_json_object(raw)
    verification = {
        "source": "vlm_region",
        "prompt": prompt,
        "system_prompt": system,
        "raw": raw,
        "image_order": ["rgb_file"] + (["correction_rgb_file"] if correction else []),
        "valid": False,
    }
    return parsed, verification


def select_supported_region(rgb, depth, query, description, *, client, min_depth, max_depth, strategy="point"):
    """One geometry-feedback correction at most; semantic abstentions stand."""
    if strategy == "depth_candidates":
        return select_candidate_surface(
            rgb, depth, query, description, client=client, min_depth=min_depth, max_depth=max_depth
        )
    if strategy != "point":
        raise ValueError(f"Unknown region strategy: {strategy}")
    correction = None
    attempts = []
    mask = np.full(depth.shape, -1, dtype=np.int32)
    for _ in range(2):
        parsed, audit = select_vlm_region(rgb, query, description, client=client, correction=correction)
        attempts.append({**audit, "selection": parsed})
        if parsed.get("verified") is not True:
            audit["reason"] = "VLM abstained or returned invalid output"
            attempts[-1]["reason"] = audit["reason"]
            break
        try:
            mask = region_depth_mask(
                depth, parsed.get("box", []), parsed.get("point", []), min_depth=min_depth, max_depth=max_depth
            )
            audit["valid"] = True
            attempts[-1]["valid"] = True
            break
        except (ValueError, TypeError) as exc:
            audit["reason"] = str(exc)
            attempts[-1]["reason"] = str(exc)
            if correction is None:
                correction = {"region": parsed, "error": str(exc)}
    audit["attempts"] = attempts
    audit["correction"] = correction
    return parsed, mask, audit


def select_candidate_surface(rgb, depth, query, description, *, client, min_depth, max_depth):
    from emet.memory.surface_candidates import candidate_mask, surface_candidate_image, surface_candidates

    parsed, audit = select_vlm_region(rgb, query, description, client=client, box_only=True)
    mask = np.full(depth.shape, -1, dtype=np.int32)
    audit.update(strategy="depth_candidates", valid=False)
    if parsed.get("verified") is not True:
        audit["reason"] = "VLM abstained or returned invalid output"
        return parsed, mask, audit
    try:
        regions = surface_candidates(depth, parsed.get("box"), min_depth=min_depth, max_depth=max_depth, rgb=rgb)
    except (TypeError, ValueError) as exc:
        audit["reason"] = str(exc)
        return parsed, mask, audit
    audit["surface_candidates"] = regions
    if not regions:
        audit["reason"] = "no supported surfaces; another view is needed"
        return parsed, mask, audit
    prompt = (
        f"Select a measured surface of {description or query!r}. Image 1 is the original; image 2 "
        "shows enlarged panels of the same crop: an original crop, then one panel per candidate. "
        "In each candidate panel, ONLY the original-brightness pixels define the candidate surface; "
        "dimmed pixels are outside that candidate. These unlabelled geometry proposals may include "
        "table, wall, occluders, or mixed objects. Select only a region whose bright pixels belong "
        "to the requested object, not its support furniture. A partial visible object surface is "
        "sufficient. If a region mixes target and other objects, or the target is ambiguous, abstain. "
        f"Available IDs: {[r['id'] for r in regions]}. "
        'Return JSON {"selected_id":integer or null,"target_unambiguous":true or false,"reason":"..."}.'
    )
    system = "Choose a visually supported surface, not an object location guess. Reply with JSON only."
    raw = _call_eqa_client(
        client, [prompt, Image.fromarray(rgb), surface_candidate_image(rgb, regions)], system_prompt=system
    )
    selection = _parse_json_object(raw)
    chosen = selection.get("selected_id")
    audit["surface_selection"] = {
        "prompt": prompt,
        "system_prompt": system,
        "raw": raw,
        "image_order": ["rgb_file", "surface_candidates_rgb_file"],
    }
    if selection.get("target_unambiguous") is not True or type(chosen) is not int or not 0 <= chosen < len(regions):
        audit["reason"] = "no unambiguous surface selected; another view is needed"
        return parsed, mask, audit
    mask[candidate_mask(regions[chosen], depth.shape)] = 0
    audit.update(valid=True, selected_id=chosen, proposal_source="rgbd_components")
    return parsed, mask, audit


def ground_vlm_region(frame, query, description, *, client, min_depth, max_depth, strategy="point"):
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
    parsed, detected.instance, verification = select_supported_region(
        rgb, depth, query, description, client=client, min_depth=min_depth, max_depth=max_depth, strategy=strategy
    )
    if not verification["valid"]:
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
