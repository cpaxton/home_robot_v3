# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Observation-local semantic selection and offline admission replay records."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw


def proposal_view_context(support, frame, *, source_obs_id):
    """Attach view-bound image bearings, never an object XYZ or a safe route.

    Even a rejected proposal may help a coordinator choose another observation.
    Bearings describe its image support, not its semantic identity or visibility
    from a different pose. The caller owns the capture index and frame basis.
    """
    if not isinstance(support, dict):
        return support
    out = deepcopy(support)
    height, width = frame.depth.shape
    info = getattr(frame, "info", None)
    info = info if isinstance(info, dict) else {}
    out["view"] = {
        "source_obs_id": source_obs_id,
        "image_shape_hw": [height, width],
        "observation_scope_id": info.get("observation_scope_id"),
        "image_timing": deepcopy(info.get("image_timing")),
        "bearing_semantics": "image proposals only; positive base yaw is left/CCW; not object geometry or route permission",
    }

    def array(value):
        try:
            return np.asarray(value.tolist() if hasattr(value, "tolist") else value, dtype=float)
        except (TypeError, ValueError):
            return np.asarray(None, dtype=float)

    intrinsic = array(getattr(frame, "camera_K", None))
    pose = array(getattr(frame, "camera_pose", None))
    base = array(getattr(frame, "base_pose", None))
    calibrated = (
        intrinsic.shape == (3, 3)
        and np.isfinite(intrinsic).all()
        and intrinsic[0, 0] > 0
        and intrinsic[1, 1] > 0
        and np.allclose(intrinsic[2], [0, 0, 1])
        and abs(np.linalg.det(intrinsic)) > 1e-12
    )
    world_basis = (
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and np.allclose(pose[3], [0, 0, 0, 1])
        and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
        and base.shape == (3,)
        and np.isfinite(base).all()
    )
    for proposal in out.get("proposals", []):
        proposal["base_yaw_delta_deg"] = None
        box = array(proposal.get("bbox_xyxy"))
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
        ):
            proposal["image_geometry_status"] = "invalid_box"
            continue
        center = (box[:2] + box[2:]) / 2
        proposal["image_center_normalized_xy"] = (1000 * center / [width, height]).tolist()
        proposal["touches_image_edge"] = bool(box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height)
        if not calibrated:
            proposal["image_geometry_status"] = "calibration_unavailable"
            continue
        ray = np.linalg.solve(intrinsic, [*center, 1.0])
        proposal["optical_offset_deg"] = {
            "right": float(np.degrees(np.arctan2(ray[0], ray[2]))),
            "up": float(np.degrees(np.arctan2(-ray[1], np.hypot(ray[0], ray[2])))),
        }
        proposal["image_geometry_status"] = "optical_only"
        if world_basis:
            direction = pose[:3, :3] @ ray
            if np.linalg.norm(direction[:2]) > 1e-9:
                yaw = np.arctan2(direction[1], direction[0]) - base[2]
                proposal["base_yaw_delta_deg"] = float(np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw))))
                proposal["image_geometry_status"] = "base_bearing_available"
    return out


def numbered_detection_image(rgb, detections):
    """The exact numbered image sent to the verifier and retained for review."""
    annotated = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        box = tuple(detection["bbox_xyxy"])
        draw.rectangle(box, outline="yellow", width=2)
        draw.text(box[:2], str(detection["instance_id"]), fill="yellow")
    return annotated


def select_query_detections(query, description, detections, rgb, client=None):
    """Return explicitly verified detection IDs, not approximate label matches.

    Bare exact labels need no extra model call. Compound/relational descriptions
    require a fresh image judgment; unavailable or malformed inference abstains.
    """
    exact = [d["instance_id"] for d in detections if d["label_short"].strip().lower() == query]
    if not description and len(query.split()) == 1 and exact:
        return exact, {"source": "exact_label"}
    if client is None or not detections:
        return [], {"source": "unverified", "reason": "semantic verification unavailable"}
    from emet.eval.agentic_vlm_assess import _call_eqa_client, _parse_json_object

    annotated = numbered_detection_image(rgb, detections)
    catalog = [{"id": d["instance_id"], "class": d["label_short"]} for d in detections]
    prompt = (
        f"Find the object referred to by: {description or query!r}. Retrieval hint: {query!r}. "
        f"Numbered detector regions: {json.dumps(catalog)}. "
        "Use the original image and the numbered copy to select the target object, not its support furniture. "
        "Verify all requested attributes and spatial relationships from pixels. Detector labels and retrieval "
        "hints are proposals, not evidence. For a question, identify its referent; do not answer the question. "
        'Return JSON {"matching_ids": [integers], "constraints_verified": true or false}. '
        "List ALL matching regions; ambiguity must not be resolved by arbitrarily picking one. "
        "If the referent or any required constraint cannot be verified, return false and an empty list."
    )
    system = "Verify a robot target against fresh visual evidence. Reply only with JSON."
    raw = _call_eqa_client(
        client,
        [prompt, Image.fromarray(rgb), annotated],
        system_prompt=system,
    )
    parsed = _parse_json_object(raw)
    ids = parsed.get("matching_ids")
    allowed = {d["instance_id"] for d in detections}
    valid = (
        parsed.get("constraints_verified") is True
        and isinstance(ids, list)
        and all(type(i) is int and i in allowed for i in ids)
        and len(set(ids)) == len(ids)
    )
    return (ids if valid else []), {
        "source": "fresh_vlm",
        "raw": raw,
        "valid": valid,
        "prompt": prompt,
        "system_prompt": system,
        "image_order": ["rgb_file", "numbered_rgb_file"],
    }


def cache_grounding_record(
    directory,
    *,
    query,
    revision,
    source_obs_id,
    detections,
    matching_ids,
    verification,
    rgb=None,
    depth=None,
    masks=None,
    metadata=None,
):
    """Write immutable, pre-admission scores; never consumed by live grounding."""
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    prefix = f"grounding-{uuid4().hex}"
    record = {
        "schema_version": 1,
        "query": query,
        "observation_revision": revision,
        "source_obs_id": source_obs_id,
        "detections": detections,
        "matching_ids": matching_ids,
        "verification": verification,
        "metadata": metadata or {},
    }
    if rgb is not None:
        Image.fromarray(rgb).save(path / f"{prefix}.png")
        record["rgb_file"] = f"{prefix}.png"
        numbered_detection_image(rgb, detections).save(path / f"{prefix}-numbered.png")
        record["numbered_rgb_file"] = f"{prefix}-numbered.png"
        if verification.get("surface_candidates"):
            from emet.memory.surface_candidates import surface_candidate_image, surface_candidate_panels

            surface_candidate_image(rgb, verification["surface_candidates"]).save(path / f"{prefix}-surfaces.png")
            record["surface_candidates_rgb_file"] = f"{prefix}-surfaces.png"
            for region, panel in zip(
                verification["surface_candidates"],
                surface_candidate_panels(rgb, verification["surface_candidates"]),
                strict=True,
            ):
                filename = f"{prefix}-surface-{region['id']}.png"
                panel.save(path / filename)
                record[f"surface_candidate_{region['id']}_rgb_file"] = filename
                if verification.get("surface_selection", {}).get("presentation") == "context":
                    from emet.memory.surface_candidates import context_panel

                    context_filename = f"{prefix}-surface-{region['id']}-context.png"
                    context_panel(rgb, region).save(path / context_filename)
                    record[f"surface_candidate_{region['id']}_context_rgb_file"] = context_filename
        if verification.get("correction"):
            from emet.memory.vlm_region_grounding import region_annotation

            region_annotation(rgb, verification["correction"]["region"]).save(path / f"{prefix}-correction.png")
            record["correction_rgb_file"] = f"{prefix}-correction.png"
    arrays = {}
    for name, value in (("depth", depth), ("masks", masks)):
        if value is not None:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            arrays[name] = np.asarray(value)
    if arrays:
        np.savez_compressed(path / f"{prefix}.npz", **arrays)
        record["arrays_file"] = f"{prefix}.npz"
    with (path / f"{prefix}.json").open("x") as stream:
        json.dump(record, stream, allow_nan=False)
    return str(path / f"{prefix}.json")


def replay_grounding_admission(record, config):
    """Replay thresholds on a fixed observed candidate set, not navigation/recall."""
    from emet.memory.graph_eqa.ingest.instance_observations import filter_detections_for_graph_admission

    if record.get("schema_version") != 1:
        raise ValueError("Unsupported grounding cache schema")
    admitted, _ = filter_detections_for_graph_admission(record["detections"], config=config)
    matches = record["matching_ids"]
    return bool(
        config.enabled
        and config.use_instance_nodes
        and len(matches) == 1
        and any(d["instance_id"] == matches[0] for d in admitted)
    )
