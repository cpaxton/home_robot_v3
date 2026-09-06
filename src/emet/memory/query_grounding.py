# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Observation-local semantic selection and offline admission replay records."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw


def select_query_detections(query, description, detections, rgb, client=None):
    """Return explicitly verified detection IDs, not approximate label matches.

    Bare exact labels need no extra model call. Compound/relational descriptions
    require a fresh image judgment; unavailable or malformed inference abstains.
    """
    exact = [d["instance_id"] for d in detections if d["label_short"].strip().lower() == query]
    if not description and exact:
        return exact, {"source": "exact_label"}
    if client is None or not detections:
        return [], {"source": "unverified", "reason": "semantic verification unavailable"}
    from emet.eval.agentic_vlm_assess import _call_eqa_client, _parse_json_object

    annotated = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(annotated)
    for d in detections:
        box = tuple(d["bbox_xyxy"])
        draw.rectangle(box, outline="yellow", width=2)
        draw.text(box[:2], str(d["instance_id"]), fill="yellow")
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
    raw = _call_eqa_client(
        client,
        [prompt, Image.fromarray(rgb), annotated],
        system_prompt="Verify a robot target against fresh visual evidence. Reply only with JSON.",
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
    return (ids if valid else []), {"source": "fresh_vlm", "raw": raw, "valid": valid}


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
