#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Paired full-image relation diagnostic; not a runtime placement policy.

Manifest rows: {name, rgb, cases: [{anchor, relation, expected_boxes}]}.
Boxes are normalized xyxy in [0, 1000]. Empty expected_boxes means abstain.
Evaluator annotations are never included in inference requests. Scores measure
box agreement only, not support geometry, reachability or placement success.
"""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path


def valid_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def valid_localization(observed):
    return all(
        isinstance(observed.get(key), list) and all(valid_box(b) for b in observed[key])
        for key in ("countertops", "stove", "refrigerator")
    )


def spatial_selection(observed, anchor, relation, margin=50):
    """Conservative 2D center ordering; no 3D or clearance claim."""
    anchors = observed.get(anchor, [])
    targets = observed.get("countertops", [])
    if not isinstance(anchors, list) or len(anchors) != 1 or not valid_box(anchors[0]):
        return None
    if not isinstance(targets, list) or not all(valid_box(box) for box in targets):
        return None
    if relation not in ("left", "right"):
        raise ValueError("Only explicit image-relative left/right is supported")
    center = (anchors[0][0] + anchors[0][2]) / 2
    matches = []
    for box in targets:
        delta = (box[0] + box[2]) / 2 - center
        if (relation == "right" and delta > margin) or (relation == "left" and delta < -margin):
            matches.append(box)
    return matches[0] if len(matches) == 1 else None


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union


def score(box, expected, threshold=0.3):
    if not expected:
        return {"correct": box is None, "iou": None, "negative": True}
    overlap = max((box_iou(box, truth) for truth in expected), default=0) if box is not None else 0
    return {"correct": overlap >= threshold, "iou": overlap, "negative": False}


def overlay(rgb, boxes, path):
    from PIL import ImageDraw

    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    for label, box, color in boxes:
        if not valid_box(box):
            continue
        pixels = [
            box[0] * image.width / 1000,
            box[1] * image.height / 1000,
            box[2] * image.width / 1000,
            box[3] * image.height / 1000,
        ]
        draw.rectangle(pixels, outline=color, width=3)
        draw.text((pixels[0] + 3, pixels[1] + 3), label, fill=color, stroke_width=1, stroke_fill="black")
    image.save(path)


def main():
    from PIL import Image

    from emet.core.parameters import get_parameters
    from emet.eval.agentic_vlm_assess import _parse_json_object
    from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/eval/grounding_best_local.yaml")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    for row in rows:
        for case in row["cases"]:
            assert case["relation"] in ("left", "right") and case["anchor"] in ("stove", "refrigerator")
            assert all(valid_box(b) for b in case["expected_boxes"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(rows, indent=2))
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "repeats": args.repeats,
                "center_margin": 50,
                "iou_threshold": 0.3,
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    _, client = build_graph_eqa_vlm_clients(parameters=get_parameters(args.config))
    results = []

    def call(prompt, image):
        start = time.monotonic()
        raw = client(
            [prompt, image], system_prompt="Inspect visual evidence carefully. Return JSON only.", max_new_tokens=512
        )
        return _parse_json_object(raw), {"prompt": prompt, "raw": raw, "elapsed_s": time.monotonic() - start}

    for repeat in range(args.repeats):
        for row in rows:
            rgb = Image.open(row["rgb"]).convert("RGB")
            folder = args.output_dir / f"{repeat}-{row['name']}"
            folder.mkdir()
            rgb.save(folder / "reference.png")
            observed, request = call(
                "Locate every separate visible countertop surface, the stove, and any refrigerator in this single full image. "
                "Box the visible countertop top surface, not cabinets beneath it. Box the visible stove appliance and refrigerator "
                "independently. Do not infer objects outside the image. Use an empty list for absent or uncertain categories. "
                "Coordinates are normalized 0..1000 in x_min,y_min,x_max,y_max order. "
                'Return {"countertops":[[x_min,y_min,x_max,y_max]],"stove":[[x_min,y_min,x_max,y_max]],"refrigerator":[[x_min,y_min,x_max,y_max]]}.',
                rgb,
            )
            (folder / "localization.json").write_text(json.dumps({"parsed": observed, "request": request}, indent=2))
            labels = []
            for key, color in [("countertops", "lime"), ("stove", "cyan"), ("refrigerator", "magenta")]:
                values = observed.get(key, [])
                if isinstance(values, list):
                    labels.extend((key, b, color) for b in values)
            overlay(rgb, labels, folder / "localized.png")
            for index, case in enumerate(row["cases"]):
                query = f"countertop {case['relation']} of {case['anchor']}"
                direct, direct_request = call(
                    f"Locate the {query} in this single full image. Left/right is relative to the full image. "
                    "Box its visible countertop top surface, not cabinets beneath it. The anchor must be visible, "
                    "and the relation unambiguous. If missing or ambiguous, abstain. Coordinates are normalized "
                    '0..1000 in xyxy order. Return {"verified":true,"box":[x_min,y_min,x_max,y_max]} '
                    'or {"verified":false,"reason":"short explanation"}.',
                    rgb,
                )
                direct_box = (
                    direct.get("box") if direct.get("verified") is True and valid_box(direct.get("box")) else None
                )
                computed = spatial_selection(observed, case["anchor"], case["relation"])
                for variant, box, requests in [
                    ("direct_full_image", direct_box, [direct_request]),
                    ("independent_boxes", computed, [request]),
                ]:
                    format_valid = (
                        valid_localization(observed)
                        if variant == "independent_boxes"
                        else direct.get("verified") is False
                        or (direct.get("verified") is True and valid_box(direct.get("box")))
                    )
                    scored = score(box, case["expected_boxes"])
                    scored["correct"] = scored["correct"] and format_valid
                    result = dict(
                        view=row["name"],
                        repeat=repeat,
                        query=query,
                        variant=variant,
                        box=box,
                        format_valid=format_valid,
                        **scored,
                        requests=requests,
                        rgb_sha256=hashlib.sha256(Path(row["rgb"]).read_bytes()).hexdigest(),
                    )
                    results.append(result)
                    overlay(
                        rgb,
                        [("prediction", box, "lime"), *(("reference", b, "orange") for b in case["expected_boxes"])],
                        folder / f"{index}-{variant}.png",
                    )
                    print(json.dumps({k: v for k, v in result.items() if k != "requests"}), flush=True)
                (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
