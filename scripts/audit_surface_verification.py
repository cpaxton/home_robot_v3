#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Replay frozen surface proposals, isolating batch versus individual verification.

Consumes artifacts from the measured-support diagnostic, not simulator labels.
Uses the SAME saved panels, prompt, geometry, anchors and scoring in both modes.
Repeated calls on these development views are not independent task episodes.
No runtime policy or placement safety setting is changed.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from audit_placement_relations import overlay, score, spatial_selection

PROMPT = (
    "Identify which measured candidates belong to a countertop top surface. Image 1 is the full reference scene, NOT a candidate. "
    "Each subsequent pair shows one numbered candidate outlined in context, then isolated measured pixels. "
    "Countertop support furniture is valid. Reject stove surfaces, cabinets, walls, appliances, and candidates mixing them with countertops. "
    "A partial visible countertop surface is sufficient if its identity is unambiguous. Black pixels are missing data. "
    'Do not select based on a spatial relation; none is requested. Return {"countertop_ids":[integer],"reason":"short explanation"}. '
    "Return an empty list if none can be identified confidently."
)


def accepted_ids(parsed, allowed):
    """Malformed responses fail the entire view, including negative cases."""
    ids = parsed.get("countertop_ids")
    if (
        not isinstance(ids, list)
        or any(type(i) is not int or i not in allowed for i in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("invalid semantic output")
    return ids


def selected_boxes(regions, ids, width, height):
    return [
        [v * 1000 / scale for v, scale in zip(r["bbox_xyxy"], [width, height, width, height], strict=True)]
        for r in regions
        if r["id"] in ids
    ]


def main():
    from PIL import Image

    from emet.core.parameters import get_parameters
    from emet.eval.agentic_vlm_assess import _parse_json_object
    from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/eval/grounding_best_local.yaml")
    parser.add_argument("--modes", nargs="+", choices=["batch", "individual"], default=["batch", "individual"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--views", nargs="+", help="Optional development smoke subset")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rows = json.loads((args.source / "manifest.json").read_text())
    if args.views:
        if set(args.views) - {r["name"] for r in rows}:
            parser.error("unknown view")
        rows = [r for r in rows if r["name"] in args.views]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parameters = get_parameters(args.config)
    protocol = {
        "source": str(args.source.resolve()),
        "config": args.config,
        "resolved_config": parameters.data,
        "repeats": args.repeats,
        "modes": args.modes,
        "views": [r["name"] for r in rows],
        "prompt": PROMPT,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "iou_threshold": 0.3,
        "center_margin": 50,
        "note": "Development replay only; no simulator labels in inference; geometry and panel pixels frozen.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.output_dir / "manifest.json").write_text(json.dumps(rows, indent=2))
    (args.output_dir / "driver.py").write_text(Path(__file__).read_text())
    _, client = build_graph_eqa_vlm_clients(parameters=parameters)
    results = []
    for repeat in range(args.repeats):
        for row in rows:
            folders = sorted(args.source.glob(f"[0-9]*-{row['name']}"))
            if not folders:
                raise ValueError(f"no frozen proposals for {row['name']}")
            for source in folders:
                proposal = json.loads((source / "proposals.json").read_text())
                regions = proposal["regions"]
                image = Image.open(row["rgb"]).convert("RGB")
                # Alternate order to reduce systematic time/order confounding.
                modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
                for mode in modes:
                    folder = args.output_dir / f"{repeat}-{source.name}-{mode}"
                    folder.mkdir()
                    (folder / "proposals.json").write_text(json.dumps(proposal, indent=2))
                    ids, requests = [], []
                    error = proposal.get("error")
                    groups = [regions] if mode == "batch" else [[r] for r in regions]
                    for group in groups if not error and regions else []:
                        panels, hashes = [], {}
                        for region in group:
                            for index in (2 * region["id"], 2 * region["id"] + 1):
                                path = source / f"panel-{index}.png"
                                panels.append(Image.open(path).convert("RGB"))
                                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                        start = time.monotonic()
                        raw = client(
                            [PROMPT, image, *panels],
                            system_prompt="Inspect visual evidence carefully. Return JSON only.",
                            max_new_tokens=512,
                        )
                        parsed = _parse_json_object(raw)
                        requests.append(
                            {
                                "ids": [r["id"] for r in group],
                                "raw": raw,
                                "parsed": parsed,
                                "elapsed_s": time.monotonic() - start,
                                "panel_sha256": hashes,
                            }
                        )
                        try:
                            ids.extend(accepted_ids(parsed, {r["id"] for r in group}))
                        except ValueError as exc:
                            error = str(exc)
                            break
                    if error:
                        ids = []
                    ids = [r["id"] for r in regions if r["id"] in ids]
                    boxes = selected_boxes(regions, ids, image.width, image.height)
                    observed = {**proposal["cached_localization"], "countertops": boxes}
                    (folder / "semantic.json").write_text(
                        json.dumps(
                            {
                                "parsed": {"countertop_ids": ids},
                                "requests": requests,
                                "selection_input": observed,
                                "error": error,
                            },
                            indent=2,
                        )
                    )
                    overlay(
                        image, [(str(i), b, "lime") for i, b in zip(ids, boxes, strict=True)], folder / "localized.png"
                    )
                    for case in row["cases"]:
                        chosen = spatial_selection(observed, case["anchor"], case["relation"]) if not error else None
                        scored = score(chosen, case["expected_boxes"])
                        scored["correct"] = scored["correct"] and not error
                        result = dict(
                            repeat=repeat,
                            source_view=source.name,
                            view=row["name"],
                            variant=mode,
                            query=f"countertop {case['relation']} of {case['anchor']}",
                            box=chosen,
                            semantic_ids=ids,
                            pipeline_error=error,
                            rgb_sha256=hashlib.sha256(Path(row["rgb"]).read_bytes()).hexdigest(),
                            **scored,
                        )
                        results.append(result)
                        print(json.dumps(result), flush=True)
                    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
