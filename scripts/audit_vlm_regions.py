#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Replay fixed RGB-D inputs through the production Qwen region selector.

Manifest: list of {arrays: NPZ, rgb: optional PNG, query, description: optional}.
NPZ contains depth and optionally rgb/camera_K/camera_pose. No evaluator labels
or simulator segmentation are passed to Qwen. Missing calibration means no XYZ
claim, not guessed intrinsics. This is a perception audit, not task success.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from emet.core.parameters import get_parameters
from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients
from emet.memory.vlm_region_grounding import region_depth_mask, select_vlm_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    params = get_parameters("dynav_config.yaml")
    _, client = build_graph_eqa_vlm_clients(parameters=params)
    rows = yaml.safe_load(args.manifest.read_text())
    results = []
    for index, row in enumerate(rows):
        with np.load(row["arrays"], allow_pickle=False) as arrays:
            rgb = np.asarray(Image.open(row["rgb"]).convert("RGB")) if row.get("rgb") else arrays["rgb"]
            depth = arrays["depth"]
            parsed, audit = select_vlm_region(rgb, row["query"], row.get("description", row["query"]), client=client)
            result = {"input": row, "selection": parsed, "audit": audit, "surface_points": 0, "xyz": None}
            overlay = Image.fromarray(rgb)
            if parsed.get("verified") is True:
                try:
                    mask = (
                        region_depth_mask(
                            depth, parsed.get("box", []), parsed.get("point", []), min_depth=0.25, max_depth=4.5
                        )
                        == 0
                    )
                    result["surface_points"] = int(mask.sum())
                    np.savez_compressed(args.output_dir / f"{index}-support.npz", mask=mask)
                    if "camera_K" in arrays and "camera_pose" in arrays:
                        yy, xx = np.indices(depth.shape)
                        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
                        camera = (pixels @ np.linalg.inv(arrays["camera_K"]).T) * depth[..., None]
                        pose = arrays["camera_pose"]
                        world = camera @ pose[:3, :3].T + pose[:3, 3]
                        result["xyz"] = np.median(world[mask], axis=0).tolist()
                    h, w = depth.shape
                    box = np.asarray(parsed["box"]) * [w, h, w, h] / 1000
                    ImageDraw.Draw(overlay).rectangle(tuple(box), outline="yellow", width=3)
                except (ValueError, TypeError) as exc:
                    result["geometry_error"] = str(exc)
            overlay.save(args.output_dir / f"{index}-region.png")
            results.append(result)
            (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
