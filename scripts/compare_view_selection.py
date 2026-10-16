#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Compare selectors on one frozen image/metadata manifest, without inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from emet.memory.view_coverage import CoverageObjective
from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views


def compare_manifest(path: Path, *, max_images: int = 4) -> dict:
    manifest = json.loads(path.read_text())
    frames = []
    for row in manifest["frames"]:
        values = dict(row)
        image_path = path.parent / values.pop("image_path")
        with Image.open(image_path) as image:
            values["rgb"] = np.asarray(image.convert("RGB"))
            frames.append(ViewFrame(**values))
    objective = CoverageObjective(**manifest["coverage_objective"]) if "coverage_objective" in manifest else None
    policies = ["recency", "pose", "embedding", "coverage", "hybrid"]
    if objective is not None:
        policies.append("weighted_coverage")
        if all(f.cell_quality is not None for f in frames):
            policies.append("quality_coverage")
    return {
        policy: select_views(
            frames,
            current_id=manifest["current_id"],
            pre_turn_id=manifest.get("pre_turn_id"),
            config=SelectionConfig(
                **{**manifest.get("selection_config", {}), "policy": policy, "max_images": max_images}
            ),
            query_embedding=manifest.get("query_embedding"),
            embedding_space=manifest.get("embedding_space"),
            coverage_objective=objective,
        ).to_dict()
        for policy in policies
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(compare_manifest(args.manifest, max_images=args.max_images), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
