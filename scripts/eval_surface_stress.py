#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Paired, serial perception diagnostic; ground truth enters only post-hoc scoring.

No robot actuation. This does not measure exploration or pick/place success.
Outputs retain rendered RGB-D, exact selections, and independent mask scores.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from probe_stationary_objects import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ovmm/surface_stress.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    spec = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load(Path(spec["base"]).read_text())
    manifest, truth = [], []
    for variant in spec["variants"]:
        config = copy.deepcopy(base)
        config.update({k: v for k, v in variant.items() if k not in ("name", "targets")})
        for target in config["targets"]:
            target.update(variant.get("targets", {}).get(target["body"], {}))
        folder = args.output_dir / variant["name"]
        run(config, folder)  # Visibility failure is measured, not silently dropped.
        report = json.loads((folder / "result.json").read_text())
        for query in [t["query"] for t in config["targets"]] + config["negative_queries"]:
            manifest.append({"arrays": str(folder / "frame.npz"), "query": query})
            target = next((t for t in report["targets"] if t["query"] == query), None)
            center = next((t["position"] for t in config["targets"] if t["query"] == query), None)
            truth.append({"variant": variant["name"], "query": query, "target": target, "center": center})
    manifest_path = args.output_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    (args.output_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    scores = []
    for strategy in ("point", "depth_candidates"):
        output = args.output_dir / strategy
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_vlm_regions.py",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output),
                "--strategy",
                strategy,
            ],
            check=True,
        )
        results = json.loads((output / "results.json").read_text())
        for index, (result, expected) in enumerate(zip(results, truth, strict=True)):
            accepted = result["audit"]["valid"]
            target = expected["target"]
            score = {
                **expected,
                "strategy": strategy,
                "accepted": accepted,
                "purity": None,
                "visible_recall": None,
                "center_error_m": None,
            }
            if accepted and target is not None:
                with np.load(manifest[index]["arrays"]) as frame:
                    seg = frame["segmentation"]
                    mask_gt = (seg[..., 0] == target["geometry_id"]) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
                with np.load(output / f"{index}-support.npz") as support:
                    mask = support["mask"]
                score["purity"] = float(mask_gt[mask].mean())
                score["visible_recall"] = float((mask_gt & mask).sum() / max(1, mask_gt.sum()))
                score["center_error_m"] = float(np.linalg.norm(np.array(result["xyz"]) - expected["center"]))
            # Preregistered diagnostic gate, not a relaxed action tolerance.
            score["passed"] = (
                not accepted
                if target is None or target["visible_pixels"] == 0
                else accepted and score["purity"] >= 0.95 and score["center_error_m"] <= 0.15
            )
            scores.append(score)
        (args.output_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
