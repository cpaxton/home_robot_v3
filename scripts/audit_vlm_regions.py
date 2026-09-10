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
from PIL import Image

from emet.core.parameters import get_parameters
from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients
from emet.memory.vlm_region_grounding import region_annotation, select_supported_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynav_config.yaml", help="Explicit model configuration")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=["point", "depth_candidates"], default="point")
    parser.add_argument("--remote-image-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--proposal-cache", type=Path, help="Cached external masks; Qwen performs final selection")
    parser.add_argument(
        "--box-ablation", choices=["baseline", "expand_25", "whole_object", "verify", "repair"], default="baseline"
    )
    parser.add_argument(
        "--replay-boxes",
        type=Path,
        help="Reuse search-box responses from prior results.json; surface-selection ablation only",
    )
    parser.add_argument("--depth-noise-std-m", type=float, default=0.0)
    parser.add_argument("--depth-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.proposal_cache and (
        args.box_ablation != "baseline" or args.replay_boxes or args.strategy != "depth_candidates"
    ):
        parser.error("external proposals require baseline depth_candidates without replayed boxes")
    if args.box_ablation != "baseline" and args.strategy != "depth_candidates":
        parser.error("box ablations require depth_candidates")
    if args.box_ablation == "whole_object" and args.replay_boxes:
        parser.error("prompt ablation requires fresh localization")
    if args.replay_boxes and args.strategy != "depth_candidates":
        parser.error("--replay-boxes requires depth_candidates")
    if not np.isfinite(args.depth_noise_std_m) or args.depth_noise_std_m < 0 or not 0 <= args.depth_dropout <= 1:
        parser.error("depth noise must be finite/nonnegative and dropout in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = yaml.safe_load(args.manifest.read_text())
    proposals = json.loads((args.proposal_cache / "manifest.json").read_text()) if args.proposal_cache else None
    if proposals is not None and [r["input"] for r in proposals] != rows:
        raise ValueError("proposal cache must match exact inputs and order")
    replay = json.loads(args.replay_boxes.read_text()) if args.replay_boxes else None
    if replay is not None and (
        len(replay) != len(rows) or any(r["input"] != row for r, row in zip(replay, rows, strict=True))
    ):
        raise ValueError("replayed boxes must match the exact manifest and order")
    params = get_parameters(args.config)
    params.set("eqa/vl_image_format", args.remote_image_format)
    _, client = build_graph_eqa_vlm_clients(parameters=params)
    results = []
    for index, row in enumerate(rows):
        first_call = True

        def selection_client(*call_args, **kwargs):
            nonlocal first_call
            if first_call and replay is not None:
                first_call = False
                return json.dumps(replay[index]["selection"])
            first_call = False
            return client(*call_args, **kwargs)

        with np.load(row["arrays"], allow_pickle=False) as arrays:
            rgb = np.asarray(Image.open(row["rgb"]).convert("RGB")) if row.get("rgb") else arrays["rgb"]
            depth = arrays["depth"]
            perturbation = {"noise_std_m": args.depth_noise_std_m, "dropout": args.depth_dropout, "seed": args.seed}
            if args.depth_noise_std_m or args.depth_dropout:
                # Reset per image so repeated queries see identical corrupted
                # evidence. This is a synthetic stress test, not sensor realism.
                rng = np.random.default_rng(args.seed)
                depth = depth.astype(float).copy()
                valid = np.isfinite(depth) & (depth > 0)
                noise = rng.normal(0, args.depth_noise_std_m, depth.shape)
                depth[valid] += noise[valid]
                depth[rng.random(depth.shape) < args.depth_dropout] = np.nan
                np.savez_compressed(args.output_dir / f"{index}-input-depth.npz", depth=depth)
            from grounding_ablation import BoxAblation

            ablation = BoxAblation(selection_client, rgb, row["query"], args.box_ablation)
            if proposals is None:
                parsed, support, audit = select_supported_region(
                    rgb,
                    depth,
                    row["query"],
                    row.get("description", row["query"]),
                    client=ablation,
                    min_depth=0.25,
                    max_depth=4.5,
                    strategy=args.strategy,
                )
            else:
                from emet.memory.vlm_region_grounding import select_candidate_surface

                # No synthetic localization call: the next call is surface verification.
                ablation.first = False
                with np.load(args.proposal_cache / f"{index}-masks.npz", allow_pickle=False) as cached:
                    masks = cached["masks"]
                parsed, support, audit = select_candidate_surface(
                    rgb,
                    depth,
                    row["query"],
                    row.get("description", row["query"]),
                    client=ablation,
                    min_depth=0.25,
                    max_depth=4.5,
                    proposal_masks=masks,
                )
            result = {
                "input": row,
                "box_ablation": args.box_ablation,
                "proposal_cache": str(args.proposal_cache) if args.proposal_cache else None,
                "effective_requests": ablation.requests,
                "remote_image_format": args.remote_image_format,
                "replayed_box_source": str(args.replay_boxes) if args.replay_boxes else None,
                "depth_perturbation": perturbation,
                "selection": parsed,
                "audit": audit,
                "surface_points": 0,
                "xyz": None,
            }
            for name, image in ablation.verification_images.items():
                image.save(args.output_dir / f"{index}-{name}.png")
            overlay = region_annotation(rgb, parsed)
            if audit["valid"]:
                try:
                    mask = support == 0
                    result["surface_points"] = int(mask.sum())
                    np.savez_compressed(args.output_dir / f"{index}-support.npz", mask=mask)
                    if "camera_K" in arrays and "camera_pose" in arrays:
                        yy, xx = np.indices(depth.shape)
                        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
                        camera = (pixels @ np.linalg.inv(arrays["camera_K"]).T) * depth[..., None]
                        pose = arrays["camera_pose"]
                        world = camera @ pose[:3, :3].T + pose[:3, 3]
                        result["xyz"] = np.median(world[mask], axis=0).tolist()
                except (ValueError, TypeError) as exc:
                    result["geometry_error"] = str(exc)
            elif parsed.get("verified") is True:
                result["geometry_error"] = audit.get("reason")
            if audit.get("correction"):
                region_annotation(rgb, audit["correction"]["region"]).save(args.output_dir / f"{index}-correction.png")
            if audit.get("surface_candidates"):
                from emet.memory.surface_candidates import surface_candidate_image, surface_candidate_panels

                surface_candidate_image(rgb, audit["surface_candidates"]).save(
                    args.output_dir / f"{index}-surfaces.png"
                )
                for region, panel in zip(
                    audit["surface_candidates"], surface_candidate_panels(rgb, audit["surface_candidates"]), strict=True
                ):
                    panel.save(args.output_dir / f"{index}-surface-{region['id']}.png")
            overlay.save(args.output_dir / f"{index}-region.png")
            results.append(result)
            (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
