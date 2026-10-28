#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Build 12 controlled dev snapshots. CPU geometry checks; optional sensor render."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from emet.eval.embodied_qa_datasets import EvaluationQuestion
from emet.eval.embodied_qa_evidence import bind_recorded_evidence
from emet.eval.harness_capability_fixtures import load_development_cases, qualify_geometry, snapshot_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/harness_capability_dev.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--render", action="store_true", help="Render RGB-D and scorer-only segmentation; requires GL. No VLM."
    )
    args = parser.parse_args()
    cases = load_development_cases(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "benchmark": "harness-capability-dev", "episodes": {}}
    rows = []
    for case in cases:
        result = qualify_geometry(case)
        case_dir = args.output / case["id"]
        case_dir.mkdir()
        frames = []
        for index in range(2):
            model, data, xml = snapshot_model(case, index)
            (case_dir / f"snapshot-{index}.xml").write_text(xml)
            if not args.render:
                continue
            height, width = 384, 512
            with mujoco.Renderer(model, height=height, width=width) as renderer:
                renderer.update_scene(data, camera="evidence")
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
            rgb_path = case_dir / f"rgb-{index}.png"
            Image.fromarray(rgb).save(rgb_path)
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
            mask = (
                (segmentation[..., 0] == target_id) & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                if target_id >= 0
                else np.zeros(depth.shape, dtype=bool)
            )
            pose = np.eye(4)
            camera = model.camera("evidence").id
            pose[:3, :3] = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
            pose[:3, 3] = data.cam_xpos[camera]
            focal = 0.5 * height / np.tan(np.radians(model.cam_fovy[camera]) / 2)
            intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
            np.savez_compressed(
                case_dir / f"scorer-{index}.npz", depth=depth, target_mask=mask, camera_pose=pose, camera_K=intrinsics
            )
            frames.append(
                {
                    "index": index,
                    "rgb": str(rgb_path.relative_to(args.output)),
                    "sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                }
            )
            result.setdefault("target_pixels", []).append(int(mask.sum()))
        if args.render:
            manifest["episodes"][case["id"]] = {
                "evidence_id": case["id"],
                "scene": "controlled-development",
                "frames": frames,
            }
        rows.append(result)
    if args.render:
        for case in cases:
            question = EvaluationQuestion(
                "harness-capability-dev",
                case["id"],
                "development",
                case["question"],
                case["family"],
                "controlled-development",
                case["expected"],
                (0, 1),
                case["id"],
            )
            packet = bind_recorded_evidence(question, manifest, root=args.output)
            assert len(packet.model_input(max_images=2)) == 3
        (args.output / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        sha = None
    report = {
        "schema_version": 1,
        "source_sha": sha,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "CPU geometry and optional recorded RGB-D only; no active robot, VLM, manipulation or external benchmark acceptance",
        "rendered": args.render,
        "episodes": rows,
    }
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "rendered": args.render, "output": str(args.output)}))


if __name__ == "__main__":
    main()
