#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Cache controlled scene views. Simulator labels are saved separately from inference inputs.

This is an oracle-aimed perception diagnostic, NOT robot search/navigation evidence.
Assets stay in their original scenes; no physics or robot actuation is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image


def object_geom_ids(model, root):
    """Include every descendant body, not just the first geometry of an object."""
    bodies = {int(root)}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))


def capture(model, data, camera, height, width):
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    option = mujoco.MjvOption()
    # Visual geometry only; exclude conventional collision-only groups 3..5.
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
        cams = renderer.scene.camera
        position = (cams[0].pos + cams[1].pos) / 2
        forward = np.asarray(cams[0].forward, dtype=float)
        up = np.asarray(cams[0].up, dtype=float)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([np.cross(forward, up), -up, forward])
        pose[:3, 3] = position
        focal = height * cams[0].frustum_near / (cams[0].frustum_top - cams[0].frustum_bottom)
        K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    return rgb, depth, K, pose, segmentation


def build(config, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest, truth = [], []
    for scene in config["scenes"]:
        if scene.get("kind") == "robocasa":
            from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

            model, xml, placements = model_generation_wizard(
                task=scene.get("task", "PickPlaceCounterToCabinet"),
                layout=scene["layout"],
                style=scene["style"],
                seed=scene["seed"],
                robot="rby1",
            )
            source = output / f"{scene['name']}.xml"
            source.write_text(xml)
            scene["targets"] = [
                {"name": name, "query": info["cat"].replace("_", " "), "body_prefix": name}
                for name, info in placements.items()
                if not name.startswith("_emet_")
            ]
        else:
            source = Path(scene["xml"]).expanduser().resolve()
            model = mujoco.MjModel.from_xml_path(str(source))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for target in scene["targets"]:
            matches = [
                i
                for i in range(1, model.nbody)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(target["body_prefix"])
            ]
            # Root match must be explicit and unique; never guess between instances.
            roots = [i for i in matches if int(model.body_parentid[i]) not in matches]
            if len(roots) != 1:
                raise ValueError(f"{target['body_prefix']}: expected one root, found {roots}")
            gids = object_geom_ids(model, roots[0])
            visual = gids[np.isin(model.geom_group[gids], [0, 1, 2])]
            if not len(visual):
                raise ValueError(f"No visual geometry: {target}")
            center = np.mean(data.geom_xpos[visual], axis=0)
            for view in config["views"]:
                folder = output / scene["name"] / target["name"] / view["name"]
                folder.mkdir(parents=True)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = center + np.asarray(view.get("lookat_offset", [0, 0, 0]))
                camera.distance = view["distance"]
                camera.azimuth = view["azimuth"]
                camera.elevation = view["elevation"]
                height, width = view.get("image_size", [480, 640])
                rgb, depth, K, pose, seg = capture(model, data, camera, height, width)
                mask = np.isin(seg[..., 0], gids) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
                Image.fromarray(rgb).save(folder / "rgb.png")
                # No GT arrays in the inference file, even if an audit reader changes.
                np.savez_compressed(folder / "frame.npz", rgb=rgb, depth=depth, camera_K=K, camera_pose=pose)
                np.savez_compressed(folder / "truth.npz", mask=mask, segmentation=seg)
                manifest.append({"arrays": str(folder / "frame.npz"), "query": target["query"]})
                truth.append(
                    {
                        "input": manifest[-1],
                        "mask_file": str(folder / "truth.npz"),
                        "scene": scene["name"],
                        "split": scene["split"],
                        "target": target["name"],
                        "view": view,
                        "visible_pixels": int(mask.sum()),
                        "geometry_ids": gids.tolist(),
                        "source_xml": str(source),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "mode": "oracle_aimed_free_camera_no_physics",
                    }
                )
                print(f"{scene['name']}/{target['name']}/{view['name']}: {mask.sum()} target pixels", flush=True)
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (output / "truth.json").write_text(json.dumps(truth, indent=2))
    (output / "config.yaml").write_text(yaml.safe_dump(config))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(yaml.safe_load(args.config.read_text()), args.output_dir)
