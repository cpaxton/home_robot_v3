#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Separate view coverage, target texture, and support contact in offline RGB-D."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image

from emet.eval.stretch_table_observation import _enclose_views, _measure_target
from emet.simulation.depth_registration import camera_opencv_pose, rendered_camera_intrinsics
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators

WIDTH, HEIGHT = 640, 480
FOV = 62.0
VIEWS = (
    ((.06, -.22, .80), (.06, -.64, .53)),
    ((.48, -.64, .72), (.06, -.64, .53)),
    ((.06, -1.06, .72), (.06, -.64, .53)),
    ((-.36, -.64, .72), (.06, -.64, .53)),
    ((.06, -.65, 1.15), (.06, -.64, .53)),
    ((.075, -.576, .72), (.08, -.55, .55)),
    ((.30, -.55, .585), (.08, -.55, .60)),
)


def vec(values):
    return " ".join(str(float(value)) for value in values)


def build_model(scene, output, *, textured):
    tree = ET.parse(scene)
    root = tree.getroot()
    for include in root.findall("include"):
        include.set("file", str((scene.parent / include.attrib["file"]).resolve()))
    world = root.find("worldbody")
    # Aim at the tabletop work area. No rendered segmentation IDs are used.
    for index, (position, focus) in enumerate(VIEWS):
        forward = np.array(focus) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        ET.SubElement(world, "camera", name=f"ablation_{index}", pos=vec(position),
                      xyaxes=vec(np.r_[right, up]), fovy=str(FOV))
    if textured:
        rng = np.random.default_rng(0)
        swatches = rng.integers(114, 255, (10, 10), dtype=np.uint8)
        pattern = np.zeros((10, 10, 3), dtype=np.uint8)
        pattern[..., 0] = swatches
        pattern[..., 1:] = (rng.random((10, 10, 2)) * swatches[..., None] / 3.7).astype(np.uint8)
        filename = output / "red-pattern.png"
        Image.fromarray(pattern).resize((224, 224), Image.Resampling.NEAREST).save(filename)
        asset = root.find("asset")
        ET.SubElement(asset, "texture", name="pattern_probe", type="2d", file=str(filename))
        ET.SubElement(asset, "material", name="pattern_probe", texture="pattern_probe",
                      texuniform="false", texrepeat="1 1", rgba="1 1 1 1")
        geom = world.find('./body[@name="object2"]/geom')
        geom.attrib.pop("rgba", None)
        geom.set("material", "pattern_probe")
    xml = output / ("textured.xml" if textured else "plain.xml")
    tree.write(xml)
    return mujoco.MjModel.from_xml_path(str(xml))


def captures(model, output, *, suspended):
    data = mujoco.MjData(model)
    data.ctrl[:] = model.key_ctrl[0]
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    mujoco.mj_step(model, data, nstep=round(2. / model.opt.timestep))
    if suspended:
        # Place the same body in free air for a static sensing-only control.
        joint = model.body("object2").jntadr[0]
        data.qpos[model.jnt_qposadr[joint] + 2] += .12
    mujoco.mj_forward(model, data)
    received = time.monotonic()
    k = rendered_camera_intrinsics(FOV, WIDTH, HEIGHT)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for index in range(len(VIEWS)):
            camera = f"ablation_{index}"
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            pose = camera_opencv_pose(model, data, model.camera(camera).id)
            path = output / f"view-{index}.npz"
            np.savez_compressed(path, rgb=rgb, depth=depth, camera_K=k, camera_pose=pose)
            Image.fromarray(rgb).save(path.with_suffix(".png"))
            yield SimpleNamespace(rgb=rgb, depth=depth, camera_K=k, camera_pose=pose,
                                  received_monotonic=received)


def evaluate(views):
    measured = [_measure_target(view, f"view:{index}") for index, view in enumerate(views)]
    plane = next((np.asarray(report["destination"]["plane"]) for report, _ in measured
                  if report["destination"]["status"] == "measured"), None)
    if plane is not None:
        # A coherent static scan can retain the independently measured plane
        # while a close camera crops the destination out of its image.
        measured = [
            _measure_target(views[index], f"view:{index}", reference_plane=plane)
            if sample is None and report["destination"]["status"] == "blocked" else (report, sample)
            for index, (report, sample) in enumerate(measured)
        ]
    support_views = [sample for _, sample in measured if sample is not None]
    result = {
        "views": [report for report, _ in measured],
        "associated_views": len(support_views),
        "baseline_m": max(float(np.linalg.norm(a.camera_pose[:3, 3] - b.camera_pose[:3, 3]))
                          for a in views for b in views),
    }
    if support_views:
        result["single_view_enclosure"] = _enclose_views(support_views[:1])
    if len(support_views) >= 2:
        result["multiview_enclosure"] = _enclose_views(support_views)
        result["multiview_5mm_margin_sensitivity"] = _enclose_views(support_views, depth_margin_m=.005)
    return result


def run(scene, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "offline rendered RGB-D ablations with reference camera poses",
              "scene": str(scene), "conditions": {}}
    for name, textured, suspended in (("plain_supported", False, False), ("textured_supported", True, False),
                                      ("plain_suspended", False, True)):
        folder = output / name
        folder.mkdir()
        model = build_model(scene, folder, textured=textured)
        report["conditions"][name] = evaluate(list(captures(model, folder, suspended=suspended)))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene.resolve(), args.output.resolve())
