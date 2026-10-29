#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Rendered moving-stage sensor diagnostic, not a robot task-success benchmark.

The evaluator moves a MuJoCo stage, renders RGBD, and separately scores masks and
geometry against simulator truth. Runtime tracking receives pixels, camera
calibration, original capture times and an explicit primitive prior only. Initial
mask selection uses an operator-declared red appearance, not semantic grounding.
No VLM or local GPU is needed. Launch via emet jobs with OSMesa when sharing GPU.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.moving_rgbd_tracking import MovingRgbdTracker
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.eval.episode_video import write_rgb_sequence_mp4
from emet.simulation.depth_registration import register_depth, rendered_camera_intrinsics
from emet.simulation.synchronized_capture import camera_opencv_pose
from emet.visualization.manip_video import overlay_manip_frame


def fixture(view):
    position = np.array([0.45 * view, -0.7, 0.5])
    z = position - np.array([0.0, 0.0, 0.08])
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    axes = " ".join(map(str, np.r_[x, y]))
    return mujoco.MjModel.from_xml_string(f'''<mujoco><visual><global offwidth="640" offheight="480"/>
      <quality shadowsize="1024"/></visual><worldbody>
      <light pos="0 -1 2" diffuse=".8 .8 .8"/>
      <geom type="plane" size="2 2 .01" rgba=".35 .38 .4 1"/>
      <camera name="sensor" pos="{position[0]} {position[1]} {position[2]}" xyaxes="{axes}" fovy="45"/>
      <body name="target" pos="-.1 0 .04"><joint name="x" type="slide" axis="1 0 0"/>
        <joint name="z" type="slide" axis="0 0 1"/>
        <geom name="target_geom" type="cylinder" size=".02 .04" rgba=".8 .2 .2 1"/></body>
      <body name="distractor" pos="2 0 .04"><joint name="distractor_x" type="slide" axis="1 0 0"/>
        <geom type="cylinder" size=".02 .04" rgba=".8 .2 .2 1"/></body>
      <body pos="2 -.05 .15"><joint name="occluder_x" type="slide" axis="1 0 0"/>
        <geom type="box" size=".2 .02 .15" rgba=".1 .1 .1 1"/></body>
      </worldbody></mujoco>''')


def capture(model, data, renderer, *, camera="sensor", depth_camera=None, target_geom_id=None):
    start = time.monotonic()
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    pose = camera_opencv_pose(model, data, camera_id)
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    depth_id = camera_id if depth_camera is None else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, depth_camera)
    if depth_camera is not None:
        renderer.update_scene(data, camera=depth_camera)
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    end = time.monotonic()
    k = rendered_camera_intrinsics(model.cam_fovy[camera_id], rgb.shape[1], rgb.shape[0])
    if depth_camera is not None:
        depth_k = rendered_camera_intrinsics(model.cam_fovy[depth_id], depth.shape[1], depth.shape[0])
        rgb_from_depth = np.linalg.inv(pose) @ camera_opencv_pose(model, data, depth_id)
        depth = register_depth(depth, depth_k, k, rgb_from_depth, rgb.shape[:2])
        renderer.update_scene(data, camera=camera)
    _, _, world = capture_arrays(SimpleNamespace(rgb=rgb, depth=depth, camera_K=k, camera_pose=pose))
    # Independent scorer channel; it is never supplied to the tracker.
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    target_geom_id = model.geom("target_geom").id if target_geom_id is None else target_geom_id
    truth = (segmentation[..., 0] == target_geom_id) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    return rgb, depth, world, truth, start, end, k, pose


def run_case(out, name, view, failure):
    out.mkdir(parents=True, exist_ok=True)
    model, rows, frames = fixture(view), [], []
    data = mujoco.MjData(model)
    prior = PrimitiveModel("cylinder", (0.02, 0.04))
    tracker = None
    previous_capture = 0.0
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for index in range(13):
            time.sleep(max(0.0, 0.22 - (time.monotonic() - previous_capture)))
            data.joint("x").qpos[0] = 0.02 * index
            data.joint("z").qpos[0] = 0.008 * min(index, 10)
            if failure == "distractor" and index >= 8:
                data.joint("distractor_x").qpos[0] = -2.0 + (-0.1 + 0.02 * index + 0.06)
                model.body("distractor").pos[2] = 0.04 + 0.008 * min(index, 10)
            if failure == "occlusion" and index >= 8:
                data.joint("occluder_x").qpos[0] = -2.0 + (-0.1 + 0.02 * index)
            rgb, depth, world, truth, start, received, k, pose = capture(model, data, renderer)
            previous_capture = start
            obs_id = f"{name}:{index}"
            row = {
                "index": index,
                "observation_id": obs_id,
                "capture_start": start,
                "received_monotonic": received,
                "state": "candidate",
            }
            mask = np.zeros(depth.shape, bool)
            try:
                if index == 0:
                    # Explicit operator appearance initialization, no GT pixel ids.
                    colors = rgb.astype(float)
                    mask = (
                        (colors[..., 0] > 2 * colors[..., 1])
                        & (colors[..., 0] > 2 * colors[..., 2])
                        & (colors[..., 0] > 50)
                    )
                    tracker = MovingRgbdTracker(
                        prior,
                        object_ref="operator-red-instance",
                        rgb=rgb,
                        depth=depth,
                        world=world,
                        mask=mask,
                        observation_id=obs_id,
                        received_monotonic=received,
                        scene_epoch=name,
                    )
                    result = tracker.latest
                else:
                    if tracker is None:
                        raise ObservationUnavailable("moving_tracking_lost_reinitialize")
                    result = tracker.update(
                        rgb, depth, world, observation_id=obs_id, received_monotonic=received, scene_epoch=name
                    )
                mask = result.mask
                center = result.fit.poses[0, :3, 3]
                row.update(result.as_dict())
                row["scorer_center_error_m"] = float(np.linalg.norm(center - data.body("target").xpos))
                row["scorer_mask_iou"] = float(np.sum(mask & truth) / max(1, np.sum(mask | truth)))
                row["scorer_axis_error_rad"] = float(np.arccos(np.clip(abs(result.fit.poses[0, 2, 2]), 0.0, 1.0)))
            except ObservationUnavailable as error:
                row.update(state="unknown", reason=str(error))
            rows.append(row)
            np.savez_compressed(
                out / f"capture-{index:03}.npz",
                rgb=rgb,
                depth=depth,
                world=world,
                K=k,
                camera_pose=pose,
                predicted_mask=mask,
                scorer_mask=truth,
                metadata_json=json.dumps(
                    {"observation_id": obs_id, "capture_start": start, "received_monotonic": received}
                ),
            )
            colored = rgb.copy()
            colored[mask] = (0.45 * colored[mask] + 0.55 * np.array([40, 255, 70])).astype(np.uint8)
            panel = np.concatenate([rgb, colored], axis=1)
            label = row.get("reason", f"center error {row.get('scorer_center_error_m', 0) * 1000:.2f} mm")
            panel = overlay_manip_frame(
                panel,
                overlay_style="border",
                title=f"Moving RGBD | {name} | raw / candidate mask",
                action="track moving instance" if row["state"] == "candidate" else "STOP: unknown identity",
                goal="diagnose translation, lift, distractors and loss",
                detail=f"frame {index}: {label}",
                flags="RGBD + declared model prior | truth used by scorer only | NOT robot task success",
            )
            frames.append(panel)
            if index in (0, 7, 8, 12):
                Image.fromarray(panel).save(out / f"frame-{index:03}.png")
            (out / "result.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_rgb_sequence_mp4(frames, out / "moving-mask.mp4", fps=5)
    passed = all((r["state"] == "candidate") == (not failure or i < 8) for i, r in enumerate(rows))
    passed &= all(r.get("scorer_center_error_m", 0) < 0.005 and r.get("scorer_mask_iou", 1) > 0.9 for r in rows)
    return {
        "case": name,
        "passed": bool(passed),
        "frames": len(rows),
        "candidate_frames": sum(r["state"] == "candidate" for r in rows),
        "expected_candidate_frames": 8 if failure else len(rows),
        "max_center_error_m": max(r.get("scorer_center_error_m", 0) for r in rows),
        "rejection_codes": sorted({r["reason"] for r in rows if "reason" in r}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        run_case(args.output / name, name, view, failure)
        for name, view, failure in [
            ("front-lift", 1, ""),
            ("opposite-lift", -1, ""),
            ("distractor-stop", 1, "distractor"),
            ("occlusion-stop", 1, "occlusion"),
        ]
    ]
    summary = {
        "schema_version": 1,
        "scope": "rendered stage sensor diagnostics, not observed robot task success",
        "initialization": "operator-declared red appearance; no VLM or GT segmentation",
        "execution_qualified": False,
        "cases": cases,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if all(c["passed"] for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
