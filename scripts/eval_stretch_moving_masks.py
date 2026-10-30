#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Observe a privileged native Stretch contact transfer through calibrated RGBD.

Task control/scoring stays privileged. A separate tracker gets measured camera
pixels, capture times and the declared cylinder prior, never object poses/masks.
Operator red-colour initialization is not agent semantic discovery. Tracking loss
is latched and does not stop this diagnostic's privileged controller. Launch with
emet jobs and a software renderer. No VLM or local GPU is used.
"""

import argparse
import json
import time
from pathlib import Path

import eval_native_table_contact as contact
import mujoco
import numpy as np
from eval_moving_rgbd_geometry import capture
from PIL import Image
from scipy.ndimage import binary_erosion

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.moving_rgbd_tracking import MovingRgbdTracker
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.eval.episode_video import write_rgb_sequence_mp4
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.visualization.manip_video import overlay_manip_frame


class CameraContactRobot(contact.ContactRobot):
    output = None
    instances = []
    view = "head"

    def __init__(self, model, object_body, **kwargs):
        self.renderer, self.tracker, self.rows, self.frames = None, None, [], []
        self.last_capture, self.next_sim_capture = 0.0, 0.0
        super().__init__(model, object_body, **kwargs)
        # Standard motor commands orient the camera before the diagnostic begins.
        self.commands.append({"sim_time": float(self.data.time), "head": [-1.57, -0.7]})
        self.data.ctrl[model.actuator("head_pan").id] = -1.57
        self.data.ctrl[model.actuator("head_tilt").id] = -0.7
        self.step(1.0)
        self.rgb_camera, self.depth_camera = (
            (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth)
            if self.view == "head"
            else (StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth)
        )
        settings = self.rgb_camera.initial_camera_settings
        self.width, self.height = (848, 480) if self.view == "head" else (settings.width, settings.height)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
        for camera in (self.rgb_camera, self.depth_camera):
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.camera_name_in_mjcf)
            model.cam_fovy[camera_id] = camera.initial_camera_settings.field_of_view_vertical_in_degrees
        self.target_geom_id = int(model.body(object_body).geomadr[0])
        self.target_body = object_body  # Independent scorer only.
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.instances.append(self)
        self.observe()

    def step(self, duration):
        for _ in range(max(1, round(duration / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
            if self.renderer is not None and self.data.time >= self.next_sim_capture:
                self.next_sim_capture = self.data.time + 0.25
                self.observe()

    def observe(self):
        # Keep simulated motion from outrunning the declared wall-clock motion
        # envelope; timestamps themselves remain the real capture times.
        time.sleep(max(0.0, 0.25 - (time.monotonic() - self.last_capture)))
        rgb, depth, world, truth, start, received, k, pose = capture(
            self.model,
            self.data,
            self.renderer,
            camera=self.rgb_camera.camera_name_in_mjcf,
            target_geom_id=self.target_geom_id,
            depth_camera=self.depth_camera.camera_name_in_mjcf,
        )
        self.last_capture = start
        index, mask = len(self.rows), np.zeros(depth.shape, bool)
        obs_id = f"{self.view}:{index}:{start}"
        row = {
            "index": index,
            "capture_start": start,
            "received_monotonic": received,
            "sim_time": float(self.data.time),
            "state": "candidate",
            "scorer_visible_pixels": int(truth.sum()),
        }
        try:
            if index == 0:
                c = rgb.astype(float)
                mask = (c[..., 0] > 2 * c[..., 1]) & (c[..., 0] > 2 * c[..., 2]) & (c[..., 0] > 50)
                # Operator initialization uses only confidently interior pixels;
                # silhouette RGB can describe a blend with unrelated depth.
                mask = binary_erosion(mask & np.isfinite(depth) & (depth > 0))
                self.tracker = MovingRgbdTracker(
                    PrimitiveModel("cylinder", (0.02, 0.04)),
                    object_ref="operator-red-instance",
                    rgb=rgb,
                    depth=depth,
                    world=world,
                    mask=mask,
                    observation_id=obs_id,
                    received_monotonic=received,
                    scene_epoch="contact-development",
                )
                result = self.tracker.latest
            else:
                if self.tracker is None:
                    raise ObservationUnavailable("moving_tracking_lost_reinitialize")
                result = self.tracker.update(
                    rgb,
                    depth,
                    world,
                    observation_id=obs_id,
                    received_monotonic=received,
                    scene_epoch="contact-development",
                )
            mask = result.mask
            row.update(result.as_dict())
            row["scorer_center_error_m"] = float(
                np.linalg.norm(result.fit.poses[0, :3, 3] - self.data.body(self.target_body).xpos)
            )
            row["scorer_mask_iou"] = float(np.sum(mask & truth) / max(1, np.sum(mask | truth)))
        except (ObservationUnavailable, ValueError) as error:
            row.update(state="unknown", reason=str(error))
        self.rows.append(row)
        np.savez_compressed(
            self.output / f"capture-{index:03}.npz",
            rgb=rgb,
            depth=depth,
            world=world,
            K=k,
            camera_pose=pose,
            predicted_mask=mask,
            scorer_mask=truth,
            metadata_json=json.dumps({key: row[key] for key in ("capture_start", "received_monotonic", "sim_time")}),
        )
        colored = rgb.copy()
        colored[mask] = (0.45 * colored[mask] + 0.55 * np.array([40, 255, 70])).astype(np.uint8)
        presentation = [np.rot90(p, -1) for p in (rgb, colored)] if self.view == "head" else [rgb, colored]
        panel = overlay_manip_frame(
            np.concatenate(presentation, axis=1),
            overlay_style="border",
            title=f"Stretch native contact | raw {self.view} RGB / observed mask candidate",
            action="track moving payload" if row["state"] == "candidate" else "STOP: observed identity unknown",
            goal="separate privileged control from observed sensor qualification",
            detail=f"sim t={self.data.time:.2f}: " + row.get("reason", "candidate; no material yaw"),
            flags="controller: privileged | tracker: RGBD + model prior | no GT mask input",
        )
        self.frames.append(panel)
        if index % 10 == 0 or (index > 0 and row["state"] == "unknown" and self.rows[-2]["state"] == "candidate"):
            Image.fromarray(panel).save(self.output / f"frame-{index:03}.png")
        (self.output / "tracking.json").write_text(json.dumps(self.rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera",
        choices=("head", "wrist"),
        default="head",
        help="Registered RGB/depth optical pair for the independent tracker diagnostic.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    CameraContactRobot.output = args.output
    CameraContactRobot.view = args.camera
    previous = contact.ContactRobot
    try:
        contact.ContactRobot = CameraContactRobot
        result = contact.run_case(mujoco.MjModel.from_xml_path(default_scene_xml_path), "object2")
    finally:
        contact.ContactRobot = previous
        for robot in CameraContactRobot.instances:
            robot.renderer.close()
    robot = CameraContactRobot.instances[-1]
    write_rgb_sequence_mp4(robot.frames, args.output / "stretch-moving-mask.mp4", fps=4)
    (args.output / "control.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "scope": "observed sensor diagnostics during a privileged native contact control",
        "observed_task_success_assessed": False,
        "execution_qualified": False,
        "camera": args.camera,
        "camera_geometry": "separate calibrated RGB/depth optical frames; measured depth registration, holes unfilled",
        "privileged_control_passed": result["passed"],
        "frames": len(robot.rows),
        "candidate_frames": sum(r["state"] == "candidate" for r in robot.rows),
        "rejection_codes": sorted({r["reason"] for r in robot.rows if "reason" in r}),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
