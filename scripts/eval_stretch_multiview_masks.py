#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Synchronized head/wrist mask handoff during a native contact pick/place.

Control and task scoring remain privileged. The diagnostic tracker receives only
calibrated RGBD, an operator appearance seed, a declared shape prior and the
original shared sample interval. No object poses or scorer masks enter tracking.
This isolates visibility; it does not qualify the environment/contact planner.
Launch via emet jobs with CPU OSMesa; no VLM or local GPU is used.
"""

import argparse
import json
import time
from pathlib import Path

import eval_native_table_contact as contact
import mujoco
import numpy as np
from eval_moving_rgbd_geometry import capture
from eval_stretch_moving_masks import CameraContactRobot
from PIL import Image
from scipy.ndimage import binary_erosion

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.multiview_rgbd_tracking import MultiViewRgbdTracker, SynchronizedRgbdView
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.eval.episode_video import write_rgb_sequence_mp4
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.visualization.manip_video import overlay_manip_frame


class PairedContactRobot(CameraContactRobot):
    def __init__(self, model, object_body, **kwargs):
        self.wrist_renderer = None
        self._sensor_finished = 0.0
        super().__init__(model, object_body, **kwargs)
        head_settings = self.rgb_camera.initial_camera_settings
        self.renderer.close()
        self.renderer = mujoco.Renderer(model, height=head_settings.height, width=head_settings.width)
        rgb, depth = StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth
        self.wrist_pair = rgb.camera_name_in_mjcf, depth.camera_name_in_mjcf
        settings = rgb.initial_camera_settings
        for camera in (rgb, depth):
            cid = model.camera(camera.camera_name_in_mjcf).id
            model.cam_fovy[cid] = camera.initial_camera_settings.field_of_view_vertical_in_degrees
        self.wrist_renderer = mujoco.Renderer(model, height=settings.height, width=settings.width)
        self.observe()
        self.refresh()

    def refresh(self):
        # Native planning may take longer than the source lifetime. A live camera
        # keeps sampling then as well; take a genuinely new sample, never retime
        # the cached pixels. Debug controller state is read independently after it.
        super().refresh()
        if (
            self.renderer is not None
            and self.wrist_renderer is not None
            and time.monotonic() - self._sensor_finished > 0.25
        ):
            self.observe()
            super().refresh()

    def observe(self):
        if self.wrist_renderer is None:
            return  # Parent creates the head renderer before the paired sample.
        time.sleep(max(0.0, 0.25 - (time.monotonic() - self.last_capture)))
        probe = mujoco.MjData(self.model)
        start = time.monotonic()
        mujoco.mj_copyData(probe, self.model, self.data)
        end = time.monotonic()
        self.last_capture = start
        arrays = {
            "head": capture(
                self.model,
                probe,
                self.renderer,
                camera=self.rgb_camera.camera_name_in_mjcf,
                depth_camera=self.depth_camera.camera_name_in_mjcf,
                target_geom_id=self.target_geom_id,
            ),
            "wrist": capture(
                self.model,
                probe,
                self.wrist_renderer,
                camera=self.wrist_pair[0],
                depth_camera=self.wrist_pair[1],
                target_geom_id=self.target_geom_id,
            ),
        }
        index = len(self.rows)
        ticket = CaptureInterval(f"paired:{index}:{start}", start, end, time.monotonic(), "sim-shared-snapshot")
        views = {name: SynchronizedRgbdView(a[0], a[1], a[2], ticket) for name, a in arrays.items()}
        row = {
            "index": index,
            "capture": {
                "observation_id": ticket.observation_id,
                "earliest": start,
                "latest": end,
                "received_monotonic": ticket.received_monotonic,
            },
            "state": "candidate",
            "sim_time": float(probe.time),
            "scorer_visible_pixels": {name: int(a[3].sum()) for name, a in arrays.items()},
        }
        masks = {}
        try:
            if index == 0:
                seed = views["head"]
                c = seed.rgb.astype(float)
                mask = binary_erosion(
                    (c[..., 0] > 2 * c[..., 1])
                    & (c[..., 0] > 2 * c[..., 2])
                    & (c[..., 0] > 50)
                    & np.isfinite(seed.depth)
                    & (seed.depth > 0)
                )
                self.tracker = MultiViewRgbdTracker(
                    PrimitiveModel("cylinder", (0.02, 0.04)),
                    reference_view="head",
                    reference=seed,
                    mask=mask,
                    object_ref="operator-red-instance",
                    scene_epoch="contact-development",
                    allow_clipped_views=True,
                    component_policy="complete_prior_union",
                    appearance_model="neutral_specular",
                )
                result = self.tracker.latest
            else:
                if self.tracker is None:
                    raise ObservationUnavailable("moving_tracking_lost_reinitialize")
                result = self.tracker.update(views, scene_epoch="contact-development")
            masks = dict(result.masks)
            row.update(result.as_dict())
            row["scorer_center_error_m"] = float(
                np.linalg.norm(result.fit.poses[0, :3, 3] - probe.body(self.target_body).xpos)
            )
        except (ObservationUnavailable, ValueError) as error:
            row.update(state="unknown", reason=str(error))
            if self.tracker is not None:
                row["view_rejections"] = dict(self.tracker.last_view_rejections)
        self.rows.append(row)
        panels = []
        for name, a in arrays.items():
            rgb, depth, world, truth, _, _, k, pose = a
            mask = masks.get(name, np.zeros(depth.shape, bool))
            np.savez_compressed(
                self.output / f"{name}-{index:03}.npz",
                rgb=rgb,
                depth=depth,
                K=k,
                camera_pose=pose,
                predicted_mask=mask,
                scorer_mask=truth,
                metadata_json=json.dumps(row["capture"]),
            )
            colored = rgb.copy()
            colored[mask] = (0.45 * colored[mask] + 0.55 * np.array([40, 255, 70])).astype(np.uint8)
            if name == "head":
                colored = np.rot90(colored, -1)
                colored = colored[colored.shape[0] // 2 :]
            panel = np.zeros((400, 480, 3), np.uint8)
            h, w = colored.shape[:2]
            factor = min(400 / h, 480 / w)
            colored = np.array(Image.fromarray(colored).resize((int(w * factor), int(h * factor))))
            h, w = colored.shape[:2]
            panel[:h, :w] = colored
            panels.append(panel)
        panel = overlay_manip_frame(
            np.concatenate(panels, axis=1),
            overlay_style="border",
            title="Stretch native contact | head (presentation crop) / wrist | current masks",
            action="head/wrist tracking" if row["state"] == "candidate" else "STOP: global geometry unknown",
            goal="diagnose observed visibility through a full native contact transfer",
            detail=f"t={probe.time:.2f}s views={row.get('views_used', [])} " + row.get("reason", ""),
            flags="RGBD + prior only for tracker | privileged controller/scorer | no observed task-success claim",
        )
        self.frames.append(panel)
        if index % 10 == 0 or (index > 0 and row["state"] == "unknown" and self.rows[-2]["state"] == "candidate"):
            Image.fromarray(panel).save(self.output / f"frame-{index:03}.png")
        (self.output / "tracking.json").write_text(json.dumps(self.rows, indent=2) + "\n")
        self._sensor_finished = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    PairedContactRobot.output = args.output
    previous = contact.ContactRobot
    try:
        contact.ContactRobot = PairedContactRobot
        control = contact.run_case(mujoco.MjModel.from_xml_path(default_scene_xml_path), "object2")
    finally:
        contact.ContactRobot = previous
        for robot in PairedContactRobot.instances:
            robot.renderer.close()
            if robot.wrist_renderer is not None:
                robot.wrist_renderer.close()
    robot = PairedContactRobot.instances[-1]
    write_rgb_sequence_mp4(robot.frames, args.output / "head-wrist-contact.mp4", fps=4)
    (args.output / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    summary = {
        "scope": "co-sampled camera handoff during privileged contact control",
        "observed_task_success_assessed": False,
        "execution_qualified": False,
        "privileged_control_passed": control["passed"],
        "frames": len(robot.rows),
        "candidate_frames": sum(r["state"] == "candidate" for r in robot.rows),
        "rejection_codes": sorted({r["reason"] for r in robot.rows if "reason" in r}),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if control["passed"] and summary["candidate_frames"] == summary["frames"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
