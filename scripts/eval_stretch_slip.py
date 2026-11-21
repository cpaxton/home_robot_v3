#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Exercise a wheel-slip stop against a calibrated overhead RGB-D view."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.controller.manipulation.capture_clock import local_monotonic_clock_id
from emet.controller.manipulation.encoder_capture import read_local_sim_wheel_capture
from emet.core.navigation_result import NavigationPoseUncertainty
from emet.mapping.wheel_motion_guard import WheelMotionGuard
from emet.mapping.wheel_odometry import DifferentialDriveOdometry
from emet.simulation.depth_registration import rendered_camera_intrinsics
from emet.simulation.molmospaces_mobile_autoplace import read_body_se2_xyt
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators
from emet.simulation.stretch_mujoco.config import robot_settings
from emet.simulation.stretch_mujoco.utils import diff_drive_inv_kinematics
from emet.simulation.synchronized_capture import capture_stretch_wheels

CAMERA_XYZ = np.array([.25, 0., 4.])
HEIGHT, WIDTH, FOVY = 240, 320, 22.
MARKERS = ((.145, -.34, .265), (-.145, .34, .265))
COLOURS = ((1, 0, 1), (0, 1, 1))
MARKER_HALF_SIZE = (.043, .035, .004)
TARGET_M = .32
PERIOD_S = .15


def build_scene(scene, output, *, slippery):
    tree = ET.parse(scene)
    include = tree.find("./include")
    robot_path = (scene.parent / include.attrib["file"]).resolve()
    robot = ET.parse(robot_path)
    base = robot.find(".//body[@name='base_link']")
    asset = robot.find("./asset")
    for index, (position, colour) in enumerate(zip(MARKERS, COLOURS, strict=True)):
        name = f"slip_marker_{index}"
        ET.SubElement(asset, "material", name=name, rgba=" ".join(map(str, (*colour, 1))),
                      emission=".7", specular="0")
        ET.SubElement(base, "geom", name=name, type="box", pos=" ".join(map(str, position)),
                      size=" ".join(map(str, MARKER_HALF_SIZE)), material=name, group="2",
                      contype="0", conaffinity="0", mass="0")
    instrumented = output / "robot.xml"
    robot.write(instrumented)
    include.set("file", str(instrumented.resolve()))
    world = tree.find("./worldbody")
    ET.SubElement(world, "camera", name="slip_rgbd", pos=" ".join(map(str, CAMERA_XYZ)), fovy=str(FOVY))
    if slippery:
        floor = world.find("geom[@name='floor']")
        floor.set("priority", "10")
        floor.set("friction", ".0008 .00005 .00001")
    path = output / "scene.xml"
    tree.write(path)
    return path


def reference_from_depth(rgb, depth):
    k = rendered_camera_intrinsics(FOVY, WIDTH, HEIGHT)
    xyz = []
    for colour in COLOURS:
        selected = np.asarray(colour, dtype=bool)
        bright = rgb[:, :, selected].astype(np.int16)
        dark = rgb[:, :, ~selected]
        mask = (bright.min(axis=2) - dark.max(axis=2) > 70) & (np.ptp(bright, axis=2) < 50)
        mask &= np.isfinite(depth) & (np.abs((CAMERA_XYZ[2] - depth) - MARKERS[0][2]) < .05)
        rows, cols = np.nonzero(mask)
        if len(rows) < 15:
            return None
        z = depth[rows, cols]
        face_area = 4 * MARKER_HALF_SIZE[0] * MARKER_HALF_SIZE[1]
        expected_pixels = face_area * k[0, 0] * k[1, 1] / np.median(z) ** 2
        if len(rows) < .75 * expected_pixels:
            return None
        xyz.append(np.median(np.column_stack(((cols - k[0, 2]) * z / k[0, 0],
                                              -(rows - k[1, 2]) * z / k[1, 1])), axis=0) + CAMERA_XYZ[:2])
    baseline = np.subtract(*xyz)
    calibrated_baseline = np.subtract(MARKERS[0][:2], MARKERS[1][:2])
    if abs(np.linalg.norm(baseline) - np.linalg.norm(calibrated_baseline)) > .018:
        return None
    yaw = np.arctan2(baseline[1], baseline[0]) - np.arctan2(calibrated_baseline[1], calibrated_baseline[0])
    return np.r_[np.mean(xyz, axis=0), np.arctan2(np.sin(yaw), np.cos(yaw))]


def capture_wheels(model, data, stamp, end):
    packet = capture_stretch_wheels(model, data, {"clock_id": local_monotonic_clock_id(), "interval_ns": [stamp, end]})
    timing = {
        "timestamp_ns": stamp, "available": True, "source": "renderer_synchronized_snapshot",
        "clock_domain": "mujoco_render_monotonic", "clock_id": packet["clock_id"],
        "capture_interval_ns": [stamp, end], "depth_capture_interval_ns": [stamp, end], "wheel_sample": packet,
    }
    return read_local_sim_wheel_capture(SimpleNamespace(
        is_simulation=True, received_monotonic=time.monotonic(), image_timing={"head_cam_left": timing}
    ))


def run(scene, output, mode):
    output.mkdir()
    model = mujoco.MjModel.from_xml_path(str(build_scene(scene, output, slippery=mode == "slip")))
    data = mujoco.MjData(model)
    data.ctrl[:] = model.key_ctrl[0]
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    mujoco.mj_step(model, data, nstep=round(3 / model.opt.timestep))
    wheels = [model.actuator(name).id for name in ("left_wheel_vel", "right_wheel_vel")]
    radius = robot_settings["wheel_diameter"] / 2
    odometry = DifferentialDriveOdometry(wheel_radius_m=radius, track_width_m=robot_settings["wheel_separation"])
    guard = WheelMotionGuard(NavigationPoseUncertainty(.04, .025), max_age_s=.5)
    rows = []
    truth_qpos = []
    decision = "test_motion_deadline"
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        # Prime software rendering while the robot is still stationary.
        renderer.update_scene(data, camera="slip_rgbd")
        renderer.render()
        renderer.enable_depth_rendering()
        renderer.render()
        for index in range(65):
            # Physics is paused during the paired capture.
            stamp, end = time.monotonic_ns(), time.monotonic_ns()
            measurement = capture_wheels(model, data, stamp, end)
            estimate = odometry.update(measurement)
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera="slip_rgbd")
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            if mode == "occluded" and estimate[0] >= .065:
                depth[:] = 0.
            reference = reference_from_depth(rgb, depth)
            check = guard.update(estimate, reference, wheel_timestamp_ns=stamp, reference_timestamp_ns=stamp,
                                 now_ns=time.monotonic_ns())
            rows.append({
                "sim_time": data.time, "encoder_xyt": estimate.tolist(),
                "reference_xyt": None if reference is None else reference.tolist(), **asdict(check),
                "acquisition_wall_s": (time.monotonic_ns() - stamp) / 1e9,
            })
            truth_qpos.append(data.qpos.copy())
            if not check.ok or estimate[0] >= TARGET_M:
                decision = check.status if not check.ok else "reached"
            if index == 0 or decision != "test_motion_deadline":
                np.savez_compressed(output / f"capture-{index}.npz", rgb=rgb, depth=depth)
            if decision != "test_motion_deadline":
                break
            data.ctrl[wheels] = diff_drive_inv_kinematics(.24, 0.)
            mujoco.mj_step(model, data, nstep=round(PERIOD_S / model.opt.timestep))
        data.ctrl[wheels] = 0.
        mujoco.mj_step(model, data, nstep=round(1 / model.opt.timestep))
    stamp, end = time.monotonic_ns(), time.monotonic_ns()
    at_rest = capture_wheels(model, data, stamp, end)
    resting_position = odometry.update(at_rest)
    mujoco.mj_step(model, data, nstep=round(PERIOD_S / model.opt.timestep))
    stamp, end = time.monotonic_ns(), time.monotonic_ns()
    after_rest = odometry.update(capture_wheels(model, data, stamp, end))
    truth_qpos.append(data.qpos.copy())
    # Held-out chassis measurements are available only to this scorer.
    scorer = mujoco.MjData(model)
    truth = []
    for qpos in truth_qpos:
        scorer.qpos[:] = qpos
        mujoco.mj_kinematics(model, scorer)
        truth.append(read_body_se2_xyt(scorer).tolist())
    (output / "trace.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "scoring-only.json").write_text(json.dumps(truth, indent=2) + "\n")
    final_reference, initial_reference = rows[-1]["reference_xyt"], rows[0]["reference_xyt"]
    return {
        "decision": decision,
        "samples": len(rows),
        "encoder_travel_at_decision_m": float(np.linalg.norm(estimate[:2])),
        "independent_travel_at_decision_m": (
            float(np.linalg.norm(np.subtract(final_reference[:2], initial_reference[:2])))
            if final_reference is not None and initial_reference is not None else None
        ),
        "chassis_travel_at_decision_m": float(np.linalg.norm(np.subtract(truth[-2][:2], truth[0][:2]))),
        "chassis_travel_after_stop_m": float(np.linalg.norm(np.subtract(truth[-1][:2], truth[-2][:2]))),
        "wheel_motion_after_settle_m": float(np.linalg.norm(after_rest[:2] - resting_position[:2])),
        "disagreement_at_decision_m": rows[-1]["position_disagreement_m"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    result = {}
    for mode in ("normal", "slip", "occluded"):
        result[mode] = run(args.scene, args.output / mode, mode)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["normal"]["decision"] == "reached", result
    assert result["slip"]["decision"] == "wheel_motion_disagreement", result
    assert result["occluded"]["decision"] == "independent_pose_unavailable", result
    assert abs(result["normal"]["chassis_travel_at_decision_m"] - TARGET_M) < .015, result
    assert result["slip"]["chassis_travel_at_decision_m"] < .01, result
    assert result["occluded"]["chassis_travel_at_decision_m"] > .05, result
    assert all(case["chassis_travel_after_stop_m"] < .01 for case in result.values()), result
    assert all(case["wheel_motion_after_settle_m"] < .005 for case in result.values()), result


if __name__ == "__main__":
    main()
