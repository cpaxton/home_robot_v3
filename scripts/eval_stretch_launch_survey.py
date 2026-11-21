#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Check survey registration and footprint admission using rendered depth."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from emet.controller.manipulation.encoder_capture import read_local_sim_motor_capture
from emet.controller.manipulation.stretch_encoder_kinematics import stretch_base_from_head_camera
from emet.mapping.voxel.launch_floor_survey import register_launch_floor_survey
from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap
from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace
from emet.motion.navigation_sweep import validate_navigation_sweep
from emet.robots.stretch import StretchBackend
from emet.simulation.depth_registration import rendered_camera_intrinsics
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync
from emet.utils.image import pinhole_camera_from_intrinsics_and_depth

MOUNT = np.array([.4, 0., 1.8])
ANCHOR_COLOURS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0))
ANCHOR_POSITIONS = ((.56, -.17), (.56, .17), (.97, -.17), (.97, .17))


def scene_variants(scene, output):
    tree = ET.parse(scene)
    root = tree.getroot()
    robot = root.find("include")
    if robot is None:
        raise ValueError("expected a fixture including a separate calibrated robot model")
    robot.set("file", str((scene.parent / robot.attrib["file"]).resolve()))
    world = root.find("worldbody")
    for (x, y), rgb in zip(ANCHOR_POSITIONS, ANCHOR_COLOURS, strict=True):
        ET.SubElement(world, "geom", type="box", pos=f"{x} {y} .003", size=".041 .041 .003",
                      rgba=" ".join(map(str, (*rgb, 1))), contype="0", conaffinity="0")
    with_robot = output / "onboard.xml"
    tree.write(with_robot)
    root.remove(robot)
    ET.SubElement(world, "camera", name="survey", pos=" ".join(map(str, MOUNT)), fovy="58")
    before_robot = output / "survey.xml"
    tree.write(before_robot)
    return with_robot, before_robot, Path(robot.attrib["file"])


def measured_points(rgb, depth, k, pose):
    camera = pinhole_camera_from_intrinsics_and_depth(k, depth)
    points = camera.depth_to_xyz(depth, data_type=np.float64)
    points = points @ pose[:3, :3].T + pose[:3, 3]
    valid = np.isfinite(depth) & (depth > .1) & (depth < 4.)
    marker_height = np.abs(points[:, :, 2]) < .025
    colours = rgb.astype(float)
    anchors = []
    for colour in ANCHOR_COLOURS:
        active = np.asarray(colour, dtype=bool)
        foreground = colours[:, :, active]
        background = colours[:, :, ~active]
        match = valid & marker_height & (foreground.min(axis=-1) - background.max(axis=-1) > 45)
        match &= foreground.max(axis=-1) - foreground.min(axis=-1) < 35
        match &= background.max(axis=-1) - background.min(axis=-1) < 15
        if np.count_nonzero(match) < 5:
            raise RuntimeError(f"insufficient measured points for anchor {colour}")
        anchors.append(np.median(points[match], axis=0))
    return points[valid], np.stack(anchors)


def survey_capture(scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        renderer.update_scene(data, camera="survey")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    k = rendered_camera_intrinsics(58., 480, 360)
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = MOUNT
    return rgb, depth, k, pose, time.monotonic()


def onboard_capture(scene, robot):
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.ctrl[:] = model.key_ctrl[0]
    data.ctrl[model.actuator("head_tilt").id] = -.88
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    mujoco.mj_step(model, data, nstep=round(3 / model.opt.timestep))
    frames = []
    proxy = SimpleNamespace(mjmodel=model, mjdata=data, data_proxies=SimpleNamespace(set_cameras=frames.append))
    camera = StretchCameras.cam_d435i_rgb
    manager = MujocoServerCameraManagerSync(15, [camera, StretchCameras.cam_d435i_depth], proxy)
    try:
        manager._pull_synchronized_camera_data()
    finally:
        manager.close()
    frame, = frames
    timing = frame.image_timing_for(camera)
    timing.pop("robot_sample", None)
    received = time.monotonic()
    sample = SimpleNamespace(
        is_simulation=True, received_monotonic=received, image_timing={"head_cam_left": timing}
    )
    pose = stretch_base_from_head_camera(read_local_sim_motor_capture(sample), robot)
    rgb = np.rot90(frame.cam_d435i_rgb, k=-1)
    depth = np.rot90(frame.cam_d435i_depth, k=-1)
    raw = frame.cam_d435i_K
    k = np.array([[raw[1, 1], 0., depth.shape[1] - 1 - raw[1, 2]], [0., raw[0, 0], raw[0, 2]], [0., 0., 1.]])
    return rgb, depth, k, pose, timing["timestamp_ns"] / 1e9


def assess(scene, output):
    output.mkdir()
    onboard_path, survey_path, robot_path = scene_variants(scene, output)
    robot = mujoco.MjModel.from_xml_path(str(robot_path))
    survey_rgb, survey_depth, survey_k, survey_pose, surveyed_at = survey_capture(survey_path)
    onboard_rgb, onboard_depth, onboard_k, onboard_pose, observed_at = onboard_capture(onboard_path, robot)
    survey_points, survey_anchors = measured_points(survey_rgb, survey_depth, survey_k, survey_pose)
    onboard_points, onboard_anchors = measured_points(onboard_rgb, onboard_depth, onboard_k, onboard_pose)
    np.savez_compressed(output / "captures.npz", survey_depth=survey_depth, survey_rgb=survey_rgb,
                        onboard_depth=onboard_depth, onboard_rgb=onboard_rgb, survey_anchors=survey_anchors,
                        onboard_anchors=onboard_anchors, survey_K=survey_k, onboard_K=onboard_k,
                        survey_pose=survey_pose, onboard_pose=onboard_pose)
    voxel = SparseVoxelMap(
        grid_size=[72, 72], grid_resolution=.05, smooth_kernel_size=0, min_depth=.1,
        log=str(output), device="cpu", image_shape=None, use_instance_memory=False,
        add_local_radius_points=False, parameters={"floor_evidence": {"enabled": True, "reference_z_m": 0.}},
    )
    voxel.voxel_pcd.add(torch.from_numpy(onboard_points).float(), features=None,
                        rgb=torch.zeros((len(onboard_points), 3)), obs_count=1)
    footprint = StretchBackend().get_spec().footprint
    space = SparseVoxelMapNavigationSpace(voxel, obstacle_map_mode="physical", footprint=footprint)
    start = np.zeros(3)
    goal = np.array([1.5, 0., 0.])
    result = {"missing_survey_start_valid": space.is_valid(start)}
    timing = {
        "observation_scope_id": voxel.observation_scope_id, "survey_captured_at": surveyed_at,
        "anchors_captured_at": observed_at, "now": time.monotonic(), "start_xy": start[:2],
        "radius_m": .55, "max_age_s": 60., "max_anchor_error_m": .025,
    }
    shifted = onboard_anchors.copy()
    shifted[0, 0] += .12
    try:
        register_launch_floor_survey(survey_points, survey_anchors, shifted, **timing)
    except ValueError as error:
        result["misaligned_registration"] = str(error)
    result["after_misaligned_survey_start_valid"] = space.is_valid(start)
    survey = register_launch_floor_survey(survey_points, survey_anchors, onboard_anchors, **timing)
    voxel.install_launch_floor_survey(survey)
    result["registered_start_valid"] = space.is_valid(start)
    result["start_footprint"] = space.last_validity
    result["route_admitted"], result["route_reason"] = validate_navigation_sweep(space, start, [goal])
    result["anchor_max_residual_m"] = survey.max_anchor_residual_m
    result["local_survey_points"] = len(survey.points)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-scene", type=Path, required=True)
    parser.add_argument("--occupied-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    result = {}
    for case, scene in (("clear", args.clear_scene), ("occupied", args.occupied_scene)):
        result[case] = assess(scene, args.output / case)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        assert not result[case]["missing_survey_start_valid"], result[case]
        assert not result[case]["after_misaligned_survey_start_valid"], result[case]
        assert result[case]["registered_start_valid"], result[case]
    assert result["clear"]["route_admitted"], result["clear"]
    assert result["occupied"]["route_reason"] == "occupied_footprint", result["occupied"]


if __name__ == "__main__":
    main()
