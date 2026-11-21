#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Stretch floor controls using registered RGB-D and encoder readouts."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.controller.manipulation.encoder_capture import read_local_sim_motor_capture, read_local_sim_wheel_capture
from emet.controller.manipulation.stretch_encoder_kinematics import stretch_base_from_head_camera
from emet.mapping.voxel.floor_evidence import floor_evidence
from emet.mapping.wheel_odometry import DifferentialDriveOdometry
from emet.simulation.molmospaces_mobile_autoplace import read_body_se2_xyt
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators
from emet.simulation.stretch_mujoco.config import robot_settings
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync
from emet.simulation.stretch_mujoco.utils import diff_drive_inv_kinematics
from emet.utils.geometry import xyt2sophus


def motion_schedule(moving):
    yield (-.65, 0., 0., 2.)
    if moving:
        for count, speed, yaw_rate in ((24, .27, 0.), (24, .17, .40), (4, 0., 0.)):
            for _ in range(count):
                yield (-.65, speed, yaw_rate, 1 / 6)
    yield (-1., 0., 0., 2.)


def score_trajectory(views, model, qpos_samples, output):
    """Score the estimate after mapping, in the frame of the first actual pose."""
    scorer = mujoco.MjData(model)
    world_truth = []
    for qpos in qpos_samples:
        scorer.qpos[:] = qpos
        mujoco.mj_kinematics(model, scorer)
        world_truth.append(read_body_se2_xyt(scorer))
    local_from_world = xyt2sophus(world_truth[0]).inverse()
    truth = np.stack([(local_from_world * xyt2sophus(pose)).matrix() for pose in world_truth])
    estimated = np.stack([view["local_base_xyt"] for view in views])
    xy_error = np.linalg.norm(estimated[:, :2] - truth[:, :2, 3], axis=1)
    yaw = np.arctan2(truth[:, 1, 0], truth[:, 0, 0])
    yaw_error = np.abs(np.rad2deg(np.arctan2(np.sin(estimated[:, 2] - yaw), np.cos(estimated[:, 2] - yaw))))
    np.savez_compressed(output / "scoring-only.npz", truth_local_from_base=truth, estimated_xyt=estimated,
                        xy_error_m=xy_error, yaw_error_deg=yaw_error)
    return {
        "scoring_source": "held-out simulator chassis pose",
        "distance_from_start_m": float(np.linalg.norm(truth[-1, :2, 3])),
        "final_yaw_deg": float(np.rad2deg(yaw[-1])),
        "max_xy_error_m": float(xy_error.max()),
        "max_yaw_error_deg": float(yaw_error.max()),
    }


def capture_floor(scene, robot_model, output, *, moving=False):
    output.mkdir()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.ctrl[:] = model.key_ctrl[0]
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    mujoco.mj_step(model, data, nstep=int(3.0 / model.opt.timestep))
    wheel_actuators = [model.actuator(name).id for name in ("left_wheel_vel", "right_wheel_vel")]
    wheel_radius = robot_settings["wheel_diameter"] / 2
    odometry = DifferentialDriveOdometry(wheel_radius_m=wheel_radius, track_width_m=robot_settings["wheel_separation"])
    origin_wheels = None
    camera = StretchCameras.cam_d435i_rgb
    frames = []
    server = SimpleNamespace(
        mjmodel=model, mjdata=data, data_proxies=SimpleNamespace(set_cameras=frames.append)
    )
    manager = MujocoServerCameraManagerSync(15, [camera, StretchCameras.cam_d435i_depth], server)
    points = []
    views = []
    truth_qpos = []
    try:
        for index, (tilt, speed, yaw_rate, duration) in enumerate(motion_schedule(moving)):
            data.ctrl[model.actuator("head_pan").id] = 0.0
            data.ctrl[model.actuator("head_tilt").id] = tilt
            data.ctrl[wheel_actuators] = diff_drive_inv_kinematics(speed, yaw_rate)
            mujoco.mj_step(model, data, nstep=round(duration / model.opt.timestep))
            manager._pull_synchronized_camera_data()
            frame = frames.pop()
            timing = frame.image_timing_for(camera)
            # The estimator receives only clock metadata and the motor packet.
            timing.pop("robot_sample", None)
            sample = SimpleNamespace(
                is_simulation=True, image_timing={"head_cam_left": timing}, received_monotonic=time.monotonic()
            )
            motors = read_local_sim_motor_capture(sample)
            wheels = read_local_sim_wheel_capture(sample)
            wheel_positions = np.array([wheels.left_rad, wheels.right_rad])
            if origin_wheels is None:
                origin_wheels = wheel_positions
            travel = (wheel_positions - origin_wheels) * wheel_radius
            if not moving and np.max(np.abs(travel)) > .002:
                raise RuntimeError("wheel readings exceeded the stationary trial bound")
            xyt = odometry.update(wheels) if moving else np.zeros(3)
            local_from_base = xyt2sophus(xyt).matrix()
            base_from_camera = stretch_base_from_head_camera(motors, robot_model)
            local_from_camera = local_from_base @ base_from_camera
            rgb = np.rot90(frame.cam_d435i_rgb, k=-1)
            depth = np.rot90(frame.cam_d435i_depth, k=-1)
            raw_k = frame.cam_d435i_K
            k = np.array([
                [raw_k[1, 1], 0., depth.shape[1] - 1 - raw_k[1, 2]],
                [0., raw_k[0, 0], raw_k[0, 2]],
                [0., 0., 1.],
            ])
            rows, cols = np.nonzero(np.isfinite(depth) & (depth > .1) & (depth < 3.))
            z = depth[rows, cols]
            xyz = np.column_stack(((cols-k[0, 2])*z/k[0, 0], (rows-k[1, 2])*z/k[1, 1], z))
            points.append(xyz @ local_from_camera[:3, :3].T + local_from_camera[:3, 3])
            measured_tilt = motors.as_dict()["head_tilt"]
            if abs(measured_tilt - tilt) > .03:
                raise RuntimeError("head encoder did not reach the requested tilt")
            np.savez_compressed(
                output / f"frame-{index}.npz", rgb=rgb, depth=depth, camera_K=k, base_from_camera=base_from_camera,
                motor_positions=motors.positions, wheel_travel_m=travel, local_from_camera=local_from_camera,
            )
            views.append({
                "head_tilt_rad": measured_tilt, "wheel_travel_m": travel.tolist(),
                "points": len(z), "capture": timing, "local_base_xyt": xyt.tolist(),
            })
            if moving:
                # Physics is paused during capture. Keep this state for the scorer.
                truth_qpos.append(data.qpos.copy())
    finally:
        manager.close()
    resolution = .04
    origin = np.array([60, 60])
    layers = floor_evidence(np.concatenate(points), (120, 120), resolution, origin, reference_z_m=0.0)
    # The route is supplied by the exercise. Both cases receive the same
    # corridor, camera commands, calibration and evidence classification.
    x, y = (np.indices(layers.raised.shape) - origin[:, None, None]) * resolution
    corridor = (x >= .3) & (x <= 1.5) & (abs(y) <= .2)
    np.savez_compressed(output / "floor.npz", observed=layers.observed, raised=layers.raised, drops=layers.drops)
    result = {
        "views": views,
        "corridor_floor_cells": int((layers.observed & corridor).sum()),
        "corridor_raised_cells": int((layers.raised & corridor).sum()),
        "corridor_drop_cells": int((layers.drops & corridor).sum()),
    }
    if moving:
        result["odometry"] = score_trajectory(views, model, truth_qpos, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-only-model", type=Path, required=True)
    parser.add_argument("--clear-scene", type=Path, required=True)
    parser.add_argument("--occupied-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moving", action="store_true", help="Integrate a short drive and turn from wheel readouts.")
    args = parser.parse_args()
    robot_model = mujoco.MjModel.from_xml_path(str(args.robot_only_model))
    args.output.mkdir(parents=True, exist_ok=False)
    mode = "moving" if args.moving else "stationary"
    result = {"scope": f"{mode} flat-floor calibration, onboard RGB-D and encoder equivalents"}
    for name, scene in (("clear", args.clear_scene), ("occupied", args.occupied_scene)):
        result[name] = capture_floor(scene, robot_model, args.output / name, moving=args.moving)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["clear"]["corridor_floor_cells"] > 0 and result["clear"]["corridor_raised_cells"] == 0, result
    assert result["occupied"]["corridor_raised_cells"] > 0, result
    if args.moving:
        for case in (result["clear"], result["occupied"]):
            odometry = case["odometry"]
            assert odometry["distance_from_start_m"] > .4 and abs(odometry["final_yaw_deg"]) > 15, odometry
            assert odometry["max_xy_error_m"] < .06 and odometry["max_yaw_error_deg"] < 5., odometry


if __name__ == "__main__":
    main()
