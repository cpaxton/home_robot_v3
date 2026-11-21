#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Scorer-only route, arm path, and RGB-D visibility controls for a frozen sink fixture."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from eval_robocasa_stretch_reach import ARM, CHAIN, TELESCOPE, StanceScreen
from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.eval.physical_tamp import SceneNavigationSpace, base_pose, make_scene_checker, write_offline_base_pose
from emet.motion.mujoco_collision import body_subtree
from emet.simulation.depth_registration import camera_opencv_pose, rendered_camera_intrinsics

SHAPE = (240, 320)


def xyt(root):
    root = np.asarray(root)
    return np.r_[root[:2], Rotation.from_quat(root[[4, 5, 6, 3]]).as_euler("xyz")[2]]


def set_pose(model, data, pose):
    if not write_offline_base_pose(model, data, base_body_name="base_link",
                                   x=pose[0], y=pose[1], theta=pose[2]):
        raise ValueError("missing offline Stretch free joint")


def route_probe(space, start, goal):
    route = space.plan_route(start, goal)
    return {"passed": bool(route), "waypoints": route, "last_validity": dict(space.last_validity)}


def arm_probe(screen, initial, pose, targets, *, lift_first=None):
    model, data = screen.room, screen.room_data
    checker = make_scene_checker(model, {"object_body": "base_link", "gripper_bodies": ()})
    executor = PhysicalPickPlaceExecutor(None, model=model, data=data, ee_body=CHAIN.ee_body,
        joint_names=ARM, collision=checker, synchronize=None, command_joints=None,
        coupled_groups=(TELESCOPE,), input_mode="privileged",
        gripper_open_configuration=CHAIN.gripper_open_configuration, position_tolerance_m=.003)
    before = data.qpos.copy()
    trials = []
    try:
        for pitch in (-1.35, -.70, 0., .45):
            data.qpos[:] = initial
            set_pose(model, data, pose)
            pitch_address = model.joint("joint_wrist_pitch").qposadr[0]
            initial_pitch = data.qpos[pitch_address]
            data.qpos[pitch_address] = pitch
            mujoco.mj_kinematics(model, data)
            rotation = data.body(CHAIN.ee_body).xmat.reshape(3, 3).copy()
            data.qpos[pitch_address] = initial_pitch
            opening, error = executor.plan_gripper_open()
            row = {"wrist_pitch_rad": pitch, "stages": [], "passed": False}
            trials.append(row)
            if error:
                row["reason"] = error
                continue
            row["opening_samples"] = len(opening)
            if lift_first is not None:
                goal = data.qpos[executor.qadr].copy()
                goal[ARM.index("joint_lift")] = lift_first
                path, error = executor.plan_joint_target(goal)
                row["stages"].append({"stage": "raise_stowed_lift", "target_lift_m": lift_first,
                                      "samples": len(path) if path is not None else 0, "reason": error,
                                      "contacts": checker.last_contacts if error else []})
                if error:
                    continue
            for name, point in targets:
                path, error = executor.plan_pose(point, rotation, ordered_groups=not row["stages"])
                row["stages"].append({"stage": name, "target_world_m": point.tolist(),
                                      "samples": len(path) if path is not None else 0, "reason": error,
                                      "contacts": checker.last_contacts if error else []})
                if error:
                    break
            else:
                row["passed"] = True
                break
        return {"passed": any(row["passed"] for row in trials), "trials": trials}
    finally:
        data.qpos[:] = before
        mujoco.mj_kinematics(model, data)


def view_candidates(model, data, camera, k, point):
    pan, tilt = (model.joint("joint_head_" + name) for name in ("pan", "tilt"))
    qa, qb = (int(joint.qposadr[0]) for joint in (pan, tilt))
    candidates = []
    for a in np.linspace(*pan.range, 25):
        for b in np.linspace(*tilt.range, 10):
            data.qpos[[qa, qb]] = a, b
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            pose = camera_opencv_pose(model, data, camera)
            local = pose[:3, :3].T @ (point - pose[:3, 3])
            if local[2] <= 0:
                continue
            uv = (k @ local)[:2] / local[2]
            if 0 <= uv[0] < SHAPE[1] and 0 <= uv[1] < SHAPE[0]:
                candidates.append((float(np.linalg.norm(uv - k[:2, 2])), float(a), float(b)))
    return (qa, qb), sorted(candidates)[:8]


def rgbd_probe(model, data, renderer, output, point, roles, *, radius_m=None):
    camera = model.camera("d435i_camera_rgb").id
    k = rendered_camera_intrinsics(float(model.cam_fovy[camera]), SHAPE[1], SHAPE[0])
    before = data.qpos.copy()
    best = None
    try:
        addresses, candidates = view_candidates(model, data, camera, k, point)
        for _, pan, tilt in candidates:
            data.qpos[list(addresses)] = [pan, tilt]
            mujoco.mj_forward(model, data)
            pose = camera_opencv_pose(model, data, camera)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            # Simulator IDs stay scorer-side. The saved sensor packet contains
            # only the camera data and calibration.
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[..., 0], roles)
            mask &= np.isfinite(depth) & (depth > .1) & (depth < 5.)
            if radius_m is not None:
                obs = SimpleNamespace(rgb=rgb, depth=depth, camera_K=k, camera_pose=pose)
                _, _, world = capture_arrays(obs)
                mask &= np.linalg.norm(world[..., :2] - point[:2], axis=-1) <= radius_m
            visible = int(mask.sum())
            if best is None or visible > best["scorer_visible_depth_pixels"]:
                best = {"pan_rad": pan, "tilt_rad": tilt, "scorer_visible_depth_pixels": visible}
                np.savez_compressed(output.with_suffix(".npz"), rgb=rgb, depth=depth, camera_K=k, camera_pose=pose)
                Image.fromarray(rgb).save(output.with_suffix(".png"))
        return {"candidate_angles": len(candidates), "best": best}
    finally:
        data.qpos[:] = before
        mujoco.mj_forward(model, data)


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    offset_report = json.loads(args.offsets.read_text())
    fixture_hash = hashlib.sha256((args.fixture / "stretch.xml").read_bytes()).hexdigest()
    if fixture_hash != offset_report["scene_sha256"]:
        raise ValueError("sink offsets refer to a different scene")
    model_screen = StanceScreen(args.fixture)
    model, data = model_screen.room, model_screen.room_data
    data.qpos[model_screen.room_arm] = CHAIN.navigation_arm_q
    initial = data.qpos.copy()
    start = base_pose(model, data)
    source = data.body("obj_main").xpos.copy()
    reach = json.loads(args.reach.read_text())["fixtures"][args.fixture.name]
    if reach["fixture_sha256"] != fixture_hash:
        raise ValueError("pickup stance refers to a different scene")
    pickup_pose = xyt(reach["targets"]["counter_object"]["sampled_stance"]["base_pose"])
    checker = make_scene_checker(model, {"object_body": "base_link", "gripper_bodies": ()})
    nav = SceneNavigationSpace(model, data, checker, allow_reverse=True, route_timeout_s=9.)
    result = {"scope": "scorer-only native geometry, sampled paths and fresh static camera renders",
              "fixture": str(args.fixture), "fixture_sha256": fixture_hash, "frozen_start": start.tolist(),
              "pickup_stance": pickup_pose.tolist(), "landings": []}

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    result["frozen_approach"] = route_probe(nav, start, pickup_pose)
    reset = start.copy()
    reset[0] += .05
    result["development_reset"] = {"pose": reset.tolist(), "offset_m": [.05, 0.],
                                    "approach": route_probe(nav, reset, pickup_pose)}
    save()
    result["pickup_pregrasp"] = arm_probe(model_screen, initial, pickup_pose,
                                          [("pregrasp", source + [0., 0., .08])])
    data.qpos[:] = initial
    source_ids = body_subtree(model, "obj_main")
    target_geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(source_ids)))
    sink_body = model.body(int(model.geom_bodyid[model.geom(offset_report["basin"]).id])).name
    sink_ids = body_subtree(model, sink_body)
    sink_geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(sink_ids)))
    mujoco.mj_kinematics(model, data)
    rim_reference = float(data.body(sink_body).xpos[2])
    result["sink_rim_reference_z_m"] = rim_reference
    with mujoco.Renderer(model, height=SHAPE[0], width=SHAPE[1]) as renderer:
        result["frozen_pickup_rgbd"] = rgbd_probe(model, data, renderer, out / "frozen-pickup", source, target_geoms)
        set_pose(model, data, pickup_pose)
        result["pickup_rgbd"] = rgbd_probe(model, data, renderer, out / "pickup", source, target_geoms)
        save()
        for index, row in enumerate(offset_report["alternative_sites"]):
            if not row["solved"] or row["endpoint_scene_contacts"]:
                continue
            data.qpos[:] = initial
            goal, pose = np.asarray(row["site_world_m"]), xyt(row["base_pose"])
            landing = {"offset_index": index, "site_world_m": goal.tolist(), "stance": pose.tolist(),
                       "unloaded_route": route_probe(nav, pickup_pose, pose)}
            result["landings"].append(landing)
            if landing["unloaded_route"]["passed"]:
                landing["unloaded_arm_path"] = arm_probe(
                    model_screen, initial, pose, [("preplace", goal + [0., 0., .12]), ("basin", goal)]
                )
                landing["above_rim_arm_path"] = arm_probe(
                    model_screen, initial, pose, [
                        ("high_approach", np.r_[goal[:2], rim_reference + .16]),
                        ("above_rim", np.r_[goal[:2], rim_reference + .07]),
                    ]
                )
                if args.lift_first_landings:
                    away = np.array([-np.sin(pose[2]), np.cos(pose[2])])
                    for offset in (0., .075, .15):
                        farther = pose.copy()
                        farther[:2] += offset * away
                        check = {"base_offset_m": offset, "stance": farther.tolist(),
                                 "route": route_probe(nav, pickup_pose, farther)}
                        landing.setdefault("lift_first_trials", []).append(check)
                        if check["route"]["passed"]:
                            check["arm_path"] = arm_probe(
                                model_screen, initial, farther, [
                                    ("above_rim", np.r_[goal[:2], rim_reference + .08]),
                                    ("rim_clearance", np.r_[goal[:2], rim_reference + .04]),
                                ], lift_first=1.04,
                            )
                            if check["arm_path"]["passed"]:
                                break
                set_pose(model, data, pose)
                landing["rgbd"] = rgbd_probe(model, data, renderer, out / f"landing-{index}",
                                             goal, sink_geoms, radius_m=.08)
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reach", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lift-first-landings", action="store_true",
                        help="Check explicit stowed lift and farther landing stances.")
    run(parser.parse_args())
