#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Score selected RGB-D sink patches and robot-only reach from saved Panda captures."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.motion.mujoco_arm_ik import solve_pose_ik
from emet.motion.observed_support import fit_observed_support
from emet.robots.pandaomron import BASE_BODY, CONTROL_JOINTS, GRASP_BODY


def capture(path, camera):
    prefix = {"front": "", "wrist": "wrist_"}[camera]
    with np.load(path / "sensor.npz") as raw:
        return SimpleNamespace(
            rgb=raw[prefix + "rgb"], depth=raw[prefix + "depth"],
            camera_K=raw[prefix + "K"], camera_pose=raw[prefix + "pose"], received_monotonic=0.0,
        )


def selection_mask(shape, item):
    bounds_mask = Image.new("1", shape[::-1])
    draw = ImageDraw.Draw(bounds_mask)
    draw.polygon([tuple(pixel) for pixel in item["polygon"]], fill=1)
    for hole in item.get("exclude", ()):
        draw.polygon([tuple(pixel) for pixel in hole], fill=0)
    return np.asarray(bounds_mask)


def selected_surface(path, item):
    observation = capture(path, item["stream"])
    _, depth, world = capture_arrays(observation)
    valid = selection_mask(depth.shape, item) & (depth > 0.1) & np.isfinite(world).all(axis=-1)
    if not np.any(valid):
        raise ObservationUnavailable("support_geometry_unknown")
    heights, counts = np.unique(np.round(world[..., 2][valid] / 0.002).astype(int), return_counts=True)
    peak = heights[np.argmax(counts)] * 0.002
    mask = valid & (np.abs(world[..., 2] - peak) <= 0.003)
    patch = fit_observed_support(
        observation, mask, observation_id=f"{path.name}:{item['stream']}",
        measurement_margin_m=0.005,
    )
    return patch, int(mask.sum())


def footprint(patch, vertices):
    yaws = np.arange(0, 180, 5)
    rotations = Rotation.from_euler("z", yaws, degrees=True).as_matrix()
    spans = np.ptp(vertices @ rotations.transpose(0, 2, 1), axis=1)[:, :2]
    width = np.ptp(patch.bounds[:, :2], axis=0)
    slack = np.min(width - spans - 2 * patch.measurement_margin_m, axis=1)
    best = np.argmax(slack)
    return {
        "best_yaw_deg": int(yaws[best]), "best_axis_slack_m": float(slack[best]),
        "fitting_yaws": int(np.count_nonzero(slack >= 0)),
    }


def probe_reach(model, qpos, target, yaws):
    data = mujoco.MjData(model)
    results = []
    for yaw in yaws:
        data.qpos[:] = qpos
        orientation = Rotation.from_euler("z", yaw).as_matrix() @ np.diag([1., -1., -1.])
        waypoints = {}
        for phase, clearance in (("approach", 0.12), ("near_surface", 0.0)):
            result = solve_pose_ik(
                model, data, ee_body=GRASP_BODY, joint_names=CONTROL_JOINTS,
                target_pos=target + [0, 0, clearance], target_rotation=orientation,
                max_iters=65, tol_m=0.003, tol_rad=0.05,
            )
            waypoints[phase] = {
                "reachable": bool(result.success), "position_error_m": result.pos_error_m,
                "orientation_error_rad": result.orientation_error_rad,
            }
            if not result.success:
                break
            data.qpos[:] = result.qpos
        results.append({"yaw_deg": float(np.degrees(yaw)), "waypoints": waypoints})
    return results


def pickup_probe(depth, world, selection, source_patch=None):
    if "polygon" in selection:
        if source_patch is None:
            raise ObservationUnavailable("support_geometry_unknown")
        plane = source_patch.plane
        clearance = world[..., 2] - (world[..., :2] @ plane[:2] + plane[2])
        mask = selection_mask(depth.shape, selection) & (depth > 0.1) & np.isfinite(world).all(axis=-1)
        points = world[mask & (clearance > 2 * source_patch.measurement_margin_m)]
        if len(points) < 25:
            raise ObservationUnavailable("payload_surface_unavailable")
        lo, hi = np.quantile(points, [0.02, 0.98], axis=0)
        target = np.r_[(lo[:2] + hi[:2]) / 2, hi[2] + selection["height_offset_m"]]
        return target, np.arange(4) * np.pi / 2, {
            "source": "above-counter depth within annotated region", "points": len(points),
            "central_96_percent_bounds_m": [lo.tolist(), hi.tolist()],
        }
    pixels = [tuple(reversed(pixel)) for pixel in selection["pixels"]]
    if any(not np.isfinite(depth[p]) or depth[p] <= 0.1 for p in pixels):
        raise ValueError("pickup pixel has no valid depth")
    points = [world[p] for p in pixels]
    pick = points[0] + [0, 0, selection["height_offset_m"]]
    if len(points) == 3:
        axis = points[1] - points[2]
        yaw = np.arctan2(axis[1], axis[0]) + np.pi / 2
        yaws = yaw + np.array([0, np.pi])
    else:
        yaws = np.arange(4) * np.pi / 2
    return pick, yaws, {"source": "annotated depth pixel"}


def evaluate_case(root, case):
    capture_names = {case["robot_state"], case["pickup"]["capture"], *(row["capture"] for row in case["surfaces"])}
    fixture_ids = {
        json.loads((root / name / "report.json").read_text())["physics"]["fixture_sha256"]
        for name in capture_names
    }
    if len(fixture_ids) != 1:
        raise ValueError("RGB-D selections span different fixtures")
    vertices = np.asarray(json.loads((root / case["prior"]).read_text())["vertices_reference_m"], dtype=float)
    surfaces = []
    basins = []
    counters = {}
    for item in case["surfaces"]:
        try:
            patch, count = selected_surface(root / item["capture"], item)
        except ObservationUnavailable as error:
            surfaces.append({**item, "status": "unavailable", "reason": str(error)})
            continue
        row = {**item, "status": "measured", "bounds_m": patch.bounds.tolist(),
               "plane": patch.plane.tolist(), "plane_residual_m": patch.residual_m, "selected_points": count}
        if item["role"] == "basin":
            row.update(footprint(patch, vertices))
            basins.append((patch, row))
        else:
            counters[(item["capture"], item["stream"])] = patch
        surfaces.append(row)

    state = root / case["robot_state"]
    model = mujoco.MjModel.from_xml_path(str(state / "robot-only.xml"))
    with np.load(state / "sensor.npz") as raw:
        qpos = raw["robot_qpos"]
    robot_state = mujoco.MjData(model)
    robot_state.qpos[:] = qpos
    mujoco.mj_kinematics(model, robot_state)
    item = case["pickup"]
    _, depth, world = capture_arrays(capture(root / item["capture"], item["stream"]))
    source = counters.get((item["capture"], item["stream"]))
    pick, yaws, pick_evidence = pickup_probe(depth, world, item, source)
    if "release_gap_m" in item:
        if source is None:
            raise ObservationUnavailable("support_geometry_unknown")
        pickup_plane_z = pick[:2] @ source.plane[:2] + source.plane[2]
        release_height = pick[2] - pickup_plane_z + item["release_gap_m"]
    else:
        release_height = item["release_height_m"]
    reach = {
        "base_xy_m": robot_state.body(BASE_BODY).xpos[:2].tolist(),
        "pickup": {"target_xyz_m": pick.tolist(), "evidence": pick_evidence,
                   "trials": probe_reach(model, qpos, pick, yaws)},
    }
    reach["basins"] = []
    for basin, row in basins:
        xy = basin.bounds[:, :2].mean(axis=0)
        z = xy @ basin.plane[:2] + basin.plane[2] + release_height
        release = np.r_[xy, z]
        yaw = np.deg2rad(row["best_yaw_deg"])
        trials = probe_reach(model, qpos, release, yaw + np.arange(4) * np.pi / 2)
        reach["basins"].append({
            "target_xyz_m": release.tolist(), "capture": row["capture"], "stream": row["stream"],
            "footprint_fits": row["fitting_yaws"] > 0, "best_axis_slack_m": row["best_axis_slack_m"],
            "trials": trials,
        })
    return {"fixture_sha256": fixture_ids.pop(), "surfaces": surfaces, "robot_only_reach": reach}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selections", type=Path, help="Case-keyed JSON with manually selected RGB-D pixels.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.selections.read_text())
    report = {
        "scope": "Read-only diagnostic from annotated RGB-D, an independent object hull and robot-only FK/IK.",
        "execution_qualified": False,
        "cases": {name: evaluate_case(args.selections.parent, case) for name, case in cases.items()},
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
