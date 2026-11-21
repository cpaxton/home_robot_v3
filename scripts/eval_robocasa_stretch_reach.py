#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Screen frozen RoboCasa starts and sampled Stretch stances with evaluator-only targets."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.motion.mujoco_arm_ik import joint_qpos_addrs, solve_pose_ik
from emet.motion.mujoco_collision import body_subtree
from emet.robots.stretch import STRETCH_ARM_CHAIN

CHAIN = STRETCH_ARM_CHAIN
ARM = CHAIN.joint_names[:-1]
TELESCOPE = tuple(name for name in ARM if name.startswith("joint_arm_l"))


def camera_framing(archive, point):
    with np.load(archive) as capture:
        pose, k, depth = (capture[key] for key in ("camera_pose", "camera_K", "depth"))
        height, width = depth.shape
        ray = pose[:3, :3].T @ (point - pose[:3, 3])
        if ray[2] <= 0:
            return "behind_camera"
        u, v = np.rint((k @ ray)[:2] / ray[2]).astype(int)
        if not (0 <= u < width and 0 <= v < height):
            return "outside_frame"
        surface = float(depth[v, u])
        if not np.isfinite(surface) or surface <= 0:
            return "depth_missing"
        return "foreground_at_projection" if surface < ray[2] - .06 else "inside_frame"


class StanceScreen:
    def __init__(self, folder):
        self.robot = mujoco.MjModel.from_xml_path(str(folder / "stretch-robot.xml"))
        self.room = mujoco.MjModel.from_xml_path(str(folder / "stretch.xml"))
        self.robot_data = mujoco.MjData(self.robot)
        self.room_data = mujoco.MjData(self.room)
        self.robot_base = int(self.robot.body("base_link").jntadr[0])
        self.room_base = int(self.room.body("base_link").jntadr[0])
        self.robot_root_qpos = int(self.robot.jnt_qposadr[self.robot_base])
        self.room_root_qpos = int(self.room.jnt_qposadr[self.room_base])
        self.robot_arm = joint_qpos_addrs(self.robot, CHAIN.joint_names)
        self.room_arm = joint_qpos_addrs(self.room, CHAIN.joint_names)
        self.pitch = self.robot.joint("joint_wrist_pitch").qposadr[0]
        self.robot_bodies = body_subtree(self.room, "base_link")
        self.floor_geoms = {
            int(geom) for geom in range(self.room.ngeom)
            if (self.room.geom(geom).name.startswith("floor") or
                self.room.body(int(self.room.geom_bodyid[geom])).name.startswith("floor"))
        }
        self.spawn = self.room_data.qpos[self.room_root_qpos:self.room_root_qpos + 7].copy()
        for name, position in json.loads((folder / "fixture-joints.json").read_text()).items():
            self.room_data.joint(name).qpos[:] = position

    def contacts(self, root, arm):
        self.room_data.qpos[self.room_root_qpos:self.room_root_qpos + 7] = root
        self.room_data.qpos[self.room_arm] = arm
        mujoco.mj_forward(self.room, self.room_data)
        outside = set()
        for contact in self.room_data.contact:
            if contact.dist >= -.001:
                continue
            a, b = map(int, contact.geom)
            ar, br = (int(self.room.geom_bodyid[g]) in self.robot_bodies for g in (a, b))
            if ar != br:
                other = b if ar else a
                if other not in self.floor_geoms:
                    outside.add(self.room.geom(other).name or f"geom:{other}")
        return sorted(outside)

    def solve(self, root, point):
        results = []
        for pitch in (-1.35, -.70, 0., .45):
            self.robot_data.qpos[self.robot_root_qpos:self.robot_root_qpos + 7] = root
            self.robot_data.qpos[self.robot_arm] = CHAIN.navigation_arm_q
            self.robot_data.qpos[self.pitch] = pitch
            mujoco.mj_kinematics(self.robot, self.robot_data)
            rotation = self.robot_data.body(CHAIN.ee_body).xmat.reshape(3, 3).copy()
            result = solve_pose_ik(
                self.robot, self.robot_data, ee_body=CHAIN.ee_body, joint_names=ARM,
                target_pos=point, target_rotation=rotation, coupled_groups=(TELESCOPE,),
                max_iters=35, tol_m=.012, tol_rad=.12,
            )
            contacts = self.contacts(root, result.qpos[self.robot_arm]) if result.success else None
            results.append({"solved": result.success, "position_error_m": result.pos_error_m,
                            "orientation_error_rad": result.orientation_error_rad,
                            "wrist_pitch_seed_rad": pitch, "endpoint_scene_contacts": contacts})
            if result.success and not contacts:
                break
        return min(results, key=lambda row: (not row["solved"], bool(row["endpoint_scene_contacts"]),
                                             row["position_error_m"]))

    def stances(self, point):
        # Stretch extends to its right. The robot pose here is a scratch
        # kinematic sample and the path from the original base stays untested.
        for angle in np.linspace(0., 2 * np.pi, 8, endpoint=False):
            rotation = Rotation.from_euler("z", angle)
            for reach in (.50, .68):
                root = self.spawn.copy()
                root[:2] = point[:2] - rotation.apply([-.02, -reach, 0.])[:2]
                root[3:] = rotation.as_quat()[[3, 0, 1, 2]]
                if not self.contacts(root, CHAIN.navigation_arm_q):
                    yield root

    def in_place_yaws(self):
        orientation = Rotation.from_quat(self.spawn[[4, 5, 6, 3]])
        for angle in np.linspace(0., 2 * np.pi, 8, endpoint=False)[1:]:
            root = self.spawn.copy()
            root[3:] = (Rotation.from_euler("z", angle) * orientation).as_quat()[[3, 0, 1, 2]]
            if not self.contacts(root, CHAIN.navigation_arm_q):
                yield root


def archived_targets(folder, screen, drawer_reference):
    """Recover reference points from an existing, unchanged fixture archive."""
    model, data = screen.room, screen.room_data
    mujoco.mj_forward(model, data)
    task = folder.name.split("-", 1)[0]
    if task == "PickPlaceCounterToSink":
        basins = {}
        for index in range(model.ngeom):
            name = model.geom(index).name
            if "sink" in name and "_reg_basin" in name:
                fixture, _, suffix = name.partition("_reg_basin")
                basins.setdefault(fixture, {})["sink_interior" + suffix] = data.geom(index).xpos.copy()
        distractor = data.body("distr_sink_main").xpos
        ranked = sorted((min(float(np.linalg.norm(point - distractor)) for point in points.values()), fixture)
                        for fixture, points in basins.items())
        if not ranked or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .1):
            raise ValueError(f"ambiguous archived sink: {folder.name}")
        return {"counter_object": data.body("obj_main").xpos.copy(), **basins[ranked[0][1]]}
    if task == "CategorizeCondiments":
        return {
            target: data.body(body).xpos.copy()
            for target, body in (("counter_bottle", "obj1_main"), ("counter_shaker", "obj2_main"),
                                 ("cabinet_bottle_reference", "cab_obj1_main"),
                                 ("cabinet_shaker_reference", "cab_obj2_main"))
        }
    if task != "OpenDrawer" or drawer_reference is None:
        raise ValueError(f"missing scorer targets: {folder.name}")
    reference_folder = drawer_reference / folder.name
    if json.loads((reference_folder / "fixture-joints.json").read_text()) != json.loads(
        (folder / "fixture-joints.json").read_text()
    ):
        raise ValueError(f"archived drawer joint initialization differs: {folder.name}")
    reference = json.loads((reference_folder / "scorer-only-reach-targets.json").read_text())
    closed = np.asarray(reference["closed_handle"])
    matched = [
        model.body(index).name for index in range(model.nbody)
        if model.body(index).name.endswith("_door_handle_main")
        and np.linalg.norm(data.xpos[index] - closed) < 1e-5
    ]
    if len(matched) != 1:
        raise ValueError(f"archived drawer does not match reference handle: {folder.name}")
    return {name: np.asarray(point) for name, point in reference.items()}


def report_fixture(folder, drawer_reference):
    screen = StanceScreen(folder)
    path = folder / "scorer-only-reach-targets.json"
    targets = ({name: np.asarray(value) for name, value in json.loads(path.read_text()).items()}
               if path.is_file() else archived_targets(folder, screen, drawer_reference))
    rows = {}
    for name, point in targets.items():
        spawn = screen.solve(screen.spawn, point)
        row = {"camera_framing": camera_framing(folder / "camera.npz", point), "spawn": spawn,
               "in_place_yaw": None, "sampled_stance": None}
        if not spawn["solved"] or spawn["endpoint_scene_contacts"]:
            for key, candidates in (("in_place_yaw", screen.in_place_yaws()),
                                    ("sampled_stance", screen.stances(point))):
                for root in candidates:
                    result = screen.solve(root, point)
                    if not result["solved"]:
                        continue
                    result["base_translation_m"] = float(np.linalg.norm(root[:2] - screen.spawn[:2]))
                    result["base_pose"] = root.tolist()
                    current = row[key]
                    if current is None or len(result["endpoint_scene_contacts"]) < len(current["endpoint_scene_contacts"]):
                        row[key] = result
                    if not result["endpoint_scene_contacts"]:
                        break
        rows[name] = row
    return {"targets": rows}


def run(preflight, output, drawer_reference, task_filter):
    report = {
        "scope": "scorer-only point/pose probes and endpoint contacts. Paths, closure, and sensor-led execution untested",
        "preflight": str(preflight),
        "fixtures": {},
    }
    rows = json.loads((preflight / "report.json").read_text())["rows"]
    for row in rows:
        if row["status"] != "passed" or (task_filter and row["task"] != task_filter):
            continue
        case = row["artifact"]
        started = time.monotonic()
        try:
            folder = preflight / case
            for name, key in (("stretch.xml", "fixture_sha256"), ("stretch-robot.xml", "robot_include_sha256"),
                              ("fixture-joints.json", "fixture_joints_sha256"),
                              ("scorer-only-reach-targets.json", "reach_targets_sha256")):
                if key in row and hashlib.sha256((folder / name).read_bytes()).hexdigest() != row[key]:
                    raise ValueError(f"archived fixture content changed: {case}/{name}")
            result = report_fixture(folder, drawer_reference)
            report["fixtures"][case] = {"fixture_sha256": row["fixture_sha256"], **result}
        except ValueError as error:
            report["fixtures"][case] = {"reason": str(error)}
        report["fixtures"][case]["wall_s"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drawer-reference", type=Path, help="Reference preflight with drawer stroke targets.")
    parser.add_argument("--task", choices=("PickPlaceCounterToSink", "OpenDrawer", "CategorizeCondiments"))
    args = parser.parse_args()
    run(args.preflight.resolve(), args.output.resolve(), args.drawer_reference, args.task)
