#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Run the native RoboCasa PandaOmron sink control on an archived Stretch fixture."""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import product, zip_longest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import robocasa
import robosuite
from robocasa.utils import object_utils
from scipy.spatial.transform import Rotation

from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.mujoco_arm_ik import solve_pose_ik
from emet.motion.mujoco_collision import body_subtree
from emet.simulation.pandaomron_environment import make_pandaomron_environment

ARM = tuple(f"robot0_joint{i}" for i in range(1, 8))
BASE = tuple(f"mobilebase0_joint_mobile_{i}" for i in ("forward", "side", "yaw"))
TORSO = "mobilebase0_joint_torso_height"
EE = "gripper0_right_eef"
OBJECTS = ("obj_main", "distr_counter_main", "distr_sink_main")
ROBOT_BASE = "mobilebase0_base"
TASK = "PickPlaceCounterToSink"
LAYOUT = STYLE = 11
SEED = 0
CONTROL_HZ = 20
PENETRATION_LIMIT_M = 0.0003
BASIN_LANDING_HEIGHT_M = 0.918
CARRIED_STAGES = ("break_contact", "lift", "over_sink", "lower")


def merge_straight_waypoints(waypoints):
    """Keep path turns and reversals, so certified straight segments execute continuously."""
    keep = [waypoints[0]]
    for current, following in zip(waypoints[1:-1], waypoints[2:], strict=True):
        before = current - keep[-1]
        after = following - current
        scale = np.linalg.norm(before) * np.linalg.norm(after)
        if scale < 1e-10 or np.dot(before, after) < (1 - 1e-12) * scale:
            keep.append(current)
    keep.append(waypoints[-1])
    return keep


class Control:
    def __init__(self, fixture, output, *, layout=LAYOUT, style=STYLE, seed=SEED, plan_torso=False):
        self.output = output
        np.random.seed(seed)
        random.seed(seed)
        self.env = env = make_pandaomron_environment(task=TASK, layout=layout, style=style, seed=seed)
        # RoboCasa lazily builds its fixture references. Build that XML first,
        # without compiling an unused random asset into a native simulation.
        # The reset below then compiles the exact frozen object metadata.
        env._load_model()
        archive = ET.parse(fixture / "stretch.xml")
        objects = env._get_obj_cfgs()
        for cfg in objects:
            meshes = [
                Path(node.get("file"))
                for node in archive.findall("./asset/mesh")
                if node.get("name", "").startswith(cfg["name"] + "_") and "/objects/" in node.get("file", "")
            ]
            models = {part / "model.xml" for mesh in meshes for part in mesh.parents if (part / "model.xml").is_file()}
            (model_path,) = models
            cfg["info"] = {"mjcf_path": str(model_path)}
            cfg["type"] = "object"
        env.set_ep_meta({"object_cfgs": objects, "layout_id": layout, "style_id": style})
        env.reset()
        env.hard_reset = False
        self.robot = env.robots[0]
        self.model = m = env.sim.model._model
        self.live = d = env.sim.data._data
        self.scratch = mujoco.MjData(m)
        self.joint_names = (*ARM, TORSO) if plan_torso else ARM
        self.arm = np.array([m.joint(name).qposadr[0] for name in self.joint_names])
        self.base = np.array([m.joint(name).qposadr[0] for name in BASE])
        self.torso = int(m.joint(TORSO).qposadr[0])
        self.fingers = np.array([m.joint(f"gripper0_right_finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.robot_bodies = body_subtree(m, ROBOT_BASE)
        self.obj_bodies = body_subtree(m, OBJECTS[0])
        self.object_joint = int(m.body(OBJECTS[0]).jntadr[0])
        self.object_qpos = int(m.jnt_qposadr[self.object_joint])
        self.pad_geoms = {m.geom(f"gripper0_right_finger{i}_pad_collision").id for i in (1, 2)}
        self.finger_geoms = self.pad_geoms | {m.geom(f"gripper0_right_finger{i}_collision").id for i in (1, 2)}
        self.floor = {
            g for g in range(m.ngeom) if "floor" in m.geom(g).name or "floor" in m.body(m.geom_bodyid[g]).name
        }
        self.initial = d.qpos.copy()
        self.restore_fixture(fixture)
        d.qpos[:] = self.initial
        mujoco.mj_forward(m, d)
        regions = env.sink.get_int_sites(relative=False)
        basins = [region for name, region in sorted(regions.items()) if name.startswith("basin")]
        if not basins:
            raise ValueError("RoboCasa sink exposes no basin region")
        self.basins = [
            (origin, np.column_stack((front - origin, side - origin, top - origin)))
            for origin, front, side, top in basins
        ]
        origin, front, side, _ = basins[0]
        self.destination = origin + 0.32 * (front - origin) + 0.37 * (side - origin)
        self.destination[2] = BASIN_LANDING_HEIGHT_M
        self.actions = []
        self.samples = []
        self.rejections = Counter()
        self.report = {
            "task": TASK,
            "robot": "PandaOmron",
            "scope": "simulator-state-assisted contact control from a stopped base reset",
            "fixture_sha256": hashlib.sha256((fixture / "stretch.xml").read_bytes()).hexdigest(),
            "fixture_joints_sha256": hashlib.sha256((fixture / "fixture-joints.json").read_bytes()).hexdigest(),
            "robocasa_revision": subprocess.check_output(
                ["git", "-C", str(Path(robocasa.__file__).resolve().parent), "rev-parse", "HEAD"], text=True
            ).strip(),
            "robosuite_version": robosuite.__version__,
            "robocasa_version": robocasa.__version__,
            "mujoco_version": mujoco.__version__,
            "layout": layout,
            "style": style,
            "seed": seed,
            "planned_joints": self.joint_names,
            "destination_xyz_m": self.destination.tolist(),
            "object_models": [str(c["info"]["mjcf_path"]).split("/objects/")[1] for c in env.object_cfgs],
            "fixture_joints_restored": len(self.fixture_joints),
            "fixture_body_offsets_restored": self.fixture_offsets_restored,
        }

    def restore_fixture(self, fixture):
        archive = ET.parse(fixture / "stretch.xml")
        scene = ET.fromstring(self.env.sim.model.get_xml())
        body_maps = [{b.get("name"): b for b in xml.findall("./worldbody//body")} for xml in (archive, scene)]
        archived, native = body_maps
        for name in OBJECTS:
            for xml in (archive, scene):
                assert xml.find(f"./worldbody/body[@name='{name}']") is not None, name
            object_model = str(
                next(c["info"]["mjcf_path"] for c in self.env.object_cfgs if c["name"] + "_main" == name)
            )
            parent = str(Path(object_model).parent)

            def object_meshes(xml):
                # Archived fixtures and a restored dependency checkout can
                # name the same asset through different directory symlinks.
                # Compare canonical mesh identities, retaining model checks.
                root = Path(parent).resolve()
                return {
                    str(Path(m.get("file")).resolve().relative_to(root))
                    for m in xml.findall("./asset/mesh")
                    if m.get("file") and Path(m.get("file")).resolve().is_relative_to(root)
                    and m.get("name", "").startswith(name[:-5] + "_")
                }

            if not object_meshes(archive) or object_meshes(archive) != object_meshes(scene):
                raise ValueError(f"object model mismatch for {name}")
            adr = int(self.model.jnt_qposadr[self.model.body(name).jntadr[0]])
            self.initial[adr : adr + 3] = np.fromstring(archived[name].get("pos"), sep=" ")
            self.initial[adr + 3 : adr + 7] = np.fromstring(archived[name].get("quat"), sep=" ")
        common_fixture_bodies = set(archived) - {b.get("name") for n in OBJECTS for b in archived[n].iter("body")}
        common_fixture_bodies.discard(None)
        self.fixture_offsets_restored = []
        for name in sorted(common_fixture_bodies):
            if name not in native:
                raise ValueError(f"fixture body missing from Panda scene: {name}")
            for attr, default in (("pos", "0 0 0"), ("quat", "1 0 0 0")):
                a, b = (np.fromstring(x[name].get(attr, default), sep=" ") for x in body_maps)
                if not np.allclose(a, b, atol=3e-5):
                    body = self.model.body(name)
                    (body.pos if attr == "pos" else body.quat)[:] = a
                    self.fixture_offsets_restored.append(f"{name}.{attr}")
        self.fixture_joints = json.loads((fixture / "fixture-joints.json").read_text())
        for name, q in self.fixture_joints.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f'fixture joint missing from native model: {name}')
            self.initial[int(self.model.jnt_qposadr[joint_id])] = q

    def reset_stance(self, x, y, yaw, torso):
        m, d = self.model, self.live
        d.qpos[:] = self.initial
        d.qvel.fill(0.0)
        d.time = 0.0
        d.qpos[self.torso] = torso
        d.qpos[self.fingers] = [0.04, -0.04]
        mujoco.mj_forward(m, d)
        _, _, current_yaw = self.base_pose()
        # The slide axes are expressed in MJCF joint coordinates.
        # Solve for the slide offsets needed to achieve the declared world base pose.
        d.qpos[self.base[2]] += math.remainder(yaw - current_yaw, 2 * math.pi)
        mujoco.mj_forward(m, d)
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jac, None, m.body(ROBOT_BASE).id)
        d.qpos[self.base[:2]] += np.linalg.solve(jac[:2, self.base[:2]], np.array([x, y]) - d.body(ROBOT_BASE).xpos[:2])
        mujoco.mj_forward(m, d)
        self.stance = d.qpos.copy()
        self.torso_goal = torso

    def base_pose(self):
        base = self.live.body(ROBOT_BASE)
        return np.r_[base.xpos[:2], np.arctan2(base.xmat[3], base.xmat[0])]

    def contacts(self, data, *, allow_finger_contact=False, carried=False, allow_source_contact=False,
                 penetration_limit_m=PENETRATION_LIMIT_M):
        m = self.model
        bad = set()
        pads = set()
        for c in data.contact:
            if c.dist >= -penetration_limit_m:
                continue
            a, b = map(int, c.geom)
            ab, bb = map(int, m.geom_bodyid[[a, b]])
            ar, br = ab in self.robot_bodies, bb in self.robot_bodies
            ao, bo = ab in self.obj_bodies, bb in self.obj_bodies
            if ar and bo and a in self.pad_geoms:
                pads.add(a)
            if br and ao and b in self.pad_geoms:
                pads.add(b)
            if ar != br:
                robot, other = (a, b) if ar else (b, a)
                if other in self.floor and m.geom(robot).name.startswith("mobilebase0_"):
                    continue
                if (bo if ar else ao) and allow_finger_contact and robot in self.finger_geoms:
                    continue
            elif not (carried and (ao != bo)):
                continue
            if (ao or bo) and allow_source_contact and not (ar or br):
                continue
            bad.add(tuple(m.geom(g).name or m.body(m.geom_bodyid[g]).name for g in (a, b)))
        return sorted(bad), pads

    def move_held_payload(self, data, attachment):
        r = data.body(EE).xmat.reshape(3, 3)
        pose = data.qpos[self.object_qpos : self.object_qpos + 7]
        pose[:3] = data.body(EE).xpos + r @ attachment[0]
        pose[3:] = Rotation.from_matrix(r @ attachment[1]).as_quat(scalar_first=True)

    def configuration_collides(self, model, data, *, attachment=None):
        mujoco.mj_kinematics(model, data)
        if attachment is not None:
            self.move_held_payload(data, attachment)
            mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        loaded = attachment is not None
        return bool(self.contacts(data, carried=loaded, allow_finger_contact=loaded)[0])

    def plan(
        self, grasp, rot, *, destination_rotation=None, approach_offset=(0, 0, 0.10),
        approach_rrt=False, carry_rrt=False, lift_m=0.19, retreat_offset=(0, 0, 0.20),
    ):
        m, d = self.model, self.scratch
        d.qpos[:] = self.stance
        mujoco.mj_kinematics(m, d)
        q = d.qpos[self.arm].copy()
        grasp_hi = grasp + approach_offset
        lift = grasp + [0, 0, lift_m]
        end_rot = destination_rotation
        if end_rot is None:
            end_rot = Rotation.from_rotvec([0, 0, -np.arctan2(rot[1, 0], rot[0, 0])]).as_matrix() @ rot
        dest = self.destination.copy()
        dest_hi = dest + [0, 0, 0.20]
        stages = [
            ("pregrasp", grasp_hi, rot),
            ("near_grasp", grasp + [0, 0, 0.027], rot),
            ("grasp", grasp, rot),
            ("break_contact", grasp + [0, 0, 0.025], rot),
            ("lift", lift, rot),
        ]
        stages += [("over_sink", dest_hi, end_rot), ("lower", dest, end_rot),
                   ("retreat", dest + retreat_offset, end_rot)]
        payload = None
        result = []
        for stage, point, rotation in stages:
            d.qpos[self.arm] = q
            sol = solve_pose_ik(
                m,
                d,
                ee_body=EE,
                joint_names=self.joint_names,
                target_pos=point,
                target_rotation=rotation,
                tol_m=0.0015,
                tol_rad=0.035,
                max_iters=65,
            )
            if not sol.success:
                self.rejections[f"{stage}:ik"] += 1
                return None
            goal = sol.qpos[self.arm]
            count = max(2, int(np.ceil(max(abs(goal - q)) / 0.028)))
            carried = stage in CARRIED_STAGES
            if carried:
                d.qpos[self.fingers] = [0.014, -0.014]
            else:
                d.qpos[self.fingers] = [0.04, -0.04]
            for index, state in enumerate(np.linspace(q, goal, count)[1:], 1):
                d.qpos[self.arm] = state
                if carried:
                    mujoco.mj_kinematics(m, d)
                    self.move_held_payload(d, payload)
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                bad, _ = self.contacts(
                    d,
                    allow_finger_contact=carried or stage == "grasp",
                    carried=carried,
                    allow_source_contact=stage == "break_contact" and index <= 2,
                )
                if bad:
                    self.rejections[f"{stage}:contact:{bad[0]}"] += 1
                    break
            if bad:
                if not ((stage == "pregrasp" and approach_rrt) or (stage == "over_sink" and carry_rrt)):
                    return None
                collision = self
                if carried:
                    def loaded_collision(model, data):
                        return self.configuration_collides(model, data, attachment=payload)

                    collision = SimpleNamespace(configuration_collides=loaded_collision)
                detour = plan_arm_joint_path(
                    m, d, joint_names=self.joint_names, q_start=q, q_goal=goal, collision=collision,
                    max_iter=100, step_size=0.035, shortcut_iter=20, linear_fallback=False,
                    rng=np.random.default_rng(0),
                )
                if not detour.success:
                    self.rejections[f"{stage}:detour:{detour.reason}:{detour.detail}"] += 1
                    return None
                sparse = merge_straight_waypoints(detour.waypoints)
                for waypoint in sparse[1:-1]:
                    d.qpos[self.arm] = waypoint
                    mujoco.mj_kinematics(m, d)
                    ee = d.body(EE)
                    result.append((f"{stage}_detour", waypoint, ee.xpos.copy(), ee.xmat.reshape(3, 3).copy()))
                key = f"{stage}_detours"
                self.report[key] = self.report.get(key, 0) + 1
            if stage == "grasp":
                r = d.body(EE).xmat.reshape(3, 3)
                payload = (
                    r.T @ (d.body(OBJECTS[0]).xpos - d.body(EE).xpos),
                    r.T @ d.body(OBJECTS[0]).xmat.reshape(3, 3),
                )
            elif stage == "lower":
                # The released object settles below the retreat path. Do not grant
                # source-object contact permission during withdrawal.
                d.qpos[self.object_qpos + 2] -= 0.05
            result.append((stage, goal.copy(), point, rotation))
            q = goal.copy()
        return result

    def sample_candidates(self):
        count = 0
        for stance in (
            (0.92, -2.43, math.pi, 0.16),
            (0.92, -2.49, math.pi, 0.16),
            (0.93, -2.35, math.pi, 0.16),
            (0.88, -2.53, -2.65, 0.16),
            (0.88, -2.55, -2.65, 0.22),
            (0.92, -2.43, math.pi, 0.23),
            (1.02, -2.44, math.pi, 0.22),
            (0.88, -2.48, math.pi, 0.10),
        ):
            self.env.reset()
            self.reset_stance(*stance)
            self.settle()
            hit, _ = self.contacts(self.live)
            if hit:
                self.rejections[f"reset:contact:{hit[0]}"] += 1
                continue
            bbox = self.live.geom("obj_reg_bbox")
            center = bbox.xpos.copy()
            axis = bbox.xmat.reshape(3, 3)[:, 1].copy()
            yaw = np.arctan2(axis[1], axis[0])
            # Test pinches across the narrow side at several points along the handle.
            mass_offset = float((self.live.xipos[self.model.body(OBJECTS[0]).id] - center) @ axis)
            for offset in (mass_offset, mass_offset - 0.012, mass_offset + 0.012, -0.025, 0.025, -0.048, -0.067, 0.06):
                for opposite in (0.0, np.pi):
                    # The Panda pads close along the grasp frame's X axis,
                    # so point that axis across the handle and Z into the table.
                    a = yaw + np.pi / 2 + opposite
                    rotation = Rotation.from_euler("z", a).as_matrix() @ np.diag([1.0, -1.0, -1.0])
                    for height in (0.001, 0.004, -0.002, 0.007):
                        grasp = center + offset * axis + [0, 0, height]
                        count += 1
                        path = self.plan(grasp, rotation)
                        if path:
                            print("admitted", stance, offset, opposite, height, count, flush=True)
                            yield stance, offset, opposite, height, path
            self.save()
            print("screened stance", stance, count, self.rejections.most_common(5), flush=True)

    def advance(self, stage, q, gripper):
        torso = q[len(ARM)] if len(q) > len(ARM) else self.torso_goal
        action = self.robot.create_action_vector({"right": q[: len(ARM)], "right_gripper": [gripper], "torso": [torso]})
        self.env.step(action)
        self.actions.append(action.tolist())
        obj = self.live.body(OBJECTS[0])
        contacts, pads = self.contacts(
            self.live, allow_finger_contact=stage not in (
                "settle", "pregrasp", "pregrasp_detour", "near_grasp", "retreat", "final_hold"
            )
        )
        self.samples.append(
            {
                "stage": stage,
                "sim_s": float(self.live.time),
                "xyz_m": obj.xpos.tolist(),
                "quaternion_wxyz": obj.xquat.tolist(),
                "base_xyt": self.base_pose().tolist(),
                "pad_count": len(pads),
                "unintended_contacts": contacts,
            }
        )

    def settle(self):
        self.actions = []
        self.samples = []
        home = self.stance[self.arm].copy()
        for _ in range(30):
            self.advance("settle", home, -1.0)
        self.stance = self.live.qpos.copy()

    def execute(self, path):
        start = self.live.body(OBJECTS[0]).xpos.copy()
        start_base = self.base_pose()
        old_goal = self.live.qpos[self.arm].copy()
        result = {"stages": [], "initial_object_xyz_m": start.tolist()}
        grip = -1.0
        for stage, q, point, _ in path:
            transit = stage.endswith("_detour")
            n = max(4 if transit else 12, int(np.ceil(max(abs(q - old_goal)) / 0.018)))
            for i in range(1, n + 1):
                f = i / n
                self.advance(stage, old_goal + (q - old_goal) * (f * f * (3.0 - 2.0 * f)), grip)
            for _ in range(4 if transit else 12):
                self.advance(stage, q, grip)
            loaded_pads = self.samples[-1]["pad_count"]
            if stage == "grasp":
                grip = 1.0
                for _ in range(24):
                    self.advance("close", q, grip)
            if stage == "lower":
                grip = -1.0
                for _ in range(24):
                    self.advance("open", q, grip)
            q_error = float(max(abs(self.live.qpos[self.arm] - q)))
            result["stages"].append(
                {
                    "stage": stage,
                    "object_xyz_m": self.live.body(OBJECTS[0]).xpos.tolist(),
                    "pos_error_m": float(np.linalg.norm(self.live.body(EE).xpos - point)),
                    "joint_error_rad": q_error,
                    "pad_count": self.samples[-1]["pad_count"],
                    "pads_before_gripper_command": loaded_pads,
                }
            )
            print(result["stages"][-1], flush=True)
            if stage == "lift" and (
                self.live.body(OBJECTS[0]).xpos[2] - start[2] < 0.05 or self.samples[-1]["pad_count"] < 2
            ):
                break
            old_goal = q
        for _ in range(28):
            self.advance("final_hold", q, grip)
        self.report.setdefault("attempts", []).append(result)
        object_position = self.live.body(OBJECTS[0]).xpos
        gripper_site = self.live.site_xpos[self.robot.eef_site_id["right"]]
        result.update(
            upstream_success=bool(self.env._check_success()),
            upstream_inside_sink=bool(object_utils.obj_inside_of(self.env, "obj", self.env.sink, partial_check=True)),
            upstream_gripper_far=bool(object_utils.gripper_obj_far(self.env)),
            final_gripper_distance_m=float(np.linalg.norm(gripper_site - object_position)),
            max_lift_m=max(s["xyz_m"][2] for s in self.samples) - float(start[2]),
            motion_contacts=Counter(str(c) for s in self.samples for c in s["unintended_contacts"]),
            final_object_xyz_m=self.live.body(OBJECTS[0]).xpos.tolist(),
            final_object_speed_mps=float(
                np.linalg.norm(self.live.qvel[self.model.jnt_dofadr[self.object_joint] :][:3])
            ),
            sim_s=float(self.live.time),
            steps=len(self.samples),
            max_base_translation_m=max(
                np.linalg.norm(np.asarray(s["base_xyt"][:2]) - start_base[:2]) for s in self.samples
            ),
            basin_footprint_contains_tongs=self.basin_contains_tongs(),
        )
        stage_results = {s["stage"]: s for s in result["stages"]}
        result["retained_to_release"] = all(
            stage_results.get(stage, {}).get("pads_before_gripper_command") == 2
            for stage in ("lift", "over_sink", "lower")
        )
        result["passed"] = bool(
            result["upstream_success"]
            and result["retained_to_release"]
            and result["basin_footprint_contains_tongs"]
            and result["max_lift_m"] >= 0.05
            and result["final_object_speed_mps"] <= 0.01
            and not result["motion_contacts"]
        )
        index = len(self.report["attempts"])
        (self.output / f"attempt-{index}.json").write_text(
            json.dumps({"actions": self.actions, "samples": self.samples}) + "\n"
        )
        return result

    def basin_contains_tongs(self):
        """Check that the resting tongs fit within the basin's horizontal bounds."""
        bbox = self.live.geom("obj_reg_bbox")
        half_size = self.model.geom("obj_reg_bbox").size
        offsets = np.asarray(list(product((-1, 1), repeat=3))) * half_size
        corners = bbox.xpos + offsets @ bbox.xmat.reshape(3, 3).T
        for origin, axes in self.basins:
            normalized = np.linalg.solve(axes, (corners - origin).T)
            if np.all((normalized[:2] >= 0.0) & (normalized[:2] <= 1.0)):
                return True
        return False

    def run(self, args):
        self.save()
        for stance, offset, opposite, height, path in self.sample_candidates():
            # The generator's planning data is scratch only. Restore stopped scene
            # for each real contact rollout.
            self.env.reset()
            self.reset_stance(*stance)
            self.settle()
            result = self.execute(path)
            result.update(stance=stance, handle_offset_m=offset, flipped=bool(opposite), pinch_height_m=height)
            self.save()
            if result["passed"] or len(self.report["attempts"]) >= args.attempts:
                break
        self.save()

    def run_observed_goals(
        self, goals, *, attempts, max_plans, approach_variants, approach_rrt, carry_rrt, lift_heights, retreat_m,
    ):
        """Test image-derived waypoints with simulator-state collision supervision."""
        self.report["scope"] = "image-derived goals with simulator-state collision and physics supervision"
        self.report["picked_from"] = goals["pickup"]["target_xyz_m"]
        self.report["goals_screened"] = 0
        self.save()
        down = np.diag([1.0, -1.0, -1.0])
        pickup = np.asarray(goals["pickup"]["target_xyz_m"], dtype=float)
        pick_yaws = reachable_yaws(goals["pickup"])
        basins = sorted(goals["basins"], key=lambda row: -row["best_axis_slack_m"])
        routes = [[(row, yaw) for yaw in reachable_yaws(row)] for row in basins if row["footprint_fits"]]
        destinations = [route for round_ in zip_longest(*routes) for route in round_ if route is not None]
        direction = np.asarray(goals["base_xy_m"]) - pickup[:2]
        norm = np.linalg.norm(direction)
        if norm < 0.01:
            raise ValueError("pickup is too close to the measured base center")
        direction /= norm
        approaches = [(0.0, 0.0, 0.10)]
        if approach_variants:
            approaches += [(*direction * lateral, height) for lateral, height in
                           ((0.08, 0.07), (0.12, 0.06), (0.17, 0.07), (0.10, 0.12), (0.05, 0.05))]
        qpos = self.initial.copy()
        self.torso_goal = float(qpos[self.torso])

        def reset():
            self.env.reset()
            self.live.qpos[:] = qpos
            self.live.qvel.fill(0.0)
            mujoco.mj_forward(self.model, self.live)
            self.stance = self.live.qpos.copy()
            self.settle()

        reset()
        if self.contacts(self.live)[0]:
            self.rejections["reset:contact"] += 1
            return
        tuning = product((0.0, 0.015, -0.012), (0.0, 0.035), approaches, pick_yaws, destinations, lift_heights)
        for dz, landing_height, approach, pick_yaw, (dest, dest_yaw), lift_m in tuning:
            if self.report["goals_screened"] >= max_plans:
                return
            nominal = np.asarray(dest["target_xyz_m"], dtype=float)
            rot = Rotation.from_euler("z", pick_yaw, degrees=True).as_matrix() @ down
            end_rot = Rotation.from_euler("z", dest_yaw, degrees=True).as_matrix() @ down
            self.destination = nominal + [0, 0, landing_height]
            grasp = pickup + [0, 0, dz]
            self.report["goals_screened"] += 1
            path = self.plan(
                grasp, rot, destination_rotation=end_rot, approach_offset=approach,
                approach_rrt=approach_rrt, carry_rrt=carry_rrt, lift_m=lift_m,
                retreat_offset=(0.0, 0.0, retreat_m),
            )
            if path is None:
                continue
            result = self.execute(path)
            result.update(
                pickup_xyz_m=grasp.tolist(), destination_xyz_m=self.destination.tolist(),
                destination_camera=f"{dest['capture']}:{dest['stream']}", approach_offset_m=approach,
                lift_target_m=lift_m, retreat_target_m=retreat_m,
            )
            self.save()
            if result["passed"] or len(self.report["attempts"]) >= attempts:
                return
            reset()

    def save(self):
        self.report["planning_rejections"] = dict(self.rejections)
        self.report["passed"] = any(a["passed"] for a in self.report.get("attempts", []))
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")


def reachable_yaws(goal):
    return [
        trial["yaw_deg"] for trial in goal["trials"]
        if trial["waypoints"].get("near_surface", {}).get("reachable")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--observed-shaker-goals", type=Path, help="Read the held-out shaker observation diagnostic.")
    parser.add_argument("--max-plans", type=int, default=32)
    parser.add_argument("--shaker-approach-variants", action="store_true")
    parser.add_argument("--shaker-approach-rrt", action="store_true")
    parser.add_argument("--shaker-carry-rrt", action="store_true")
    parser.add_argument("--shaker-lift-m", action="append", type=float)
    parser.add_argument("--shaker-retreat-m", type=float, default=0.20)
    parser.add_argument("--shaker-plan-torso", action="store_true")
    args = parser.parse_args()
    if min(args.attempts, args.max_plans) < 1:
        parser.error("attempt and planning limits must be positive")
    lift_heights = args.shaker_lift_m or [0.19]
    if any(not 0.06 <= value <= 0.25 for value in lift_heights):
        parser.error("shaker lift heights must be within 0.06 to 0.25 m")
    if not 0.20 <= args.shaker_retreat_m <= 0.32:
        parser.error("shaker retreat height must be within 0.20 to 0.32 m")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.observed_shaker_goals:
        control = Control(args.fixture, args.output, layout=1, style=1, seed=1, plan_torso=args.shaker_plan_torso)
        diagnostic = json.loads(args.observed_shaker_goals.read_text())["cases"]["held-out-shaker"]
        if (
            diagnostic["fixture_sha256"] != control.report["fixture_sha256"]
            or control.report["object_models"][0]
            != "lightwheel/salt_and_pepper_shaker/SaltShaker012/model.xml"
        ):
            raise ValueError("observed goals do not match the frozen shaker fixture")
        control.run_observed_goals(
            diagnostic["robot_only_reach"], attempts=args.attempts, max_plans=args.max_plans,
            approach_variants=args.shaker_approach_variants, approach_rrt=args.shaker_approach_rrt,
            carry_rrt=args.shaker_carry_rrt, lift_heights=lift_heights, retreat_m=args.shaker_retreat_m,
        )
    else:
        control = Control(args.fixture, args.output)
        control.run(args)
    control.report["wall_s"] = time.monotonic() - started
    control.save()
    sys.exit(0 if control.report["passed"] else 1)
