#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Render frozen RoboCasa scenes and verify upstream predicates on Stretch physics."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image

from emet.simulation.sim_manipulation import set_free_body_pose

ROOT = Path(__file__).resolve().parents[1]


class StretchPredicate:
    """Evaluator-owned upstream task bound to the adapted model's named bodies."""

    def __init__(self, env, model):
        from robosuite.utils.binding_utils import MjSim

        self.model = model
        self.sim = MjSim(model)
        self.data = self.sim.data._data
        # The upstream methods retrieve site positions using the robot's
        # eef_site_id. Supply the Stretch grasp center in the scorer view.
        self.grasp_center = model.body("link_grasp_center").id
        self.task = copy.copy(env)
        self.task.obj_body_id = {name: model.body(obj.root_body).id for name, obj in env.objects.items()}
        self.task.robots = (SimpleNamespace(eef_site_id={"right": model.nsite}),)

    def score(self, data):
        self.data.qpos[:] = data.qpos
        self.data.qvel[:] = data.qvel
        self.sim.forward()
        wrapped = self.sim.data

        class ScorerData:
            site_xpos = np.vstack((wrapped.site_xpos, self.data.xpos[self.grasp_center]))

            def __getattr__(self, key):
                return getattr(wrapped, key)

        self.task.sim = SimpleNamespace(model=self.sim.model, data=ScorerData())
        return bool(self.task._check_success())


def predicate_control(task_id, scorer, initial):
    """Produce a known success state on scratch data solely to verify scoring."""
    model, env = scorer.model, scorer.task
    data = mujoco.MjData(model)
    data.qpos[:] = initial.qpos
    mujoco.mj_forward(model, data)
    if task_id == "OpenDrawer":
        data.joint(f"{env.drawer.name}_slidejoint").qpos[:] = -.55 * env.drawer.size[1]
    else:
        if task_id == "PickPlaceCounterToSink":
            p0, px, py, pz = next(iter(env.sink.get_int_sites(relative=False).values()))
            targets = {"obj": p0 + ((px - p0) + (py - p0) + (pz - p0)) / 2}
        else:
            targets = {name: data.body(env.objects["cab_" + name].root_body).xpos.copy()
                       for name in ("obj1", "obj2")}
        for name, point in targets.items():
            if not set_free_body_pose(model, data, env.objects[name].root_body, point):
                raise ValueError(f"scorer control cannot position {name}")
        base_joint = int(model.body("base_link").jntadr[0])
        data.qpos[model.jnt_qposadr[base_joint]] += 3.
    return scorer.score(data)


def fixture_reach_targets(task, env, model, data):
    """Reference points for evaluator-only reachability screening."""
    def object_position(name):
        return data.body(env.objects[name].root_body).xpos.copy().tolist()

    if task == "PickPlaceCounterToSink":
        p0, px, py, pz = next(iter(env.sink.get_int_sites(relative=False).values()))
        return {"counter_object": object_position("obj"),
                "sink_interior": (p0 + ((px - p0) + (py - p0) + (pz - p0)) / 2).tolist()}
    if task == "CategorizeCondiments":
        return {
            target: object_position(name)
            for target, name in (("counter_bottle", "obj1"), ("counter_shaker", "obj2"),
                                 ("cabinet_bottle_reference", "cab_obj1"), ("cabinet_shaker_reference", "cab_obj2"))
        }
    handle = f"{env.drawer.name}_door_handle_main"
    before = data.body(handle).xpos.copy().tolist()
    opened = mujoco.MjData(model)
    opened.qpos[:] = data.qpos
    opened.joint(f"{env.drawer.name}_slidejoint").qpos[:] = -.55 * env.drawer.size[1]
    mujoco.mj_kinematics(model, opened)
    return {"closed_handle": before, "opened_handle": opened.body(handle).xpos.copy().tolist()}


def inspect_fixture(task, env, model, out):
    from emet.simulation.depth_registration import camera_opencv_pose, rendered_camera_intrinsics

    data = mujoco.MjData(model)
    fixture_joints = {}
    for index in range(model.njnt):
        if model.jnt_type[index] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = model.joint(index).name
        try:
            reference = env.sim.data.get_joint_qpos(name)
        except (ValueError, KeyError):
            continue
        data.joint(index).qpos[:] = reference
        fixture_joints[name] = float(reference)
    mujoco.mj_forward(model, data)
    camera = model.camera("d435i_camera_rgb").id
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    intrinsic = rendered_camera_intrinsics(float(model.cam_fovy[camera]), 320, 240)
    pose = camera_opencv_pose(model, data, camera, upright=False)
    Image.fromarray(rgb).save(out / "camera.png")
    np.savez_compressed(out / "camera.npz", rgb=rgb, depth=depth, camera_K=intrinsic, camera_pose=pose)
    (out / "fixture-joints.json").write_text(json.dumps(fixture_joints, indent=2) + "\n")
    valid = np.isfinite(depth) & (depth > .1) & (depth < 5.)
    if np.count_nonzero(valid) < 200 or np.std(rgb) < 1.:
        raise ValueError("adapted camera has insufficient metric scene evidence")
    scorer = StretchPredicate(env, model)
    initial_success = scorer.score(data)
    constructed_success = predicate_control(task, scorer, data)
    if initial_success or not constructed_success:
        raise ValueError(f"upstream predicate controls failed ({initial_success=}, {constructed_success=})")
    targets = fixture_reach_targets(task, env, model, data)
    (out / "scorer-only-reach-targets.json").write_text(json.dumps(targets, indent=2) + "\n")
    return {
        "valid_depth_pixels": int(np.count_nonzero(valid)),
        "predicate": f"{type(env).__module__}.{type(env).__name__}._check_success",
        "initial_success": initial_success,
        "scorer_only_constructed_success": constructed_success,
        "preserved_fixture_joints": len(fixture_joints),
    }


def run(source, protocol_path, output, task_filter):
    protocol = json.loads(protocol_path.read_text())
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != protocol["robocasa_revision"]:
        raise ValueError(f"expected RoboCasa revision {protocol['robocasa_revision']}, found {revision}")
    if subprocess.check_output(["git", "-C", str(source), "diff", "--name-only", "HEAD"], text=True).strip():
        raise ValueError("the pinned RoboCasa tracked source has local changes")
    sys.path.insert(0, str(source))
    import robocasa

    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    if not Path(robocasa.__file__).resolve().is_relative_to(source):
        raise ValueError("RoboCasa imported from outside the pinned source checkout")
    if robocasa.__version__ != protocol["robocasa_version"]:
        raise ValueError("RoboCasa package version does not match the frozen pilot")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Stretch adaptation renderer and predicate control",
        "robocasa_revision": revision,
        "source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "mujoco_version": mujoco.__version__,
        "robosuite_version": version("robosuite"),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "rows": [],
    }
    for task in protocol["tasks"]:
        if task_filter and task["id"] != task_filter:
            continue
        for split in ("development", "held_out"):
            fixture = protocol[split]
            for seed in fixture["seeds"]:
                case = output / f"{task['id']}-{split}-{seed}"
                case.mkdir()
                row = {"task": task["id"], "split": split, "seed": seed, "layout": fixture["layout"],
                       "style": fixture["style"], "artifact": case.name}
                report["rows"].append(row)
                started = time.monotonic()
                try:
                    def inspect(env, model):
                        row.update(inspect_fixture(task["id"], env, model, case))

                    model_generation_wizard(task=task["id"], layout=fixture["layout"], style=fixture["style"],
                                            robot="stretch", seed=seed, inspect_environment=inspect,
                                            write_to_file=str(case / "stretch.xml"))
                    row.update(status="passed", task_execution="pending_observed_admission")
                    row["fixture_sha256"] = hashlib.sha256((case / "stretch.xml").read_bytes()).hexdigest()
                    row["robot_include_sha256"] = hashlib.sha256((case / "stretch-robot.xml").read_bytes()).hexdigest()
                    row["fixture_joints_sha256"] = hashlib.sha256((case / "fixture-joints.json").read_bytes()).hexdigest()
                    row["reach_targets_sha256"] = hashlib.sha256(
                        (case / "scorer-only-reach-targets.json").read_bytes()
                    ).hexdigest()
                except (ValueError, RuntimeError, OSError, KeyError, AssertionError) as error:
                    row.update(status="blocked", reason=f"{type(error).__name__}: {error}")
                row["wall_s"] = time.monotonic() - started
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocasa-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/eval/robocasa_stretch_pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("PickPlaceCounterToSink", "OpenDrawer", "CategorizeCondiments"))
    args = parser.parse_args()
    result = run(args.robocasa_source.resolve(), args.protocol, args.output, args.task)
    print(json.dumps(result, indent=2))
    sys.exit(any(row["status"] != "passed" for row in result["rows"]))
