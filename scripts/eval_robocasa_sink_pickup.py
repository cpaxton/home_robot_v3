#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Contact pickup on the archived development kitchen at a declared pickup reset."""

import argparse
import hashlib
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from eval_native_table_contact import ContactRobot
from eval_robocasa_sink_handoff import set_pose
from eval_robocasa_stretch_reach import CHAIN, StanceScreen
from emet.controller.manipulation import physical_pick_place
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.task.tamp import native_adapter
from emet.eval.physical_tamp import base_pose
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators

OBJECT = "obj_main"


class KitchenContactRobot(ContactRobot):
    def __init__(self, screen, pose):
        self.model, self.data = screen.room, screen.room_data
        self.signature = model_signature(self.model)
        self._state_lock = threading.RLock()
        self.commands = []
        set_pose(self.model, self.data, pose)
        self.data.qpos[screen.room_arm] = CHAIN.navigation_arm_q
        values = dict(zip(CHAIN.joint_names, CHAIN.navigation_arm_q, strict=True))
        controls = {"lift": values["joint_lift"], "arm": sum(values[f"joint_arm_l{i}"] for i in range(4)),
                    "gripper": .04}
        controls.update({name: values[f"joint_{name}"] for name in ("wrist_yaw", "wrist_pitch", "wrist_roll")})
        for name, value in controls.items():
            self.data.ctrl[self.model.actuator(name).id] = value
        snap_joint_qpos_to_ctrl_for_position_actuators(self.model, self.data)
        self.step(1.5)
        self.refresh()

    def get_emet_session(self):
        return {"is_simulation": True, "sim_object_placements": {OBJECT: {"cat": "tongs"}}}


class BaseAlignedPickup(native_adapter.NativeTableAdapter):
    pregrasp_lift_m = .04

    def _grasp_targets(self, grasp, rotation):
        yaw = base_pose(self.model, self.data)[2]
        withdrawal = .08 * np.array([-np.sin(yaw), np.cos(yaw), 0.])
        return [(grasp + withdrawal + [0., 0., self.pregrasp_lift_m], rotation), (grasp, rotation),
                (grasp + [0., 0., .17], rotation)]


def run(fixture, handoff, output, *, base_frame_approach, pregrasp_lift_m):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    fixture_hash = hashlib.sha256((fixture / "stretch.xml").read_bytes()).hexdigest()
    if fixture_hash != handoff["fixture_sha256"]:
        raise ValueError("handoff report refers to a different scene")
    screen = StanceScreen(fixture)
    robot = KitchenContactRobot(screen, handoff["pickup_stance"])
    model = robot.model
    adapter_type = BaseAlignedPickup if base_frame_approach else native_adapter.NativeTableAdapter
    adapter = adapter_type(robot, model=model)
    if base_frame_approach:
        adapter.pregrasp_lift_m = pregrasp_lift_m
    adapter.max_planning_ik_calls = 192
    synchronize = adapter.synchronize

    def refresh(data):
        robot.refresh()
        synchronize(data)

    adapter.synchronize = refresh
    executor = adapter._executor(OBJECT)
    initial_base = base_pose(model, robot.data)
    initial_object = robot.data.body(OBJECT).xpos.copy()
    source = adapter.data.qpos.copy()
    bbox = model.geom("obj_reg_bbox")
    mujoco.mj_kinematics(model, robot.data)
    center = robot.data.geom(bbox.id).xpos.copy()
    longest = int(np.argmax(bbox.size))
    axis = robot.data.geom(bbox.id).xmat.reshape(3, 3)[:, longest]
    rotation = adapter.data.body(executor.ee_body).xmat.reshape(3, 3).copy()
    frames = adapter._grasp_frames(rotation, rotation, 2 * float(np.min(bbox.size)))
    choice = None
    reason = "no_admitted_grasp"
    result = {"scope": "assisted in-process MuJoCo contact control at a declared pickup reset",
              "fixture_sha256": fixture_hash, "base_frame_approach": base_frame_approach,
              "pregrasp_lift_m": pregrasp_lift_m,
              "requested_pickup_xyt": handoff["pickup_stance"],
              "measured_pickup_xyt": initial_base.tolist(), "initial_object_xyz": initial_object.tolist()}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    # Candidate generation only reads the native scene. Moving the object in
    # private collision data represents hypothetical retention during lift.
    adapter._planning_deadline = started + 55.
    for offset in (0., .28, -.28):
        try:
            iterator = adapter._grasp_options(executor, source, center + axis * bbox.size[longest] * offset,
                                              frames, adapter._pinch_center_offset(), OBJECT)
            choice = next(iterator, None)
        except ValueError as error:
            if str(error) != "no_plan_within_budget":
                raise
            reason = str(error)
            break
        if choice is not None:
            executor.grasp_paths = choice[1]
            result["grasp_center_offset_fraction"] = offset
            break
    executor.collision.set_payload(model, adapter.data, None)
    adapter.synchronize(adapter.data)
    if choice is not None:
        old_clock = physical_pick_place.time
        try:
            physical_pick_place.time = SimpleNamespace(monotonic=lambda: float(robot.data.time), sleep=robot.step)
            receipt = executor.grasp_only("tongs", object_gt_body=OBJECT)
        finally:
            physical_pick_place.time = old_clock
        result["receipt"] = asdict(receipt)
        reason = receipt.message
    robot.refresh()
    final_object = robot.data.body(OBJECT).xpos.copy()
    achieved = float(final_object[2] - initial_object[2])
    result.update(
        passed=bool(result.get("receipt", {}).get("success") and achieved >= .05), reason=reason,
        lift_height_m=achieved, final_object_xyz=final_object.tolist(),
        final_base_xyt=base_pose(model, robot.data).tolist(), planning_ik_calls=adapter.planning_ik_calls,
        planning_rejections=adapter.planning_rejections, events=adapter.events,
        commands=robot.commands, wall_s=time.monotonic() - started,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-frame-approach", action="store_true")
    parser.add_argument("--pregrasp-lift-m", type=float, choices=(.04, .10), default=.04)
    args = parser.parse_args()
    if args.pregrasp_lift_m != .04 and not args.base_frame_approach:
        parser.error("--pregrasp-lift-m requires --base-frame-approach")
    run(args.fixture, json.loads(args.handoff.read_text()), args.output, base_frame_approach=args.base_frame_approach,
        pregrasp_lift_m=args.pregrasp_lift_m)
