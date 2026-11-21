#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Drive the development reset route with Stretch wheel actuators and a native guard."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from eval_robocasa_sink_pickup import KitchenContactRobot
from eval_robocasa_stretch_reach import StanceScreen
from emet.eval.physical_tamp import SceneNavigationSpace, base_pose, make_scene_checker
from emet.motion.control.goto_controller import GotoVelocityController
from emet.motion.navigation_sweep import execute_measured_route
from emet.simulation.stretch_mujoco.mujoco_server import BaseController
from emet.simulation.stretch_mujoco.utils import diff_drive_inv_kinematics


class MeasuredKitchenDrive(KitchenContactRobot):
    def __init__(self, screen, pose, *, calibrated_wheels):
        super().__init__(screen, pose)
        self.calibrated_wheels = calibrated_wheels
        self.controller = GotoVelocityController()
        self.controller.update_velocity_profile(v_max=.08 if calibrated_wheels else .12,
                                                w_max=.4, acc_lin=.25, acc_ang=.4)
        self.wheels = BaseController(SimpleNamespace(mjdata=self.data))
        checker = make_scene_checker(self.model, {"object_body": "base_link", "gripper_bodies": ()})
        self.space = SceneNavigationSpace(
            self.model, mujoco.MjData(self.model), checker, allow_reverse=True, route_timeout_s=9.,
        )
        self.wheel_dofs = [
            int(self.model.joint(f"joint_{name}_wheel").dofadr[0]) for name in ("left", "right")
        ]
        self.root_dof = int(self.model.jnt_dofadr[screen.room_base])
        self.samples = []
        self._command_receipt = None

    def measure(self):
        mujoco.mj_copyData(self.space.data, self.model, self.data)
        return base_pose(self.model, self.data)

    def stopped(self):
        return bool(
            np.max(np.abs(self.data.qvel[self.wheel_dofs])) < .04
            and np.linalg.norm(self.data.qvel[self.root_dof:self.root_dof + 2]) < .008
            and abs(self.data.qvel[self.root_dof + 5]) < .02
        )

    def command_velocity(self, v, omega):
        if not self.calibrated_wheels:
            self.wheels._set_base_velocity(v, omega)
            return
        # The MuJoCo velocity actuator tracks length rate after gear scaling.
        # Feed-forward offsets the declared wheel-joint friction loss.
        speeds = diff_drive_inv_kinematics(v, omega)
        for side, speed, dof in zip(("left", "right"), speeds, self.wheel_dofs, strict=True):
            aid = self.model.actuator(f"{side}_wheel_vel").id
            gear = self.model.actuator_gear[aid, 0]
            gain = self.model.actuator_gainprm[aid, 0]
            loss = self.model.dof_frictionloss[dof]
            command = gear * speed + np.sign(speed) * loss / (gear * gain)
            self.data.ctrl[aid] = np.clip(command, *self.model.actuator_ctrlrange[aid])

    def cancel_navigation(self):
        self.command_velocity(0., 0.)
        for _ in range(20):
            self.step(.05)
            if self.stopped():
                return True
        return False

    def move_base_to(self, goal, *, blocking, world_frame, timeout, **options):
        if not blocking or not world_frame:
            raise ValueError("the drive control requires a blocking world-frame route")
        self.controller.update_pose_feedback(self.measure())
        self.controller.update_goal(np.asarray(goal))
        self.controller.control.set_linear_error_tolerance(.017)
        self.controller.control.set_angular_error_tolerance(.026)
        started = float(self.data.time)
        quiet_since = None
        reason = "waypoint_sim_timeout"
        try:
            while self.data.time - started < timeout:
                pose = self.measure()
                if not self.space.is_valid(pose):
                    reason = "measured_pose_guard:" + self.space.last_validity["reason"]
                    break
                self.controller.update_pose_feedback(pose)
                v, omega = self.controller.compute_control()
                self.command_velocity(v, omega)
                self.samples.append({"simulation_time": float(self.data.time), "xyt": pose.tolist(),
                                     "velocity_command": [float(v), float(omega)],
                                     "min_clearance_m": self.space.last_validity.get("min_clearance_m")})
                if self.controller.is_done() and self.stopped():
                    if quiet_since is None:
                        quiet_since = float(self.data.time)
                    if self.data.time - quiet_since >= .5:
                        reason = "ok"
                        break
                else:
                    quiet_since = None
                self.step(.05)
        finally:
            stopped = self.cancel_navigation()
        self._command_receipt = {"result": {"reason": reason, "stop_confirmed": stopped}}
        return reason == "ok" and stopped


def run(fixture, handoff, output, *, calibrated_wheels):
    output.mkdir(parents=True, exist_ok=False)
    if hashlib.sha256((fixture / "stretch.xml").read_bytes()).hexdigest() != handoff["fixture_sha256"]:
        raise ValueError("route refers to a different scene")
    reset = handoff["development_reset"]
    if not reset["approach"]["passed"]:
        raise ValueError("no proposed reset approach")
    started = time.monotonic()
    robot = MeasuredKitchenDrive(StanceScreen(fixture), reset["pose"], calibrated_wheels=calibrated_wheels)
    events = []
    try:
        measured_reset = robot.measure()
        outcome = execute_measured_route(
            robot, goal=handoff["pickup_stance"], measure=robot.measure, space=robot.space,
            plan_route=robot.space.plan_route, initial_route=reset["approach"]["waypoints"],
            max_replans=0, waypoint_timeout_s=18., navigation_policy="precision",
            position_tolerance_m=.02, yaw_tolerance_rad=.03, event=lambda **row: events.append(row),
        )
    finally:
        stopped = robot.cancel_navigation()
    result = {
        "scope": "assisted unloaded physics control using normal wheel actuators and native guard",
        "fixture_sha256": handoff["fixture_sha256"], "measured_reset": measured_reset.tolist(),
        "calibrated_wheel_commands": calibrated_wheels,
        "outcome": asdict(outcome), "stop_confirmed": stopped,
        "guarded_feedback_samples": len(robot.samples), "wall_s": time.monotonic() - started,
        "simulation_s": float(robot.data.time) - 1.5, "events": events,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "base-trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in robot.samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrated-wheel-commands", action="store_true")
    args = parser.parse_args()
    run(args.fixture, json.loads(args.handoff.read_text()), args.output,
        calibrated_wheels=args.calibrated_wheel_commands)
