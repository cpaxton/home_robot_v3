#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Privileged contact-control audit of geometry hidden from motor encoders.

This is calibration/debug data, not an observed task score or a certified error
bound. Ground-truth contacts classify samples for analysis only. Normal native
controllers execute both transfers; only fixture initialization changes qpos.
Run through emet jobs with CPU rendering disabled.
"""

import argparse
import json
from pathlib import Path

import eval_native_table_contact as contact
import mujoco
import numpy as np

from emet.motion.mujoco_collision import body_subtree
from emet.simulation.mujoco_server_stretch import default_scene_xml_path


class AuditedContactRobot(contact.ContactRobot):
    instances = []

    def __init__(self, model, object_body, **kwargs):
        self.audit = {
            name: {
                "samples": 0,
                "max_finger_coupling_error_rad": 0.0,
                "max_telescope_split_error_m": 0.0,
                "max_rubber_joint_angle_rad": 0.0,
                "peak_context": {},
            }
            for name in (
                "initialization:target_contact",
                "initialization:no_target_contact",
                "execution:target_contact",
                "execution:no_target_contact",
            )
        }
        self.finger_joints = [
            int(model.joint(name).qposadr[0])
            for name in ("joint_gripper_finger_left_open", "joint_gripper_finger_right_open")
        ]
        self.slide = int(model.joint("joint_gripper_slide").qposadr[0])
        self.arm_joints = [int(model.joint(f"joint_arm_l{i}").qposadr[0]) for i in range(4)]
        self.rubber = [
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("rubber_")
        ]
        gripper = body_subtree(model, "link_gripper_finger_left") | body_subtree(model, "link_gripper_finger_right")
        target = body_subtree(model, object_body)
        self.gripper_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in gripper}
        self.target_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in target}
        self._initializing = True
        super().__init__(model, object_body, **kwargs)
        self._initializing = False
        self.instances.append(self)

    def step(self, duration):
        for _ in range(max(1, round(duration / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
            touching = any(
                (int(c.geom[0]) in self.gripper_geoms and int(c.geom[1]) in self.target_geoms)
                or (int(c.geom[1]) in self.gripper_geoms and int(c.geom[0]) in self.target_geoms)
                for c in self.data.contact[: self.data.ncon]
                if c.dist <= 0
            )
            phase = "initialization" if self._initializing else "execution"
            group = self.audit[phase + ":" + ("target_contact" if touching else "no_target_contact")]
            group["samples"] += 1
            q = self.data.qpos
            slack = float(np.max(np.abs(q[self.finger_joints] - 10 * q[self.slide])))
            split = float(np.max(np.abs(q[self.arm_joints] - np.mean(q[self.arm_joints]))))
            rubber = float(np.max(np.abs(q[self.rubber]))) if self.rubber else 0.0
            for metric, value in [
                ("max_finger_coupling_error_rad", slack),
                ("max_telescope_split_error_m", split),
                ("max_rubber_joint_angle_rad", rubber),
            ]:
                if value > group[metric]:
                    group[metric] = value
                    group["peak_context"][metric] = {
                        "sim_time": float(self.data.time),
                        "last_command": dict(self.commands[-1]) if self.commands else {"fixture_settling": True},
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    previous = contact.ContactRobot
    rows = []
    try:
        contact.ContactRobot = AuditedContactRobot
        for obj in ("object1", "object2"):
            result = contact.run_case(model, obj)
            (args.output / f"{obj}-control.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            row = {"object": obj, "control_passed": result["passed"], "audit": AuditedContactRobot.instances[-1].audit}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        contact.ContactRobot = previous
    summary = {
        "scope": "privileged simulator calibration/debug only; not observed task acceptance or certified uncertainty",
        "counts": "physics samples within two development control episodes, not independent trials",
        "cases": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0 if all(r["control_passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
