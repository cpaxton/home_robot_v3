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
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import eval_native_table_contact as contact
import mujoco
import numpy as np
from prepare_observed_stretch_fixture import prepare

from emet.controller.manipulation.gripper_geometry import bound_stretch_gripper
from emet.controller.manipulation.stretch_encoder_kinematics import estimate_stretch_ee_from_encoders
from emet.motion.mujoco_collision import body_subtree
from emet.motion.placement_geometry import box_corners
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.synchronized_capture import capture_scene_for_sensors, capture_stretch_motors


class AuditedContactRobot(contact.ContactRobot):
    instances = []
    envelope_model = None

    def __init__(self, model, object_body, **kwargs):
        self.envelope_samples, self._next_envelope_time = [], 0.0
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
            if self.envelope_model is not None and self.data.time >= self._next_envelope_time:
                self._next_envelope_time = self.data.time + 0.25
                self._check_envelope(phase, touching)
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

    def _check_envelope(self, phase, touching):
        probe, sample = capture_scene_for_sensors(
            self.model, self.data, self._state_lock, cameras={"head": ("d435i_camera_rgb", True)}
        )
        motors = capture_stretch_motors(self.model, probe, sample)
        stamp = sample["interval_ns"][0]
        timing = {
            "available": True,
            "source": "renderer_synchronized_snapshot",
            "timestamp_ns": stamp,
            "clock_id": sample["clock_id"],
            "clock_domain": "mujoco_render_monotonic",
            "capture_interval_ns": sample["interval_ns"],
            "depth_capture_interval_ns": sample["interval_ns"],
            "motor_sample": motors,
        }
        observation = SimpleNamespace(
            is_simulation=True,
            image_timing={"head_cam_left": timing},
            received_monotonic=time.monotonic(),
            camera_pose=np.array(sample["camera_poses"]["head"]),
        )
        estimate = estimate_stretch_ee_from_encoders(observation, self.envelope_model)
        bound = bound_stretch_gripper(self.envelope_model, estimate)
        escape, violations = 0.0, []
        for name, lo, hi in bound.joint_intervals:
            value = float(probe.joint(name).qpos[0])
            if not lo - 1e-8 <= value <= hi + 1e-8:
                violations.append({"joint": name, "overrun": float(max(lo - value, value - hi))})
        for gid, box in zip(bound.geom_ids, bound.bounds_world, strict=True):
            body = int(self.envelope_model.geom_bodyid[gid])
            name = mujoco.mj_id2name(self.envelope_model, mujoco.mjtObj.mjOBJ_BODY, body)
            offset = gid - int(self.envelope_model.body_geomadr[body])
            full_gid = int(self.model.body(name).geomadr[0]) + offset
            center, half = self.model.geom_aabb[full_gid, :3], self.model.geom_aabb[full_gid, 3:]
            corners = box_corners([center - half, center + half])
            corners = corners @ probe.geom_xmat[full_gid].reshape(3, 3).T + probe.geom_xpos[full_gid]
            escape = max(escape, float(np.max(box[0] - corners)), float(np.max(corners - box[1])))
        self.envelope_samples.append(
            {
                "sim_time": float(probe.time),
                "phase": phase,
                "target_contact": touching,
                "max_corner_escape_m": escape,
                "joint_interval_violations": violations,
                "bounds_count": len(bound.geom_ids),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gripper-envelope",
        action="store_true",
        help="Also score motor-conditioned whole-gripper bounds against separate simulator geometry.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    previous = contact.ContactRobot
    rows = []
    scratch = TemporaryDirectory(prefix="emet-gripper-audit-")
    try:
        if args.gripper_envelope:
            prepare(scratch.name)
            AuditedContactRobot.envelope_model = mujoco.MjModel.from_xml_path(str(Path(scratch.name) / "stretch.xml"))
        contact.ContactRobot = AuditedContactRobot
        for obj in ("object1", "object2"):
            result = contact.run_case(model, obj)
            (args.output / f"{obj}-control.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            row = {"object": obj, "control_passed": result["passed"], "audit": AuditedContactRobot.instances[-1].audit}
            if args.gripper_envelope:
                samples = AuditedContactRobot.instances[-1].envelope_samples
                (args.output / f"{obj}-envelope.json").write_text(json.dumps(samples, indent=2) + "\n")
                row["gripper_envelope"] = {
                    "samples": len(samples),
                    "contact_samples": sum(s["target_contact"] for s in samples),
                    "max_corner_escape_m": max(s["max_corner_escape_m"] for s in samples),
                    "samples_outside_declared_intervals": sum(bool(s["joint_interval_violations"]) for s in samples),
                    "scope": "sampled containment only; not certified hardware bounds",
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        contact.ContactRobot = previous
        AuditedContactRobot.envelope_model = None
        scratch.cleanup()
    summary = {
        "scope": "privileged simulator calibration/debug only; not observed task acceptance or certified uncertainty",
        "counts": "physics samples within two development control episodes, not independent trials",
        "cases": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return (
        0
        if all(r["control_passed"] and r.get("gripper_envelope", {}).get("max_corner_escape_m", 0) < 1e-8 for r in rows)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
