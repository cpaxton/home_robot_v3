# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Opt-in, evaluator-only contact evidence from every operational physics step.

Never expose this ground-truth stream to policy observations or use it to steer
the robot. Explicit body/ground-geom pairs permit normal support contacts only.
"""

import json
import os
import time
from pathlib import Path

import numpy as np


class NavigationSafetyTrace:
    def __init__(self, model, config, output):
        self.root = model.body(config["robot_body"]).id
        if self.root == 0:
            raise ValueError("Robot root cannot be the world")
        self.robot = {self.root}
        for body in range(self.root + 1, model.nbody):
            if int(model.body_parentid[body]) in self.robot:
                self.robot.add(body)
        self.allowed = set()
        for body, geom in config.get("allowed_support_contacts", []):
            bid, gid = model.body(body).id, model.geom(geom).id
            if bid not in self.robot or int(model.geom_bodyid[gid]) in self.robot:
                raise ValueError("Support exception must pair a robot body with an external ground geom")
            self.allowed.add((bid, gid))
        self.force = np.zeros(6)
        self.tick = 0
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = output.open("x", buffering=1)
        self.stream.write(
            json.dumps(
                {
                    "schema": 1,
                    "config": config,
                    "timestep": float(model.opt.timestep),
                    "actuator_names": [model.actuator(i).name for i in range(model.nu)],
                }
            )
            + "\n"
        )

    def record(self, model, data):
        import mujoco

        contacts = []
        for index, contact in enumerate(data.contact):
            a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
            if a not in self.robot and b not in self.robot:
                continue
            if (a, int(contact.geom2)) in self.allowed or (b, int(contact.geom1)) in self.allowed:
                continue
            if contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(model, data, index, self.force)
            if not np.isfinite(self.force[0]):
                raise ValueError("Nonfinite contact force in navigation trace")
            if self.force[0] <= 0:
                continue
            contacts.append(
                {"bodies": [model.body(a).name, model.body(b).name], "normal_force_n": float(self.force[0])}
            )
        self.tick += 1
        self.stream.write(
            json.dumps(
                {
                    "tick": self.tick,
                    "sim_time": float(data.time),
                    "wall_time": time.time(),
                    "base_xyz": data.xpos[self.root].tolist(),
                    "base_up_dot_world_z": float(data.xmat[self.root, 8]),
                    "actuator_targets": data.ctrl.tolist(),
                    "unexpected_contacts": contacts,
                },
                allow_nan=False,
            )
            + "\n"
        )

    def close(self):
        self.stream.close()


def from_environment(model):
    config = os.environ.get("EMET_NAVIGATION_TRACE_CONFIG")
    output = os.environ.get("EMET_NAVIGATION_TRACE")
    if not config and not output:
        return None
    if not config or not output:
        raise ValueError("Navigation trace requires both config and output path")
    return NavigationSafetyTrace(model, json.loads(Path(config).read_text()), output)


def score_window(path, start_wall_time, end_wall_time):
    """Require complete physics-step coverage bracketing the measured command window."""
    with Path(path).open() as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    incomplete = {"status": "incomplete_telemetry"}
    if not rows or not np.isfinite([start_wall_time, end_wall_time]).all() or end_wall_time <= start_wall_time:
        return {**incomplete, "reason": "invalid_window"}
    wall = np.asarray([r["wall_time"] for r in rows])
    if not np.isfinite(wall).all() or np.any(np.diff(wall) <= 0):
        return {**incomplete, "reason": "nonmonotonic_wall_time"}
    if wall[0] > start_wall_time or wall[-1] < end_wall_time:
        return {**incomplete, "reason": "uncovered_command_window"}
    lo = max(0, int(np.searchsorted(wall, start_wall_time, side="right")) - 1)
    hi = int(np.searchsorted(wall, end_wall_time, side="left")) + 1
    rows = rows[lo:hi]
    ticks = np.asarray([r["tick"] for r in rows])
    stamps = np.asarray([r["sim_time"] for r in rows])
    up = np.asarray([r["base_up_dot_world_z"] for r in rows])
    targets = np.asarray([r["actuator_targets"] for r in rows])
    step = header["timestep"]
    if (
        not np.isfinite(step)
        or step <= 0
        or not np.isfinite(np.r_[stamps, up]).all()
        or targets.shape != (len(rows), len(header["actuator_names"]))
        or not np.isfinite(targets).all()
    ):
        return {**incomplete, "reason": "invalid_physics_state"}
    if np.any(np.diff(ticks) != 1) or not np.allclose(np.diff(stamps), step, rtol=1e-5, atol=1e-8):
        return {**incomplete, "reason": "missing_physics_steps"}
    contact_rows = [r for r in rows if r["unexpected_contacts"]]
    return {
        "status": "failed" if contact_rows or np.min(up) < 0.57 else "physics_contact_window_clear",
        "physics_steps": len(rows),
        "min_base_up_dot_world_z": float(np.min(up)),
        "unexpected_contact_steps": len(contact_rows),
        "first_contact": contact_rows[0] if contact_rows else None,
        "scope": "declared operational command window; not startup or unsampled geometry",
    }
