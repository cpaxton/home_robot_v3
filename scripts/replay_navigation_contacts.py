#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Inspect full-robot contacts at saved MuJoCo states, without running dynamics.

This is diagnostic evidence, NOT continuous collision acceptance: contacts and
forces are reconstructed by mj_forward at the trace's sampled states. Supply the
exact resolved scene, and explicitly allow only verified ground/support geoms.
Nothing from this evaluator is exposed to the learned policy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def replay(model, rows, robot_root, allowed_geoms):
    root = model.body(robot_root).id
    if root == 0:
        raise ValueError("Robot root must not be the world")
    allowed = {model.geom(name).id for name in allowed_geoms}
    robot = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in robot:
            robot.add(body)
    data = mujoco.MjData(model)
    force = np.zeros(6)
    events = []
    count = 0
    for row in rows:
        if "sim_time" not in row:
            continue
        # Missing full state is an error, never evidence of collision-free motion.
        for key in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
            value = np.asarray(row[key], dtype=float)
            target = getattr(data, key)
            if value.shape != target.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} at {row['sim_time']}")
            target[:] = value
        data.time = row["sim_time"]
        mujoco.mj_forward(model, data)
        count += 1
        for index, contact in enumerate(data.contact):
            a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
            if (a in robot) == (b in robot):
                continue
            other = int(contact.geom2 if a in robot else contact.geom1)
            if other in allowed or contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(model, data, index, force)
            if not np.isfinite(force[0]) or force[0] <= 0:
                continue
            events.append(
                {
                    "sim_time": row["sim_time"],
                    "wall_time": row.get("wall_time"),
                    "bodies": [model.body(a).name, model.body(b).name],
                    "geoms": [model.geom(int(g)).name for g in contact.geom],
                    "distance_m": float(contact.dist),
                    "reconstructed_normal_force_n": float(force[0]),
                }
            )
    return {
        "status": "contacts_found" if events else "no_contacts_at_sampled_states" if count else "missing_states",
        "continuous_safety_verified": False,
        "sample_count": count,
        "robot_root": robot_root,
        "allowed_geoms": sorted(allowed_geoms),
        "contacts": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--robot-root", required=True)
    parser.add_argument("--allow-geom", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    with args.trace.open() as stream:
        result = replay(model, (json.loads(line) for line in stream), args.robot_root, args.allow_geom)
    result["inputs"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.scene, args.trace)}
    result["caveat"] = "Sampled full-state reconstruction, not measured continuous contact evidence."
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "contacts"}))


if __name__ == "__main__":
    main()
