#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Offline evaluator-only contact/actuation audit of immutable release snapshots."""

import argparse
import faulthandler
import json
import resource
from pathlib import Path

import mujoco
import numpy as np

from emet.motion.mujoco_collision import body_subtree
from emet.motion.release import body_bounds


def main():
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    parser.add_argument("--object", default="object2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str((args.fixture / "scene.xml").resolve()))
    rows = []
    target = body_subtree(model, args.object)
    arm = body_subtree(model, "link_lift")
    for folder in args.states:
        for path in sorted(folder.glob("*.npz")):
            saved = np.load(path, allow_pickle=False)
            data = mujoco.MjData(model)
            data.qpos[:], data.qvel[:], data.ctrl[:] = saved["qpos"], saved["qvel"], saved["ctrl"]
            mujoco.mj_forward(model, data)
            contacts = []
            for contact in data.contact:
                a, b = map(int, model.geom_bodyid[contact.geom])
                if not {a, b} & (target | arm):
                    continue
                contacts.append(
                    {
                        "bodies": [model.body(a).name, model.body(b).name],
                        "geoms": list(map(int, contact.geom)),
                        "distance_m": float(contact.dist),
                    }
                )
            j = model.joint("joint_lift")
            q = int(j.qposadr[0])
            v = int(j.dofadr[0])
            act = model.actuator("lift")
            bounds = body_bounds(model, data, target)
            rows.append(
                {
                    "snapshot": str(path),
                    "ee": data.body("link_grasp_center").xpos.tolist(),
                    "payload_bounds": bounds.tolist(),
                    "gap_to_table_m": float(bounds[0, 2] - 0.48),
                    "payload_transform": saved["payload_transform"].tolist(),
                    "lift": {
                        "position": float(data.qpos[q]),
                        "command": float(data.ctrl[act.id]),
                        "velocity": float(data.qvel[v]),
                        "acceleration": float(data.qacc[v]),
                        "bias": float(data.qfrc_bias[v]),
                        "passive": float(data.qfrc_passive[v]),
                        "actuator": float(data.qfrc_actuator[v]),
                        "constraint": float(data.qfrc_constraint[v]),
                        "force_range": model.actuator_forcerange[act.id].tolist(),
                        "gain": model.actuator_gainprm[act.id].tolist(),
                        "bias_parameters": model.actuator_biasprm[act.id].tolist(),
                    },
                    "contacts": contacts,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": __doc__, "snapshots": rows}, indent=2) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
