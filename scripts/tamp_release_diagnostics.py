# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Evaluator-only measured release snapshots; never used as runtime sensor inputs."""

import json
from pathlib import Path

import mujoco
import numpy as np
from tamp_sim_checkpoint import save_sim_checkpoint


def install_release_diagnostics(executor, robot, output):
    ex, output = executor, Path(output)
    original = ex.event
    count = 0

    def event(**row):
        nonlocal count
        if row.get("phase") == "release_plan":
            output.mkdir(parents=True, exist_ok=True)
            path = output / f"release-{count:02d}.npz"
            save_sim_checkpoint(path.with_suffix(".mjstate"), ex.model, robot.data)
            count += 1
            probe = mujoco.MjData(ex.model)
            mujoco.mj_copyData(probe, ex.model, robot.data)
            mujoco.mj_forward(ex.model, probe)
            np.savez_compressed(
                path,
                qpos=probe.qpos,
                qvel=probe.qvel,
                ctrl=probe.ctrl,
                planning_qpos=ex.data.qpos,
                time=probe.time,
                payload_transform=ex.collision.payload_transform,
                place_positions=np.array([p for p, _ in ex.place_targets]),
                place_rotations=np.array([r for _, r in ex.place_targets]),
                qfrc_bias=probe.qfrc_bias,
                qfrc_actuator=probe.qfrc_actuator,
                actuator_force=probe.actuator_force,
            )
            joints = {}
            for name in ex.joint_names:
                j = ex.model.joint(name)
                joints[name] = {
                    "measured": float(probe.qpos[j.qposadr[0]]),
                    "velocity": float(probe.qvel[j.dofadr[0]]),
                    "bias_force": float(probe.qfrc_bias[j.dofadr[0]]),
                    "actuator_force": float(probe.qfrc_actuator[j.dofadr[0]]),
                }
            row = {**row, "snapshot": path.name, "joints": joints, "source": "simulator_ground_truth_debug"}
            path.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
        original(**row)

    ex.event = event
