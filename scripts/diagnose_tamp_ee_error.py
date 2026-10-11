#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Attribute a failed TAMP "measured EE" line to joint tracking vs FK model error.

Offline and deterministic: no server, no dynamics, no GPU. Reconstructs forward
kinematics of ``planned_q`` and ``observed_q`` at the recorded base XYT and reports
(a) whether the IK plan itself reached the target (model error) and (b) which
joints' tracking lag drove the Cartesian error, using a finite-difference Jacobian
at ``planned_q``. This distinguishes a wrong IK/plan from a right plan the arm did
not reach, and names the joints whose lag matters at the end effector.

This is reconstruction, not a benchmark success claim.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import mujoco
import numpy as np

from emet.motion.arm_manip_profile import ArmManipProfile
from emet.motion.mujoco_arm_ik import joint_qpos_addrs
from emet.robots import get_robot_spec


def _attempts(path: Path) -> list[dict]:
    """Yield ``{base_xyt, measured}`` for each ``measured EE`` line in *path*."""
    base = None
    out = []
    for line in path.read_text().splitlines():
        if "TAMP execute: approach " in line:
            base = ast.literal_eval(line.split("TAMP execute: approach ")[1])["xyt"]
        elif "measured EE: " in line:
            if base is None:
                raise ValueError("measured EE line without a preceding approach")
            out.append({"base_xyt": base, "measured": json.loads(line.split("measured EE: ")[1])})
    if not out:
        raise ValueError("no measured EE lines in log")
    return out


def _finite_difference_jacobian(model, data, qadr, q0, ee_body):
    eps = 1e-4
    f0 = None
    jac = np.zeros((3, len(qadr)))

    def fk(q):
        for a, v in zip(qadr, q, strict=True):
            data.qpos[a] = float(v)
        mujoco.mj_forward(model, data)
        return data.body(ee_body).xpos.copy()

    f0 = fk(np.asarray(q0, dtype=float))
    for i in range(len(qadr)):
        qp = np.asarray(q0, dtype=float).copy()
        qp[i] += eps
        fp = fk(qp)
        jac[:, i] = (fp - f0) / eps
    return jac, f0


def _analyze(robot: str, base_xyt: list[float], measured: dict, base_z: float) -> dict:
    profile = ArmManipProfile.for_robot(robot)
    spec = get_robot_spec(robot)
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)

    joints = list(measured["joint_names"])
    qadr = joint_qpos_addrs(model, joints)

    base = model.joint(profile.base_freejoint_name) if getattr(profile, "base_freejoint_name", None) else None
    x, y, yaw = (float(v) for v in base_xyt)
    if base is not None:
        ba = int(base.qposadr[0])
        data.qpos[ba : ba + 7] = [x, y, base_z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    planned = np.asarray(measured["planned_q"], dtype=float)
    observed = np.asarray(measured["observed_q"], dtype=float)
    target = np.asarray(measured["target_xyz"], dtype=float)

    jac, fk_planned = _finite_difference_jacobian(model, data, qadr, planned, profile.ee_body)
    for a, v in zip(qadr, observed, strict=True):
        data.qpos[a] = float(v)
    mujoco.mj_forward(model, data)
    fk_observed = data.body(profile.ee_body).xpos.copy()

    q_err = observed - planned
    contrib = jac * q_err  # (3, n)
    cart = np.linalg.norm(contrib, axis=0)  # per-joint Cartesian contribution

    # Per-joint limits / near-limit flags.
    joints_info = []
    for i, name in enumerate(joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo = hi = None
        near = ""
        if jid >= 0 and model.jnt_limited[jid]:
            lo = float(model.jnt_range[jid][0])
            hi = float(model.jnt_range[jid][1])
            if abs(observed[i] - lo) < 0.02 or abs(observed[i] - hi) < 0.02:
                near = "pinned_limit"
        joints_info.append(
            {
                "name": name,
                "planned_rad": float(planned[i]),
                "observed_rad": float(observed[i]),
                "error_rad": float(q_err[i]),
                "cartesian_contribution_m": float(cart[i]),
                "lo": lo,
                "hi": hi,
                "near_limit": near,
            }
        )

    return {
        "ee_body": profile.ee_body,
        "base_xyt": [x, y, yaw],
        "base_z": base_z,
        "target_xyz": target.tolist(),
        "fk_planned_xyz": fk_planned.tolist(),
        "fk_observed_xyz": fk_observed.tolist(),
        "plan_model_error_m": float(np.linalg.norm(fk_planned - target)),
        "observed_error_m": float(np.linalg.norm(fk_observed - target)),
        "planned_to_observed_m": float(np.linalg.norm(fk_observed - fk_planned)),
        "tolerance_m": float(measured["tolerance_m"]),
        "joints": joints_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--base-z", type=float, default=0.3009)
    parser.add_argument("--failed-only", action="store_true", help="Analyze only accepted=false attempts")
    args = parser.parse_args()

    rows = []
    for attempt in _attempts(args.log):
        m = attempt["measured"]
        if args.failed_only and bool(m.get("accepted", True)):
            continue
        rows.append(_analyze(args.robot, attempt["base_xyt"], m, args.base_z))

    if not rows:
        raise SystemExit("no matching attempts (try without --failed-only)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
