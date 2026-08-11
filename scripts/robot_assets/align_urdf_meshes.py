#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Bake URDF-joint alignment into arm-link STL meshes so consecutive links connect.

Motivation: a vendor URDF (e.g. ``lerobot-vulcan``'s ``Arm.urdf``) ships visual origins
tuned for its own STL meshes. Our STEP-derived meshes have a different local frame, so
placing them at the URDF visual origins leaves visible gaps between links. This script
instead aligns each link mesh so its long axis points along the link's entry→exit joint
direction (in the link frame) and centers it on the joint midpoint — guaranteeing
consecutive links overlap regardless of the original mesh frame.

Run with the main emet venv (trimesh + scipy)::

    uv run python scripts/robot_assets/align_urdf_meshes.py \
        --urdf /path/to/Arm.urdf --in-dir /tmp/raw_meshes_mm --out-dir /tmp/aligned_meshes \
        --links arm_shoulder=+1,arm_bicep_l=-1,arm_forearm=+1,arm_wrist=+1

``--links`` maps mesh basename -> sign (+1 or -1) of the arm direction in the link frame
(long axis aligns to ``+y`` or ``-y``). The URDF is parsed to read each link's entry/exit
revolute joints and compute the joint midpoint in the link frame.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_R(rpy: list[float]) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    return _rz(y) @ _ry(p) @ _rx(r)


def load_urdf_kinematics(urdf: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{link_name: (pos, R)}`` in the root link frame (all joints at 0)."""
    root = ET.parse(urdf).getroot()
    joints = root.findall("joint")
    children = {j.find("child").get("link"): j for j in joints}
    # find root (no incoming joint)
    all_children = set(children)
    all_links = {l.get("name") for l in root.findall("link")}
    root_name = (all_links - all_children).pop()
    trans: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def resolve(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name in trans:
            return trans[name]
        if name == root_name:
            trans[name] = (np.zeros(3), np.eye(3))
            return trans[name]
        j = children[name]
        parent = j.find("parent").get("link")
        o = j.find("origin")
        xyz = np.array([float(x) for x in o.get("xyz").split()]) if o is not None else np.zeros(3)
        rpy = [float(x) for x in o.get("rpy").split()] if o is not None and o.get("rpy") else [0.0, 0.0, 0.0]
        pp, pR = resolve(parent)
        R = rpy_to_R(rpy)
        trans[name] = (pp + pR @ xyz, pR @ R)
        return trans[name]

    for name in all_links:
        resolve(name)
    return trans


def joint_midpoint_in_link_frame(
    urdf: Path, kinematics: dict[str, tuple[np.ndarray, np.ndarray]], link_name: str, exit_joint: str
) -> float:
    """Entry joint is at the link origin; return the exit joint's y in the link frame."""
    root = ET.parse(urdf).getroot()
    rev = {j.get("name"): j for j in root.findall("joint") if j.get("type") == "revolute"}
    j = rev[exit_joint]
    o = j.find("origin")
    xyz = np.array([float(x) for x in o.get("xyz").split()])
    parent = j.find("parent").get("link")
    pp, pR = kinematics[parent]
    lp, lR = kinematics[link_name]
    w = pp + pR @ xyz
    local = lR.T @ (w - lp)
    return float(local[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--in-dir", type=Path, required=True, help="Raw (mm) meshes dir.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Aligned meshes dir (mm).")
    ap.add_argument(
        "--links",
        required=True,
        help="Comma list mesh_basename=sign (e.g. arm_shoulder=+1,arm_bicep_l=-1,arm_forearm=+1,arm_wrist=+1).",
    )
    ap.add_argument("--exit-joints", help="Comma list link_name=joint_name for the exit revolute of each link.")
    args = ap.parse_args()

    import trimesh
    from scipy.spatial.transform import Rotation as R

    kinematics = load_urdf_kinematics(args.urdf)
    links = {}
    for item in args.links.split(","):
        name, sign = item.split("=")
        links[name] = int(sign)
    exit_joints = {}
    if args.exit_joints:
        for item in args.exit_joints.split(","):
            link, joint = item.split("=")
            exit_joints[link] = joint

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # map mesh basename -> URDF link name (strips leading arm_ / trailing _l)
    for mesh_name, sign in links.items():
        link_name = mesh_name.removeprefix("arm_").removesuffix("_l")
        if link_name == "shoulder":
            link_name = "Shoulder-With-Gearbox-V3-v1"
        elif link_name == "forearm":
            link_name = "Arm-Forearm-v1"
        elif link_name == "wrist":
            link_name = "Arm-Wrist-v1"
        elif link_name == "bicep":
            link_name = "Arm-Bicep-v1"
        exit_j = exit_joints.get(mesh_name)
        mid = 0.0
        if exit_j:
            # entry joint is at the link origin (y=0); midpoint between entry and exit.
            exit_y = joint_midpoint_in_link_frame(args.urdf, kinematics, link_name, exit_j)
            mid = 0.5 * exit_y
        m = trimesh.load(str(args.in_dir / f"{mesh_name}.stl"))
        lo, hi = m.bounds
        size = hi - lo
        ax = int(np.argmax(size))
        longdir = np.zeros(3)
        longdir[ax] = 1.0
        target = np.array([0.0, float(sign), 0.0])
        q = R.align_vectors([target], [longdir])[0]
        T = np.eye(4)
        T[:3, :3] = q.as_matrix()
        m.apply_transform(T)
        yc = 0.5 * (m.bounds[0, 1] + m.bounds[1, 1])
        m.apply_translation([0.0, mid * 1000.0 - yc, 0.0])
        out = args.out_dir / f"{mesh_name}.stl"
        m.export(str(out))
        print(
            f"{mesh_name:14s} sign={sign:+d} exit={exit_j or '-':14s} center_y={0.5 * (m.bounds[0, 1] + m.bounds[1, 1]):.1f}mm -> {out}"
        )


if __name__ == "__main__":
    main()
