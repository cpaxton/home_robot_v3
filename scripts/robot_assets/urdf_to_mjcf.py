#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Convert a URDF (usually an arm) into an MJCF ``<body>`` fragment.

Carries over the kinematic chain: joint frames/axes/limits, inertial params, and
visual meshes. The optional ``--mesh-map`` JSON maps each URDF mesh basename to
either a vendored STL (with a centroid offset to compensate STEP-part recentering)
or a ``box`` geom for off-the-shelf bodies that have no STEP export (e.g. servos).

Example mesh-map::

    {
      "Arm-Base-V3-v1.stl": {"stl": "arm_base", "offset_mm": [0, 0, 0]},
      "Feetech-Servo-Motor-v1.stl": {"box_mm": [40, 20, 36.5], "color": "servo_dark"}
    }

Run with any Python that has numpy (main venv is fine)::

    uv run python scripts/robot_assets/urdf_to_mjcf.py \
        /path/to/Arm.urdf --mesh-map /tmp/mesh_map.json --out /tmp/arm_frag.xml
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFAULT_DAMPING = 0.1
DEFAULT_FRICTIONLOSS = 0.05


def euler_to_rot(rpy: list[float]) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)

    def _r(ax: str, th: float) -> np.ndarray:
        c, s = np.cos(th), np.sin(th)
        if ax == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if ax == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return _r("z", y) @ _r("y", p) @ _r("x", r)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q


def qstr(q: np.ndarray) -> str:
    return " ".join(f"{v:.8g}" for v in q)


def vstr(v) -> str:
    return " ".join(f"{float(v):.8g}" for v in v)


def resolve_joint_transform(joint: ET.Element) -> np.ndarray:
    ori = joint.find("origin")
    if ori is None:
        return np.eye(4)
    xyz = [float(x) for x in ori.get("xyz").split()]
    rpy = [float(x) for x in ori.get("rpy").split()] if ori.get("rpy") else [0.0, 0.0, 0.0]
    T = np.eye(4)
    T[:3, :3] = euler_to_rot(rpy)
    T[:3, 3] = xyz
    return T


def mesh_map_entry(mesh_map: dict, basename: str) -> dict:
    for key, entry in mesh_map.items():
        if key == basename or key == Path(basename).name:
            return entry
    return {}


class UrdfToMjcf:
    def __init__(self, urdf: Path, mesh_map: dict | None = None, root_body: str | None = None, mass_scale: float = 1.0):
        tree = ET.parse(urdf)
        self.root = tree.getroot()
        self.links = {l.get("name"): l for l in self.root.findall("link")}
        self.children: dict[str, list] = {}
        for j in self.root.findall("joint"):
            parent = j.find("parent").get("link")
            self.children.setdefault(parent, []).append(j)
        self.mesh_map = mesh_map or {}
        self.root_body = root_body or self._find_root()
        self.mass_scale = float(mass_scale)

    def _find_root(self) -> str:
        children_links = {j.find("child").get("link") for j in self.root.findall("joint")}
        for name in self.links:
            if name not in children_links:
                return name
        raise RuntimeError("No root link (every link is a child of some joint).")

    def _link_visual_mesh(self, link: ET.Element) -> str | None:
        vis = link.find("visual")
        if vis is None:
            return None
        mesh = vis.find("geometry/mesh")
        return mesh.get("filename") if mesh is not None else None

    def emit_body(self, body_name: str, T_in: np.ndarray, indent: str = "    ") -> list[str]:
        """Emit ``<body pos= quat=>`` carrying the incoming joint transform *T_in*.

        MJCF ``joint`` elements have no ``quat``/rotation, so the URDF joint origin's
        rotation must live on the child ``<body>`` (``pos``/``quat``), and the hinge
        keeps only ``axis`` (in the body == joint frame) + ``range``. Fixed joints are
        dropped (a body with no joint element is rigidly attached).
        """
        lines: list[str] = []
        link = self.links[body_name]
        pos = T_in[:3, 3]
        q = rot_to_quat(T_in[:3, :3])
        attr = f' name="{body_name}" pos="{vstr(pos)}" quat="{qstr(q)}"'
        lines.append(f"{indent}<body{attr}>")

        inert = link.find("inertial")
        if inert is not None:
            mass = float(inert.find("mass").get("value")) * self.mass_scale
            ori = inert.find("origin")
            com = [0.0, 0.0, 0.0]
            if ori is not None:
                com = [float(x) for x in ori.get("xyz").split()]
            inr = inert.find("inertia")
            full = [float(inr.get(k)) * self.mass_scale for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            lines.append(f'{indent}  <inertial pos="{vstr(com)}" mass="{mass:.8g}" fullinertia="{vstr(full)}"/>')

        meshfile = self._link_visual_mesh(link)
        if meshfile is not None:
            entry = mesh_map_entry(self.mesh_map, meshfile)
            if "box_mm" in entry:
                # Off-the-shelf bodies with no STEP export (e.g. Feetech servos) carry the
                # revolute joints but get NO visual geom: the URDF fixed-joint placement can
                # scatter them far from the visible arm, so they'd render as floating boxes.
                pass
            elif "stl" in entry:
                if entry.get("aligned"):
                    # Pre-aligned mesh (long axis along link +y, entry joint at origin):
                    # place at the body origin with identity rotation.
                    pos = "0 0 0"
                    quat = "1 0 0 0"
                    lines.append(
                        f'{indent}  <geom type="mesh" mesh="{entry["stl"]}" pos="{pos}" quat="{quat}" '
                        f'material="plastic_white" group="2" contype="0" conaffinity="0"/>'
                    )
                    lines.append(
                        f'{indent}  <geom type="mesh" mesh="{entry["stl"]}" pos="{pos}" quat="{quat}" '
                        f'class="arm_collision"/>'
                    )
                    return lines
                vis = link.find("visual")
                ori = vis.find("origin") if vis is not None else None
                xyz = [float(x) for x in ori.get("xyz").split()] if ori is not None else [0.0, 0.0, 0.0]
                rpy = (
                    [float(x) for x in ori.get("rpy").split()]
                    if ori is not None and ori.get("rpy")
                    else [0.0, 0.0, 0.0]
                )
                offset = entry.get("offset_mm", [0.0, 0.0, 0.0])
                # The URDF visual origin places the *mesh origin* (the lerobot-vulcan STL frame)
                # at ``xyz``/``rpy``. Our STEP-derived mesh was recentered on its bbox centroid,
                # so the centroid offset must be applied in the *mesh frame* (rotated by the visual
                # orientation), not the link frame. pos = xyz + R * (centroid_m)
                pos = np.array(xyz) + euler_to_rot(rpy) @ (np.array(offset) / 1000.0)
                q = rot_to_quat(euler_to_rot(rpy))
                stl = entry["stl"]
                lines.append(
                    f'{indent}  <geom type="mesh" mesh="{stl}" pos="{vstr(pos)}" quat="{qstr(q)}" '
                    f'material="plastic_white" group="2" contype="0" conaffinity="0"/>'
                )
                # collision twin (self-collision + motion planning), see default class "arm_collision"
                lines.append(
                    f'{indent}  <geom type="mesh" mesh="{stl}" pos="{vstr(pos)}" quat="{qstr(q)}" '
                    f'class="arm_collision"/>'
                )
        return lines

    def emit_tree(self, body_name: str, T_in: np.ndarray | None = None, indent: str = "    ") -> list[str]:
        lines = self.emit_body(body_name, T_in if T_in is not None else np.eye(4), indent)
        for joint in self.children.get(body_name, []):
            child = joint.find("child").get("link")
            jtype = joint.get("type")
            T = resolve_joint_transform(joint)
            if jtype == "fixed":
                # no <joint> element; the body pos/quat already carries the transform
                lines.extend(self.emit_tree(child, T, indent + "  "))
            else:
                axis = joint.find("axis").get("xyz")
                axis = [float(x) for x in axis.split()]
                lim = joint.find("limit")
                low = float(lim.get("lower"))
                high = float(lim.get("upper"))
                lines.append(
                    f'{indent}  <joint name="{joint.get("name")}" type="hinge" axis="{vstr(axis)}" '
                    f'range="{low:.6g} {high:.6g}" damping="{DEFAULT_DAMPING}" '
                    f'frictionloss="{DEFAULT_FRICTIONLOSS}"/>'
                )
                lines.extend(self.emit_tree(child, T, indent + "  "))
        lines.append(f"{indent}</body>")
        return lines

    def build(self) -> str:
        return "\n".join(self.emit_tree(self.root_body))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urdf", type=Path, help="Input URDF file.")
    ap.add_argument("--mesh-map", type=Path, help="JSON mesh map (see docstring).")
    ap.add_argument("--root-body", type=str, help="URDF link to start emission from.")
    ap.add_argument("--mass-scale", type=float, default=1.0, help="Scale all link masses/inertias (e.g. 0.25).")
    ap.add_argument("--out", type=Path, required=True, help="Output XML fragment path.")
    args = ap.parse_args()

    mesh_map = json.loads(args.mesh_map.read_text()) if args.mesh_map else {}
    conv = UrdfToMjcf(args.urdf, mesh_map=mesh_map, root_body=args.root_body, mass_scale=args.mass_scale)
    args.out.write_text(conv.build() + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
