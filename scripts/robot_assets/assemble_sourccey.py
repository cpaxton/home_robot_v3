#!/usr/bin/env python3
"""Adapt the pinned official Mk.V MuJoCo assembly to Emet's planar interface.

The untouched upstream model remains available for wheel/traction experiments.
Run this script after sync_sourccey.py; no network or CAD tools are needed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC_ASSETS = REPO / "src/emet/assets/robot/sourccey"
OUT_XML = SRC_ASSETS / "sourccey.xml"
ROLES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Conservative startup inside the hardware URDF limits (upstream uses -2 rad elbows,
# outside the URDF's -pi/2 stop). These are simulation poses, not servo calibration.
HOME = {
    "lift": -0.0142,
    "left_shoulder_pan": -0.785,
    "right_shoulder_pan": -0.801,
    "left_shoulder_lift": 2.01,
    "right_shoulder_lift": -2.01,
    "left_elbow_flex": -1.570796,
    "right_elbow_flex": -1.570796,
    "left_wrist_flex": 0.691,
    "right_wrist_flex": -0.723,
    "left_wrist_roll": 0.0,
    "right_wrist_roll": 0.0,
    "left_gripper": -0.0872664625997,
    "right_gripper": -0.0872664625997,
}


def build() -> str:
    upstream = SRC_ASSETS / "upstream/models"
    root = ET.parse(upstream / "sourccey.xml").getroot()
    urdf = ET.parse(upstream / "source/SourcceyURDF/SourcceyMkV.urdf").getroot()
    limits = {j.get("name"): j.find("limit") for j in urdf.findall("joint") if j.find("limit") is not None}
    root.set("model", "sourccey")
    root.find("compiler").set("meshdir", "upstream/models/source/SourcceyURDF")
    # Preserve explicit CAD inertials while allowing scene objects to infer theirs.
    root.find("compiler").set("inertiafromgeom", "auto")
    root.remove(root.find("keyframe"))
    # Namespace upstream floor assets so includes can share an environment floor.
    for element in root.iter():
        for attr in ("name", "material", "texture"):
            if element.get(attr) == "floor":
                element.set(attr, "sourccey_floor")
    world = root.find("worldbody")
    base = world.find("body[@name='base_link']")
    base.remove(base.find("freejoint"))
    world.remove(base)
    # World-aligned planar joints; retain the CAD's +45 degree alignment and
    # wheel-height offset on the child base_link, not on the navigation frame.
    planar = ET.SubElement(world, "body", name="base_root")
    ET.SubElement(planar, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
    for name, kind, axis in (
        ("base_x", "slide", "1 0 0"),
        ("base_y", "slide", "0 1 0"),
        ("base_yaw", "hinge", "0 0 1"),
    ):
        ET.SubElement(planar, "joint", name=name, type=kind, axis=axis, damping="5", armature="0.5")
    planar.append(base)
    for body in base.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name.endswith("_wheel"):
                body.remove(joint)  # wheel geometry remains; Emet commands planar velocity
            elif name == "linear_actuator":
                joint.set("name", "lift")
            elif name in limits:
                joint.set("range", f"{limits[name].get('lower')} {limits[name].get('upper')}")
        # Emet's kinematic scene placement uses explicit footprint/arm clip guards.
        # Keep visual CAD meshes; omit costly convex contact duplicates.
        for geom in list(body.findall("geom")):
            if geom.get("group") == "3":
                body.remove(geom)
    actuators = root.find("actuator")
    original = {a.get("joint"): a for a in actuators}
    actuators.clear()
    for name in ("base_x", "base_y", "base_yaw"):
        ET.SubElement(actuators, "velocity", name=name + "_act", joint=name, kv="1500")
    for name in ("lift", *(f"{s}_{r}" for s in ("left", "right") for r in ROLES)):
        a = original["linear_actuator" if name == "lift" else name]
        a.set("name", name + "_act")
        a.set("joint", name)
        if name in limits:
            a.set("ctrlrange", f"{limits[name].get('lower')} {limits[name].get('upper')}")
        actuators.append(a)
    # Resolve mesh paths only for this in-memory compilation; output stays portable.
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir")
    compiler.set("meshdir", str(SRC_ASSETS / meshdir))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    compiler.set("meshdir", meshdir)
    qpos, ctrl = model.qpos0.copy(), np.zeros(model.nu)
    for name, value in HOME.items():
        qpos[model.joint(name).qposadr[0]] = value
        ctrl[model.actuator(name + "_act").id] = value
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="sourccey_home", qpos=" ".join(map(str, qpos)), ctrl=" ".join(map(str, ctrl)))
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


if __name__ == "__main__":
    OUT_XML.write_text(build())
    print(f"Wrote {OUT_XML}")
