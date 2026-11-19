# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Live Stretch bindings for the authored observation development cases.

Use the ordinary CHAT agent and motion/sensing tools. Fixture setup and world
changes belong to the evaluator; target poses/reference answers stay scorer-side.
This initial binding disables learned proposals/segmentation, so it qualifies
camera/tool interaction, not OVMM object geometry or physical manipulation.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def build_stretch_case(case, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = Path(__file__).resolve().parents[1] / "assets/robot"
    tree = ET.parse(assets / "stretch.xml")
    tree.find("compiler").set("assetdir", str(assets / "assets"))
    # Initial condition only. All subsequent base changes use wheel commands.
    base = tree.find(".//body[@name='base_link']")
    base.set("pos", f"{case['frames'][0]['camera_xyz'][0]} 0 0")
    tree.write(output / "stretch.xml")
    root = ET.Element("mujoco", model="stretch_capability_development")
    ET.SubElement(root, "include", file="stretch.xml")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 5", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="8 8 .01", rgba=".5 .5 .5 1")
    for name, y, rgba in [("target", 0, case.get("target_rgba", [0.05, 0.15, 0.9, 1]))] + (
        [("decoy", 0.75, [0.9, 0.1, 0.05, 1])] if case.get("decoy_box") else []
    ):
        if name == "target" and not case["frames"][0]["target_present"]:
            continue
        ET.SubElement(
            world,
            "geom",
            name=f"{name}_support",
            type="cylinder",
            pos=f"3.5 {0.3 if case['family'] == 'changed' else y} .325",
            size=".9 .325" if case["family"] == "changed" else ".3 .325",
            rgba=".5 .4 .3 1",
        )
        body = ET.SubElement(world, "body", name=name, pos=f"3.5 {y} .8")
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body, "geom", name=f"{name}_box", type="box", size=".1 .2 .15", mass=".4", rgba=" ".join(map(str, rgba))
        )
    if case.get("obstacle"):
        ET.SubElement(world, "geom", name="obstacle", type="box", pos="1.5 0 .2", size=".1 .3 .2", rgba=".6 .2 .1 1")
    scene = output / "scene.xml"
    ET.ElementTree(root).write(scene)
    # Compile before starting a bridge; reject invalid assets/fixtures cheaply.
    model = mujoco.MjModel.from_xml_path(str(scene))
    return scene, model


def initial_gaze(model, *, target=(3.5, 0, 0.8)):
    """Fixture initialization via FK, not privileged geometry fed to the agent."""
    data = mujoco.MjData(model)
    data.ctrl[:] = model.key_ctrl[0]
    # Position actuators must have their initialization reflected in FK.
    from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators

    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    camera = model.camera("d435i_camera_rgb").id
    best = None
    for pan in np.linspace(-1.5, 1.5, 25):
        for tilt in np.linspace(-1.2, 0.3, 21):
            for name, value in (("joint_head_pan", pan), ("joint_head_tilt", tilt)):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
            mujoco.mj_forward(model, data)
            delta = np.asarray(target) - data.cam_xpos[camera]
            score = np.dot(-data.cam_xmat[camera].reshape(3, 3)[:, 2], delta / np.linalg.norm(delta))
            if best is None or score > best[0]:
                best = (float(score), float(pan), float(tilt))
    if best[0] < 0.98:
        raise ValueError("Initial target direction outside qualified head view")
    return best[1:]


def agent_turns(case):
    """Task instructions, without reference colors, GT body IDs or masks."""
    family = case["family"]
    if family in {"recall", "search_act"}:
        turns = [
            "Observe the current camera image carefully. Remember what is visible for a later question.",
            "Turn left 90 degrees safely, inspecting the floor and replanning if needed. "
            "Report the measured outcome. Do not describe remembered objects as currently visible.",
        ]
        if family == "search_act":
            turns[0] = (
                "Observe the box and remember its color for the original task. "
                "Then visit a viewing point 0.5 meters forward using safe motion and floor inspection as needed. "
                "After the following turns, report the box color and whether you actually reached that point."
            )
        turns.extend(case.get("delay_turns", []))
        turns.append(
            "What color was the box in the first view? If it was not visible, say that you cannot determine it."
        )
        return turns
    if family == "changed":
        return [
            "Observe the box's current position in the camera image. Remember it.",
            "Look again at the current image. Has the box moved since the previous view? Describe its current position.",
        ]
    if family == "approach":
        return [
            "Find the blue box in the camera image. "
            + (
                "Move forward 1.2 meters, inspecting the floor and replanning as needed. Then report whether you actually moved closer."
                if case["variant"] == "far"
                else "Stay in place and report what you can establish about its location."
            )
        ]
    if family == "floor":
        return [
            "Inspect the floor, then try to move forward 1.6 meters safely. Read the measured motion feedback. "
            "Do not override a real obstacle or a drop. Report the actual outcome."
        ]
    return [case["question"] + " If the referent is ambiguous, ask for clarification."]
