"""Manipulation scenes with explicit model admission and separate frozen fixtures."""

import hashlib
import xml.etree.ElementTree as ET

import mujoco

from emet.rl.manipulation.fixtures import scene_xml
from emet.rl.manipulation.profiles import ModelAdmissionError
from emet.rl.model import robot_xml


def build_manipulation_model(robot, *, diagnostics=False):
    if robot == "innate_mars":
        if not diagnostics:
            raise ModelAdmissionError(
                "Mars wheel/caster geometry and support locations are missing; wheel-lift gate unsupported"
            )
        root = robot_xml()
        # Diagnostic-only free chassis: do not invent wheel locations from the base hull.
        world = root.find("worldbody")
        base = root.find(".//body[@name='base_link']")
        for parent in root.iter("body"):
            if base in list(parent):
                parent.remove(base)
                break
        world.remove(world.find("body[@name='base_root']"))
        world.append(base)
        ET.SubElement(base, "freejoint", name="manipulation_base")
        for item in list(root.find("contact")):
            if any(item.get(k) in {"base_root", "base_x", "base_y"} for k in ("body1", "body2")):
                root.find("contact").remove(item)
        for item in list(root.find("actuator")):
            if item.get("joint") in {"base_x", "base_y", "base_yaw"}:
                root.find("actuator").remove(item)
        for item in list(root.findall("keyframe")):
            root.remove(item)
        ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1")
        table = ET.SubElement(world, "body", name="table", pos="0.3 0 0.1")
        ET.SubElement(table, "geom", type="box", size="0.15 0.15 0.1")
        for index, shape in enumerate(("box", "cylinder"), 1):
            body = ET.SubElement(world, "body", name=f"object{index}", pos=f"{index} 0 0.3")
            ET.SubElement(body, "freejoint")
            ET.SubElement(body, "geom", type=shape, size="0.015 0.02 0.02", mass="0.02")
    elif robot == "stretch":
        # Only the robot/contact model is shared with regressions. Training resets
        # randomize fixture positions; frozen cases cannot be selected by this API.
        root = scene_xml("left")
        for child in list(root.findall("keyframe")):
            root.remove(child)
    else:
        raise ValueError("unsupported manipulation robot")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    # Absolute asset lookup paths are not part of the portable model identity.
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("assetdir", "meshdir", "texturedir"):
            if compiler.get(key):
                compiler.set(key, "assets" if robot == "stretch" else "meshes")
    return model, hashlib.sha256(ET.tostring(root)).hexdigest()
