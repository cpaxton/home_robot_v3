"""Manipulation scenes with explicit model admission and separate frozen fixtures."""

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from emet.rl.manipulation.fixtures import scene_xml
from emet.rl.model import robot_xml
from emet.robots.innate_mars.chassis import add_physical_chassis


def build_manipulation_model(
    robot, *, diagnostics=False, payload_asset_set="training", contact_cone="pyramidal", contact_impratio=1.0, two_supports=False
):
    if type(two_supports) is not bool or (two_supports and robot != "innate_mars"):
        raise ValueError("two-support fixtures require Mars")
    if not np.isfinite(contact_impratio) or not 1 <= contact_impratio <= 10:
        raise ValueError("contact impedance ratio must be in [1, 10]")
    if contact_impratio != 1 and contact_cone != "elliptic":
        raise ValueError("nondefault contact impedance ratio requires elliptic cones")
    if contact_cone not in {"pyramidal", "elliptic"}:
        raise ValueError("invalid contact cone")
    if robot == "innate_mars":
        root = add_physical_chassis(robot_xml())
        root.find("option").set("noslip_iterations", "5")
        # Experimental bounded gripper torque; the upstream 1000 Nm/rad
        # position actuator produces destructive impulses on first contact.
        grip = root.find(".//actuator/position[@name='joint6']")
        grip.set("kp", "2")
        grip.set("kv", "0.05")
        grip.set("forcerange", "-0.08 0.08")
        for name in ("joint6", "joint6M"):
            root.find(f".//joint[@name='{name}']").set("damping", "0.02")
        for name in ("link61", "link62"):
            body = root.find(f".//body[@name='{name}']")
            body.remove(body.find(f"geom[@name='rl_{name}']"))
            collision = body.find(f"geom[@name='{name}_contact']")
            collision.set("priority", "1")
            collision.set("solref", "0.006 1")
            collision.set("solimp", "0.95 0.99 0.001")
            collision.set("condim", "4")
            collision.set("friction", "1 0.005 0.0001")
        world = root.find("worldbody")
        ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1")
        table = ET.SubElement(world, "body", name="table", pos="0.3 0 0.1")
        ET.SubElement(table, "geom", type="box", size="0.15 0.15 0.1")
        if two_supports:
            destination = ET.SubElement(world, "body", name="destination", pos="5 5 0.1")
            ET.SubElement(destination, "geom", name="destination_surface", type="box", size="0.04 0.04 0.1")
        for index, shape in enumerate(("box", "cylinder"), 1):
            body = ET.SubElement(world, "body", name=f"object{index}", pos=f"{index} 0 0.3")
            ET.SubElement(body, "freejoint")
            ET.SubElement(
                body, "geom", type=shape, size="0.008 0.03" if shape == "cylinder" else "0.008 0.008 0.03", mass="0.02"
            )
    elif robot == "stretch":
        # Only the robot/contact model is shared with regressions. Training resets
        # randomize fixture positions; frozen cases cannot be selected by this API.
        root = scene_xml("left")
        for child in list(root.findall("keyframe")):
            root.remove(child)
    else:
        raise ValueError("unsupported manipulation robot")
    if payload_asset_set != "training":
        from emet.rl.manipulation.payload_assets import native_table_bodies, provenance

        provenance(payload_asset_set)  # Reject unsupported sets before compilation.
        world = root.find("worldbody")
        for body in native_table_bodies():
            old = world.find(f"body[@name='{body.get('name')}']")
            if old is None:
                raise ValueError("scene is missing a required payload slot")
            world.remove(old)
            world.append(body)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    if contact_cone != "pyramidal":
        option.set("cone", contact_cone)
    if contact_impratio != 1:
        option.set("impratio", str(float(contact_impratio)))
    # The archived Stretch MJCF requests 5000 contacts, making each 3.5 MjData
    # reserve a very large contact arena. Our admitted robot/fixture scene peaks
    # below 40 contacts and 162 scalar constraints, so cap both capacities with
    # more than 3x headroom while avoiding enormous per-MjData arenas.
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("nconmax", "256")
    size.set("njmax", "512")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    # Absolute asset lookup paths are not part of the portable model identity.
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("assetdir", "meshdir", "texturedir"):
            if compiler.get(key):
                compiler.set(key, "assets" if robot == "stretch" else "meshes")
    identity = ET.tostring(root) + b";contact_load:efc_linear_l1_bound_v2"
    return model, hashlib.sha256(identity).hexdigest()
