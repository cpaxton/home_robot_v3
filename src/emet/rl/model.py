"""Prepare Mars collision geometry without changing the canonical robot asset."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from emet.rl.profile import MARS

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "robot"
from emet.rl.reference import REFERENCE, align_to_reference, collision_boxes

# Conservative link-frame envelopes derived from pinned source geometry.
PROXIES = collision_boxes()
# A whole-base box fills empty space around the shoulder. Use the source mesh's
# convex hull for contact; retain its full XY footprint. The 3 mm Z offset avoids
# a permanent floor contact on a base that has no vertical degree of freedom.
PROXIES["base_link"] = {"type": "mesh", "mesh": "base_mesh", "pos": "0 0 0.003"}


def robot_xml():
    root = ET.parse(ASSETS / "innate_mars" / "innate_mars.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSETS / "innate_mars" / "meshes"))
    root.find("option").set("timestep", str(MARS.physics_dt))
    root.find("option").set("integrator", "implicitfast")
    for name in MARS.arm_joints:
        actuator = root.find(f".//actuator/position[@name='{name}']")
        actuator.set("kp", str(MARS.arm_kp))
        actuator.set("kv", str(MARS.arm_kv))
        actuator.set("forcerange", f"{-MARS.arm_effort_limit} {MARS.arm_effort_limit}")
        actuator.set("forcelimited", "true")
    root = align_to_reference(root)
    contact = root.find("contact")
    # Use the source SRDF exclusions instead of the prototype's invented pairs.
    for child in list(contact):
        contact.remove(child)
    srdf = ET.parse(REFERENCE.with_name("arm.srdf")).getroot()
    for pair in srdf.findall("disable_collisions"):
        ET.SubElement(contact, "exclude", body1=pair.get("link1"), body2=pair.get("link2"))
    for name, geometry in PROXIES.items():
        body = root.find(f".//body[@name='{name}']")
        ET.SubElement(
            body,
            "geom",
            name=f"rl_{name}",
            **geometry,
            contype="1",
            conaffinity="1",
            group="3",
            rgba="0.2 0.8 0.3 0.25",
            mass="0",
        )
    return root


def model_fingerprint():
    """Control-model fingerprint deliberately excludes scene layout."""
    root = robot_xml()
    root.find("compiler").set("meshdir", "meshes")
    digest = hashlib.sha256(ET.tostring(root) + json.dumps(MARS.metadata(), sort_keys=True).encode())
    digest.update(REFERENCE.read_bytes())
    digest.update(
        b"obstacle_features:oriented_world_aabbs_v1;"
        b"contact_load:efc_linear_l1_bound_v2;arena:256_contacts_512_constraints"
    )
    for path in sorted((ASSETS / "innate_mars" / "meshes").glob("*.STL")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_model(scene="procedural", obstacle_count=0):
    root = robot_xml()
    # The source model's 5000 contact/constraint slots reserve a huge arena in
    # every MjData. The proxy robot and small RL scenes stay below 40 contacts
    # and 200 scalar constraints, so retain more than 2x solver headroom.
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("nconmax", "256")
    size.set("njmax", "512")
    world = root.find("worldbody")
    if scene == "table":
        scene_root = ET.parse(ASSETS / "scene_environment.xml").getroot()
        for child in scene_root:
            existing = root.find(child.tag)
            if existing is not None and child.tag in {"worldbody", "asset"}:
                existing.extend(list(child))
            else:
                root.append(child)
    elif scene == "procedural":
        ET.SubElement(world, "geom", name="rl_floor", type="plane", size="3 3 0.05", rgba="0.8 0.8 0.8 1")
        ET.SubElement(world, "light", pos="0 0 3")
        for i in range(obstacle_count):
            body = ET.SubElement(world, "body", name=f"rl_obstacle_{i}", mocap="true", pos=f"{2 + i} 0 0.2")
            ET.SubElement(
                body,
                "geom",
                name=f"rl_obstacle_geom_{i}",
                type="box",
                size="0.07 0.07 0.20",
                rgba="0.8 0.3 0.2 1",
                contype="1",
                conaffinity="1",
            )
    else:
        raise ValueError("RL currently supports procedural and table scenes only")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
