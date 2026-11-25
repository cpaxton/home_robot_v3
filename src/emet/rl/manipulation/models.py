"""Manipulation scenes with explicit model admission and separate frozen fixtures."""

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from emet.rl.manipulation.fixtures import scene_xml
from emet.rl.model import ASSETS, robot_xml
from emet.robots.innate_mars.chassis import add_physical_chassis


def build_manipulation_model(
    robot,
    *,
    diagnostics=False,
    payload_asset_set="training",
    contact_cone="pyramidal",
    contact_impratio=1.0,
    two_supports=False,
    mesh_storage="inline",
    scene_geoms=(),
    hide_training_fixtures=False,
    forearm_collision="proxy_box",
    head_range_rad=None,
    grip_range_rad=None,
    scene_payload_half_size=None,
):
    if type(hide_training_fixtures) is not bool or (hide_training_fixtures and robot != "innate_mars"):
        raise ValueError("fixture-free scene rehearsal requires Mars")
    if forearm_collision not in ("proxy_box", "cad_hull") or (
        forearm_collision == "cad_hull" and robot != "innate_mars"
    ):
        raise ValueError("CAD forearm collision is available only for Mars")
    joint_ranges = {}
    if grip_range_rad is not None:
        joint_ranges["joint6"] = grip_range_rad
    if head_range_rad is not None:
        joint_ranges["joint_head"] = head_range_rad
    for joint, limits in joint_ranges.items():
        limits = np.asarray(limits, dtype=float)
        if robot != "innate_mars" or limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f"finite ordered Mars limits required for {joint}")
        joint_ranges[joint] = limits
    if "joint6" in joint_ranges:
        joint_ranges["joint6M"] = -joint_ranges["joint6"][::-1]
    if scene_payload_half_size is not None:
        scene_payload_half_size = np.asarray(scene_payload_half_size, dtype=float)
        if (
            not hide_training_fixtures
            or scene_payload_half_size.shape != (3,)
            or not np.isfinite(scene_payload_half_size).all()
            or np.any(scene_payload_half_size <= 0)
            or np.any(scene_payload_half_size > 0.1)
        ):
            raise ValueError("bounded payload half sizes require a captured Mars scene")
    if not isinstance(scene_geoms, (list, tuple)) or len(scene_geoms) > 2000:
        raise ValueError("bounded scene geometry required")
    for geom in scene_geoms:
        if not isinstance(geom, dict) or set(geom) != {"position", "half_size", "rgba"}:
            raise ValueError("scene geometry requires position, half_size and rgba")
        position, size, rgba = (np.asarray(geom[k], dtype=float) for k in ("position", "half_size", "rgba"))
        if (
            position.shape != (3,)
            or size.shape != (3,)
            or rgba.shape != (4,)
            or not np.isfinite(np.r_[position, size, rgba]).all()
            or np.any(abs(position) > 5)
            or np.any(size <= 0)
            or np.any(size > 1)
            or np.any(rgba < 0)
            or np.any(rgba > 1)
        ):
            raise ValueError("finite bounded scene geometry required")
    if mesh_storage not in {"inline", "binary"}:
        raise ValueError("mesh storage must be inline or binary")
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
        for joint, limits in joint_ranges.items():
            root.find(f".//joint[@name='{joint}']").set("range", " ".join(map(str, limits)))
        if forearm_collision == "cad_hull":
            # The union box fills empty space next to the folded forearm.
            # Keep the same active geom, mass and contact rules, and use the
            # source STL's convex hull as its contact surface.
            forearm = root.find(".//geom[@name='rl_link3']")
            forearm.attrib.pop("pos")
            forearm.attrib.pop("size")
            forearm.set("type", "mesh")
            forearm.set("mesh", "link3_mesh")
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
    if hide_training_fixtures:
        # Keep named bookkeeping bodies for the unloaded reach environment, but
        # do not invent a training table/payload in the reconstructed room.
        for name in ("table", "object1", "object2"):
            if name.startswith("object"):
                # Invisible, noncolliding bookkeeping bodies must not fall
                # through the ground and trigger false physics divergence.
                root.find(f".//body[@name='{name}']").set("gravcomp", "1")
            for geom in root.findall(f".//body[@name='{name}']/geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("rgba", "0 0 0 0")
    if scene_payload_half_size is not None:
        body = root.find(".//body[@name='object1']")
        body.attrib.pop("gravcomp")
        geom = body.find("geom")
        geom.attrib.update(
            name="sock_proxy",
            size=" ".join(map(str, scene_payload_half_size)),
            contype="1",
            conaffinity="1",
            rgba="0.9 0.9 0.9 1",
            friction="1.1 0.005 0.0001",
        )
    for index, geom in enumerate(scene_geoms):
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name=f"observed_scene_{index}",
            type="box",
            pos=" ".join(map(str, geom["position"])),
            size=" ".join(map(str, geom["half_size"])),
            rgba=" ".join(map(str, geom["rgba"])),
            mass="0",
            contype="1",
            conaffinity="1",
        )
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
    if mesh_storage == "binary":
        from emet.rl.manipulation.mesh_storage import binary_mesh_assets

        compiled_xml, assets = binary_mesh_assets(xml)
        model = mujoco.MjModel.from_xml_string(compiled_xml, assets)
    else:
        model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    # Absolute asset lookup paths are not part of the portable model identity.
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("assetdir", "meshdir", "texturedir"):
            if compiler.get(key):
                compiler.set(key, "assets" if robot == "stretch" else "meshes")
    identity = ET.tostring(root) + b";contact_load:efc_linear_l1_bound_v2"
    if forearm_collision == "cad_hull":
        collision_mesh = ASSETS / "innate_mars" / "meshes" / "link3.STL"
        identity += b";forearm_contact_mesh:" + hashlib.sha256(collision_mesh.read_bytes()).digest()
    return model, hashlib.sha256(identity).hexdigest()
