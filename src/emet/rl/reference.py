"""Pinned Innate URDF alignment for the RL model; not hardware calibration."""

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "assets" / "robot" / "innate_mars"
REFERENCE = ROOT / "reference" / "maurice.urdf"


def numbers(text):
    return np.fromstring(text, sep=" ")


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rotation_quat(rotation):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(rotation).ravel())
    return " ".join(map(str, q))


def mesh_bounds(name):
    raw = (ROOT / "meshes" / name).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError("expected pinned binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    vertices = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)
    return np.stack([vertices.min(axis=0), vertices.max(axis=0)]).astype(float)


def collision_boxes():
    """Conservative union of vendor boxes and visual bounds, in link frames."""
    result = {}
    for link in ET.parse(REFERENCE).getroot().findall("link"):
        collision = link.find("collision")
        if collision is None:
            continue
        center = numbers(collision.find("origin").get("xyz"))
        size = numbers(collision.find("geometry/box").get("size")) / 2
        bounds = np.stack([center - size, center + size])
        mesh = link.find("visual/geometry/mesh")
        if mesh is not None:
            visual = mesh_bounds(Path(mesh.get("filename")).name)
            bounds[0] = np.minimum(bounds[0], visual[0])
            bounds[1] = np.maximum(bounds[1], visual[1])
        # Planar base has no vertical DOF; discard the STL's 0.6 mm below-floor
        # export sliver and retain a 2 mm supporting-floor clearance.
        if link.get("name") == "base_link":
            bounds[0, 2] = 0.002
        result[link.get("name")] = {
            "type": "box",
            "pos": " ".join(map(str, bounds.mean(axis=0))),
            "size": " ".join(map(str, (bounds[1] - bounds[0]) / 2)),
        }
    return result


def align_to_reference(root):
    urdf = ET.parse(REFERENCE).getroot()
    # Remove scene-facing mesh and head-camera edits; use URDF link frames.
    for name in ("base_geom", "head_geom"):
        geom = root.find(f".//geom[@name='{name}']")
        for key in ("euler", "quat"):
            geom.attrib.pop(key, None)
    root.find(".//body[@name='head_visual']").set("pos", "0 0 0")
    # Remove artificial mass/inertia from planar coordinate helper bodies.
    for name in ("base_root", "base_x", "base_y", "head_visual"):
        body = root.find(f".//body[@name='{name}']")
        inertial = body.find("inertial")
        if inertial is not None:
            inertial.set("mass", "0.000001")
            inertial.set("diaginertia", "1e-9 1e-9 1e-9")
    for joint in urdf.findall("joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        body = root.find(f".//body[@name='{child}']")
        if body is None:
            parent_body = root.find(f".//body[@name='{parent}']")
            body = ET.SubElement(parent_body, "body", name=child)
            link = urdf.find(f"link[@name='{child}']")
            source = link.find("inertial")
            if source is not None:
                inertia = source.find("inertia")
                ET.SubElement(
                    body,
                    "inertial",
                    pos=source.find("origin").get("xyz"),
                    mass=source.find("mass").get("value"),
                    diaginertia=" ".join(inertia.get(k) for k in ("ixx", "iyy", "izz")),
                )
        origin = joint.find("origin")
        body.set("pos", origin.get("xyz"))
        body.set("quat", rotation_quat(rpy_matrix(numbers(origin.get("rpy", "0 0 0")))))
        if joint.get("type") == "fixed":
            continue
        target = body.find(f"joint[@name='{joint.get('name')}']")
        target.set("axis", joint.find("axis").get("xyz"))
        limit = joint.find("limit")
        target.set("range", f"{limit.get('lower')} {limit.get('upper')}")
        actuator = root.find(f".//actuator/position[@joint='{joint.get('name')}']")
        if actuator is not None:
            effort = float(limit.get("effort"))
            actuator.set("forcelimited", "true")
            actuator.set("forcerange", f"{-effort} {effort}")
    # The two mirrored gripper limits in the source are not mirror-consistent.
    # Restrict the driver to their feasible intersection; do not invent travel.
    driver = root.find(".//joint[@name='joint6']")
    follower = root.find(".//joint[@name='joint6M']")
    a, b = numbers(driver.get("range")), numbers(follower.get("range"))
    driver.set("range", f"{max(a[0], -b[1])} {min(a[1], -b[0])}")
    for name, optical in (("head_left", "camera_optical_frame"), ("head_right", "right_camera_optical_frame")):
        head = root.find(".//body[@name='head']")
        camera = head.find(f"camera[@name='{name}']")
        head.remove(camera)
        camera.set("pos", "0 0 0")
        camera.set("quat", "0 1 0 0")  # ROS optical +Z,+Y-down -> MuJoCo -Z,+Y-up.
        root.find(f".//body[@name='{optical}']").append(camera)
    # URDF defines no wrist optical frame. Keep its explicit adapter orientation,
    # but remove the rendering-only translation from the physical camera mount.
    root.find(".//camera[@name='camera_arm']").set("pos", "0 0 0")
    return root
