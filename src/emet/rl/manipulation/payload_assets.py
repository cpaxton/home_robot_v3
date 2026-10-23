"""Transfer canonical TAMP objects, preserving physical properties at source scale.

Only the standalone primitive table assets are currently admitted. This transfers
objects into a robot fixture; it does not claim to reproduce the full table task.
"""

import hashlib
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco

NATIVE_TABLE = Path(__file__).resolve().parents[2] / "assets/robot/scene_environment.xml"


def provenance(asset_set):
    if asset_set == "training":
        return {"asset_set": asset_set}
    if asset_set != "native_table":
        raise ValueError("unsupported payload asset set")
    return {
        "asset_set": asset_set,
        "source": "emet/assets/robot/scene_environment.xml",
        "source_sha256": hashlib.sha256(NATIVE_TABLE.read_bytes()).hexdigest(),
        "geometry_scale": 1.0,
    }


def native_table_bodies():
    root = ET.parse(NATIVE_TABLE).getroot()
    model = mujoco.MjModel.from_xml_path(str(NATIVE_TABLE))
    bodies = []
    for name in ("object1", "object2"):
        original = root.find(f"worldbody/body[@name='{name}']")
        if original is None or original.find("body") is not None or len(original.findall("geom")) != 1:
            raise ValueError("native payload requires one rigid primitive body")
        body = deepcopy(original)
        geom = body.find("geom")
        if geom.get("type") not in {"box", "cylinder"} or geom.get("material"):
            raise ValueError("unsupported native payload geometry or material dependency")
        bid = model.body(name).id
        gid = model.body_geomadr[bid]
        inertial = body.find("inertial")
        if inertial is not None:
            body.remove(inertial)
        ET.SubElement(
            body,
            "inertial",
            pos=" ".join(map(str, model.body_ipos[bid])),
            quat=" ".join(map(str, model.body_iquat[bid])),
            mass=str(model.body_mass[bid]),
            diaginertia=" ".join(map(str, model.body_inertia[bid])),
        )
        # Robot defaults must not silently alter source collision properties.
        for field in ("friction", "solref", "solimp", "rgba", "pos", "quat"):
            geom.set(field, " ".join(map(str, getattr(model, f"geom_{field}")[gid])))
        for field in ("contype", "conaffinity", "condim", "priority"):
            geom.set(field, str(int(getattr(model, f"geom_{field}")[gid])))
        for field in ("margin", "gap", "solmix"):
            geom.set(field, str(float(getattr(model, f"geom_{field}")[gid])))
        bodies.append(body)
    return bodies


def nominal_pinch_admission(model, data, profile, payload, depth):
    """Check this fixture's centered pinch at its declared depth, not all grasps.

    Slice the actual open box pads with the line through the nominal pinch
    center. The object must fit between their inner surfaces along that line.
    A rejection does not establish that every grasp on the object is infeasible.
    """
    import numpy as np

    points = [
        data.body(name).xpos + data.body(name).xmat.reshape(3, 3) @ [depth, 0, 0] for name in profile.finger_bodies
    ]
    origin = np.mean(points, axis=0)
    axis = points[1] - points[0]
    axis /= np.linalg.norm(axis)
    intervals = []
    for name in profile.finger_bodies:
        body = model.body(name).id
        geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == body and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        if len(geoms) != 1 or model.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_BOX:
            return {"accepted": False, "reason": "unsupported_pinch_geometry"}
        g = geoms[0]
        rotation = data.geom_xmat[g].reshape(3, 3).T
        local = rotation @ (origin - data.geom_xpos[g])
        direction = rotation @ axis
        low, high = -float("inf"), float("inf")
        for coordinate, velocity, half in zip(local, direction, model.geom_size[g], strict=True):
            if abs(velocity) < 1e-10:
                if abs(coordinate) > half:
                    return {"accepted": False, "reason": "pinch_slice_misses_pad"}
            else:
                a, b = sorted(((-half - coordinate) / velocity, (half - coordinate) / velocity))
                low, high = max(low, a), min(high, b)
        if low > high:
            return {"accepted": False, "reason": "pinch_slice_misses_pad"}
        intervals.append((low, high))
    intervals.sort()
    gap = intervals[1][0] - intervals[0][1]
    geom = model.body_geomadr[payload]
    direction = data.geom_xmat[geom].reshape(3, 3).T @ axis
    size = model.geom_size[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        width = 2 * float(abs(direction) @ size)
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        width = 2 * (size[0] * np.linalg.norm(direction[:2]) + size[1] * abs(direction[2]))
    else:
        return {"accepted": False, "reason": "unsupported_payload_geometry"}
    return {
        "accepted": bool(width <= gap),
        "reason": "ok" if width <= gap else "nominal_pinch_clearance",
        "open_gap_m": float(gap),
        "object_projected_width_m": float(width),
    }
