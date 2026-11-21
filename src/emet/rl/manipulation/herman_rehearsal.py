"""Offline, source-bound Herman geometry audit and partial observed-room fixture.

Images/TF are observations, not calibration of physical joint limits, forces or
servo dynamics. Captured and proposed starts never share a success denominator.
"""

import base64
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEPTH_PUBLISHER_SHA256 = "9b862dccbffa958b8e9f4e5faa044719f57ace694cdefe35a5d3e09059fb978f"
DEPTH_RULE = {
    "encoding": "16SC1",
    "metres_per_unit": 0.001,
    "invalid": "nonpositive",
    "publisher_sha256": DEPTH_PUBLISHER_SHA256,
    "source": "maurice_cam/depth_estimator/publishing.cpp:134-175",
    "physical_depth_calibration_verified": False,
}


DEPTH_RULE_INNATE_080 = {
    **DEPTH_RULE,
    "encoding": "16UC1",
    "publisher_sha256": "01194d8261ee60021ed7db5dcaca1bd8e1ceebc415c9f1f8a2a9ca36eccc6fbc",
    "source": "mars_cam/depth_estimator/publishing.cpp:127-166",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_vector(value, size):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"finite vector of length {size} required")
    return a


def pose_matrix(pose):
    q = finite_vector(pose["quaternion_xyzw"], 4)
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("unit pose quaternion required")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q).as_matrix()
    t[:3, 3] = finite_vector(pose["position"], 3)
    return t


def joint_values(capture):
    arm = capture["samples"]["arm"]
    names = arm["names"]
    if len(names) != len(set(names)) or set(names) != {f"joint{i}" for i in range(1, 7)}:
        raise ValueError("complete unique arm joint identity required")
    return dict(zip(names, finite_vector(arm["position"], 6), strict=True))


def urdf_kinematics(xml, positions):
    root = ET.fromstring(xml)
    joints = root.findall("joint")
    limits = {}
    for joint in joints:
        limit = joint.find("limit")
        if limit is not None:
            low, high = float(limit.get("lower")), float(limit.get("upper"))
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError("ordered finite source joint limits required")
            limits[joint.get("name")] = [low, high]
    transforms = {"base_link": np.eye(4)}
    values = dict(positions)
    for joint in joints:
        mimic = joint.find("mimic")
        if mimic is not None and mimic.get("joint") in values:
            values[joint.get("name")] = values[mimic.get("joint")] * float(mimic.get("multiplier", 1)) + float(
                mimic.get("offset", 0)
            )
    unresolved = list(joints)
    while unresolved:
        remaining = []
        for joint in unresolved:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                remaining.append(joint)
                continue
            origin = joint.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, 3] = finite_vector(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "), 3)
                t[:3, :3] = Rotation.from_euler(
                    "xyz", finite_vector(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 3)
                ).as_matrix()
            if joint.get("type") != "fixed":
                axis = finite_vector(np.fromstring(joint.find("axis").get("xyz"), sep=" "), 3)
                if not np.isclose(np.linalg.norm(axis), 1):
                    raise ValueError("unit joint axis required")
                value = float(values.get(joint.get("name"), 0))
                if joint.get("type") not in {"revolute", "continuous"}:
                    raise ValueError("unsupported source joint")
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(axis * value).as_matrix()
            transforms[child] = transforms[parent] @ t
        if len(remaining) == len(unresolved):
            raise ValueError("source URDF is disconnected from base_link")
        unresolved = remaining
    return transforms, limits


def limit_violations(values, limits):
    rows = []
    for name, value in values.items():
        if name not in limits:
            raise ValueError("missing source joint limit: " + name)
        low, high = limits[name]
        if not low <= value <= high:
            rows.append(
                {
                    "joint": name,
                    "reported_rad": float(value),
                    "limits_rad": [low, high],
                    "outside_rad": float(max(low - value, value - high)),
                    "outside_degrees": float(np.rad2deg(max(low - value, value - high))),
                }
            )
    return rows


def transform_error(a, b):
    return {
        "position_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_rad": float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()),
    }


def decode_array(encoded, dtype, shape):
    if any(type(x) is not int or x <= 0 for x in shape) or math.prod(shape) > 4_194_304:
        raise ValueError("bounded positive array dimensions required")
    dtype = np.dtype(dtype)
    expected = math.prod(shape) * dtype.itemsize
    raw = base64.b64decode(encoded, validate=True)
    decoder = zlib.decompressobj()
    data = decoder.decompress(raw, expected + 1)
    if len(data) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("compressed array size mismatch")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def decode_depth(sample, rule):
    encoding = sample.get("encoding")
    if rule == DEPTH_RULE and encoding == "16SC1":
        integer_type = "i2"
    elif (
        rule == DEPTH_RULE_INNATE_080
        and encoding == "16UC1"
        and sample.get("publisher_sha256") == rule["publisher_sha256"]
    ):
        integer_type = "u2"
    else:
        raise ValueError("depth requires the pinned publisher millimetre rule")
    width, height, step = sample["width"], sample["height"], sample["step"]
    if type(step) is not int or step < width * 2 or step % 2:
        raise ValueError("depth row stride mismatch")
    dtype = (">" if sample["bigendian"] else "<") + integer_type
    raw = decode_array(sample["data_zlib_base64"], dtype, (height, step // 2))[:, :width]
    depth = raw.astype(float) * rule["metres_per_unit"]
    depth[raw <= 0] = np.nan
    return depth


def rectified_intrinsics(info, width, height):
    roi = info.get("roi", {})
    if (
        any(roi.get(k, 0) for k in ("x_offset", "y_offset"))
        or info.get("binning_x", 0) not in (0, 1)
        or info.get("binning_y", 0) not in (0, 1)
    ):
        raise ValueError("cropped/binned calibration requires an explicit adapter")
    if info["width"] <= 0 or info["height"] <= 0:
        raise ValueError("calibration resolution required")
    p = np.asarray(info["P"], float).reshape(3, 4)
    r = np.asarray(info["R"], float).reshape(3, 3)
    if (
        not np.isfinite(p).all()
        or not np.isfinite(r).all()
        or p[0, 0] <= 0
        or p[1, 1] <= 0
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(r), 1)
    ):
        raise ValueError("valid rectified calibration required")
    p = p.copy()
    p[0] *= width / info["width"]
    p[1] *= height / info["height"]
    if not np.allclose(p[:, 3], 0, atol=1e-8):
        raise ValueError("depth projection must use left camera intrinsics")
    return p[:, :3], r


def project_depth(capture, rule=None, *, stride=4):
    sample = capture["samples"]["depth"]
    if rule is None:
        rule = DEPTH_RULE_INNATE_080 if sample.get("encoding") == "16UC1" else DEPTH_RULE
    if sample["stamp_ns"] <= 0 or abs(sample["stamp_ns"] - capture["samples"]["arm"]["stamp_ns"]) > 50_000_000:
        raise ValueError("depth/arm pairing exceeds 50 ms")
    info = capture["samples"]["head_left_info"]
    tf = capture["image_transforms"]["depth"]
    if tf.get("child") != sample["frame_id"] or info["frame_id"] != sample["frame_id"] or "error" in tf:
        raise ValueError("matching depth calibration/optical TF required")
    depth = decode_depth(sample, rule)
    k, rectification = rectified_intrinsics(info, sample["width"], sample["height"])
    v, u = np.mgrid[0 : sample["height"] : stride, 0 : sample["width"] : stride]
    z = depth[::stride, ::stride]
    good = np.isfinite(z) & (z > 0.15) & (z <= 3)
    xyz = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)[good]
    transform = pose_matrix(tf)
    # R maps raw optical -> rectified. Undo it before applying raw optical TF.
    points = (xyz @ rectification) @ transform[:3, :3].T + transform[:3, 3]
    return (
        points,
        np.column_stack([u[good], v[good]]),
        {
            "valid_depth_fraction": float(np.isfinite(depth).mean()),
            "sampled_points": len(points),
            "depth_rule": rule,
            "stride_pixels": stride,
            "unknown_pixels_are_free": False,
            "calibration": "scaled_rectified_P_with_inverse_R_and_capture_time_TF",
            "pair_skew_s": abs(sample["stamp_ns"] - capture["samples"]["arm"]["stamp_ns"]) / 1e9,
        },
    )


def scene_voxels(points, colors=None, *, voxel_m=0.06):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite measured 3D points required")
    if not 0.02 <= voxel_m <= 0.1:
        raise ValueError("bounded voxel size required")
    colors = np.broadcast_to([0.3, 0.65, 0.8], points.shape) if colors is None else np.asarray(colors, float)
    if colors.shape != points.shape or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1):
        raise ValueError("matched RGB colors in [0,1] required")
    radius = np.linalg.norm(points[:, :2], axis=1)
    keep = (radius >= 0.25) & (radius <= 2.5) & (points[:, 2] >= 0.04) & (points[:, 2] <= 1.5)
    points, colors = points[keep], colors[keep]
    bins = np.floor(points / voxel_m).astype(int)
    unique, indices = np.unique(bins, axis=0, return_index=True)
    if len(unique) > 1800:
        raise ValueError("observed scene exceeds bounded voxel budget; enlarge voxel size explicitly")
    return [
        {
            "position": ((key + 0.5) * voxel_m + [0, 0, 0.003]).tolist(),
            "half_size": [voxel_m / 2] * 3,
            "rgba": [*colors[index].tolist(), 1.0],
        }
        for key, index in zip(unique, indices, strict=True)
    ]


def capture_audit(capture, description):
    values = joint_values(capture)
    head = capture["transforms"].get("head", {})
    if "error" in head:
        raise ValueError("head TF required")
    qhead = float(-Rotation.from_matrix(pose_matrix(head)[:3, :3]).as_rotvec()[1])
    transforms, limits = urdf_kinematics(description, {**values, "joint_head": qhead})
    errors = {}
    for name in ("ee_link", "head", "camera_optical_frame", "right_camera_optical_frame"):
        measured = capture["transforms"].get(name, {})
        if name not in transforms or "error" in measured or not measured:
            errors[name] = {"available": False}
            continue
        errors[name] = {"available": True, **transform_error(transforms[name], pose_matrix(measured))}
    violations = limit_violations({**values, "joint_head": qhead}, limits)
    blockers = ["published_joint_limit:" + row["joint"] for row in violations]
    for name, error in errors.items():
        if not error["available"] or error["position_m"] > 0.005 or error["rotation_rad"] > 0.02:
            blockers.append("kinematic_mismatch:" + name)
    history = capture.get("arm_position_history", [])
    positions = np.array(
        [[dict(zip(row["names"], row["position"], strict=True))[name] for name in values] for row in history]
    )
    return {
        "format": "herman-rehearsal-audit-v1",
        "captured_arm_rad": values,
        "head_rad": qhead,
        "published_joint_limits": limits,
        "limit_violations": violations,
        "live_urdf_vs_tf": errors,
        "position_span_rad": np.ptp(positions, axis=0).tolist() if len(positions) else None,
        "captured_physics_admitted": not blockers,
        "physics_blockers": blockers,
        "velocity_initialization": "zero-motion simulator hypothesis; published velocity units are unverified",
        "wrist_image_available": capture["counts"].get("wrist", 0) > 0,
        "wrist_optical_tf_available": "error"
        not in capture["transforms"].get("arm_camera_optical_frame", {"error": "missing"}),
        "hardware_ready": False,
        "hardware_motion_commands_sent": 0,
    }


def load_capture(directory):
    directory = Path(directory).resolve()
    path = directory / "capture.json"
    report = json.loads(path.read_text())
    if report.get("format") != "herman-rehearsal-capture-v1" or report.get("actuator_publishers_created") is not False:
        raise ValueError("passive versioned capture required")
    asset = report.get("robot_description")
    if not asset or not re.fullmatch(r"[a-f0-9]{64}", asset["sha256"]):
        raise ValueError("source-bound robot description required")

    def resolve(record):
        target = (directory / record["path"]).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or sha(target) != record["sha256"]:
            raise ValueError("capture asset checksum/path mismatch")
        return target

    description = resolve(asset).read_text()
    for sample in report["samples"].values():
        if isinstance(sample, dict) and "image_asset" in sample:
            resolve(sample["image_asset"])
    return report, description


def seal_artifacts(directory):
    """Local immutable replay inventory; no home imagery enters git evidence."""
    directory = Path(directory)
    record = {
        "format": "herman-rehearsal-artifacts-v1",
        "files": {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "artifacts.json"},
    }
    (directory / "artifacts.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_artifacts(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory / "artifacts.json").read_text())
    if record.get("format") != "herman-rehearsal-artifacts-v1" or not isinstance(record.get("files"), dict):
        raise ValueError("versioned replay inventory required")
    for name, expected in record["files"].items():
        target = directory / name
        if Path(name).name != name or target.is_symlink() or not target.is_file() or sha(target) != expected:
            raise ValueError("replay checksum/path mismatch: " + name)
    if not {"report.json", "scene.mjb", "captured-replay.npz", "head_left.jpg"} <= record["files"].keys():
        raise ValueError("incomplete replay inventory")
    return record
