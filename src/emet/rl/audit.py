"""Independent URDF/MuJoCo kinematic and limit audit. Run: python -m emet.rl.audit."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from emet.rl.model import build_model, model_fingerprint
from emet.rl.reference import REFERENCE, ROOT, collision_boxes, mesh_bounds, numbers, rpy_matrix


def urdf_fk(joints, values):
    transforms = {"base_link": np.eye(4)}
    for joint in joints:
        origin = joint.find("origin")
        transform = np.eye(4)
        transform[:3, :3] = rpy_matrix(numbers(origin.get("rpy", "0 0 0")))
        transform[:3, 3] = numbers(origin.get("xyz"))
        if joint.get("type") != "fixed":
            axis = numbers(joint.find("axis").get("xyz"))
            axis /= np.linalg.norm(axis)
            x, y, z = axis
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            angle = values[joint.get("name")]
            rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
            transform[:3, :3] = transform[:3, :3] @ rotation
        transforms[joint.find("child").get("link")] = transforms[joint.find("parent").get("link")] @ transform
    return transforms


def audit_model(model, samples=50):
    source = ET.parse(REFERENCE).getroot()
    joints = source.findall("joint")
    mismatches, missing, inertial_mismatches, effort_mismatches = [], [], [], []
    for link in source.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        body = model.body(link.get("name"))
        origin, inertia = inertial.find("origin"), inertial.find("inertia")
        ixx, iyy, izz, ixy, ixz, iyz = (float(inertia.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        rotation = rpy_matrix(numbers(origin.get("rpy", "0 0 0")))
        expected = rotation @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ rotation.T
        actual_rotation = np.empty(9)
        mujoco.mju_quat2Mat(actual_rotation, body.iquat)
        actual_rotation = actual_rotation.reshape(3, 3)
        actual = actual_rotation @ np.diag(body.inertia) @ actual_rotation.T
        if not (
            np.allclose(body.mass, float(inertial.find("mass").get("value")), atol=1e-10, rtol=0)
            and np.allclose(body.ipos, numbers(origin.get("xyz")), atol=1e-10, rtol=0)
            and np.allclose(actual, expected, atol=1e-10, rtol=0)
        ):
            inertial_mismatches.append(link.get("name"))
    for joint in joints:
        if joint.get("type") == "fixed":
            continue
        name = joint.get("name")
        ident = model.joint(name).id
        source_range = np.array([float(joint.find("limit").get(k)) for k in ("lower", "upper")])
        actual = model.jnt_range[ident]
        # The driver is deliberately narrowed to satisfy both mimic limits.
        expected = source_range
        if name == "joint6":
            follower = source.find("joint[@name='joint6M']/limit")
            expected = np.array(
                [
                    max(source_range[0], -float(follower.get("upper"))),
                    min(source_range[1], -float(follower.get("lower"))),
                ]
            )
        if not np.allclose(actual, expected, atol=1e-7):
            mismatches.append({"joint": name, "source": expected.tolist(), "simulation": actual.tolist()})
        for actuator in np.flatnonzero(model.actuator_trnid[:, 0] == ident):
            effort = float(joint.find("limit").get("effort"))
            if not model.actuator_forcelimited[actuator] or not np.allclose(
                model.actuator_forcerange[actuator], [-effort, effort]
            ):
                effort_mismatches.append(name)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1968)
    errors = {}
    for _ in range(samples):
        values = {}
        for joint in joints:
            if joint.get("type") == "fixed":
                continue
            name = joint.get("name")
            value = rng.uniform(*model.jnt_range[model.joint(name).id])
            if name == "joint6M":
                value = -values["joint6"]
            values[name] = value
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        reference = urdf_fk(joints, values)
        base = model.body("base_link").id
        base_rotation = data.xmat[base].reshape(3, 3)
        for name, expected in reference.items():
            try:
                body = model.body(name).id
            except KeyError:
                if name not in missing:
                    missing.append(name)
                continue
            position = base_rotation.T @ (data.xpos[body] - data.xpos[base])
            rotation = base_rotation.T @ data.xmat[body].reshape(3, 3)
            errors[name] = max(
                errors.get(name, 0),
                float(np.linalg.norm(position - expected[:3, 3])),
                float(np.max(np.abs(rotation - expected[:3, :3]))),
            )
    boxes = collision_boxes()
    coverage = {}
    for name, geometry in boxes.items():
        try:
            g = model.geom("rl_" + name).id
        except KeyError:
            coverage[name] = False
            continue
        if name == "base_link" and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            source_bounds = mesh_bounds("base.STL")
            coverage[name] = bool(
                np.all(world.min(axis=0)[:2] <= source_bounds[0, :2] + 1e-6)
                and np.all(world.max(axis=0)[:2] >= source_bounds[1, :2] - 1e-6)
            )
            continue
        expected_center, expected_size = numbers(geometry["pos"]), numbers(geometry["size"])
        coverage[name] = bool(
            model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and np.all(model.geom_pos[g] - model.geom_size[g] <= expected_center - expected_size + 1e-8)
            and np.all(model.geom_pos[g] + model.geom_size[g] >= expected_center + expected_size - 1e-8)
        )
    provenance = json.loads((ROOT / "reference" / "provenance.json").read_text())
    hashes_match = all(
        hashlib.sha256((ROOT / "reference" / name).read_bytes()).hexdigest() == sha
        for name, sha in provenance["files"].items()
    )
    meshes_match = all(
        hashlib.sha256((ROOT / "meshes" / name).read_bytes()).hexdigest() == sha
        for name, sha in provenance["meshes"].items()
    )
    source_mass = sum(float(e.get("value")) for e in source.findall("link/inertial/mass"))
    passed = (
        hashes_match
        and meshes_match
        and not (mismatches or missing or inertial_mismatches or effort_mismatches)
        and max(errors.values(), default=float("inf")) < 1e-7
        and all(coverage.values())
        and abs(float(model.body_mass.sum()) - source_mass) < 1e-5
    )
    return {
        "reference_audit_passed": bool(passed),
        "reference_hashes_match": hashes_match,
        "mesh_hashes_match": meshes_match,
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "reference_revision": json.loads((ROOT / "reference" / "provenance.json").read_text())["revision"],
        "samples": samples,
        "joint_limit_mismatches": mismatches,
        "inertial_mismatches": inertial_mismatches,
        "actuator_effort_mismatches": effort_mismatches,
        "missing_link_frames": missing,
        "max_fk_errors": errors,
        "collision_coverage": coverage,
        "total_mass_kg": float(model.body_mass.sum()),
        "source_total_mass_kg": source_mass,
        "hardware_dynamics_verified": False,
        "limitations": [
            "Planar velocity-actuated base, no wheels or tire contact model",
            "URDF inertias/effort ratings not independently measured",
            "Arm/head boxes enclose meshes; base uses source-mesh convex hull, not exact concave contacts",
            "Planar base contact mesh raised 3 mm to avoid permanent floor contact",
            "Wrist optical frame absent from source URDF; optical adapter unverified",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_model(build_model())
    report["model_fingerprint"] = model_fingerprint()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["reference_audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
