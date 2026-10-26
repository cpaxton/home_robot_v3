import numpy as np
import pytest

pytest.importorskip("mujoco")
from emet.rl.audit import audit_model
from emet.rl.model import build_model
from emet.rl.reference import ROOT, collision_boxes, mesh_bounds

pytestmark = pytest.mark.sim


def test_reference_kinematics_limits_mass_and_collision_coverage():
    report = audit_model(build_model(), samples=30)
    assert report["reference_audit_passed"]
    assert report["reference_hashes_match"] and report["mesh_hashes_match"]
    assert not report["joint_limit_mismatches"]
    assert not report["inertial_mismatches"]
    assert not report["actuator_effort_mismatches"]
    assert not report["missing_link_frames"]
    assert max(report["max_fk_errors"].values()) < 1e-7
    assert all(report["collision_coverage"].values())
    assert abs(report["total_mass_kg"] - report["source_total_mass_kg"]) < 1e-5
    assert report["hardware_dynamics_verified"] is False


def test_audit_rejects_inertia_and_effort_drift():
    model = build_model()
    model.body_inertia[model.body("link1").id] *= 2
    actuator = np.flatnonzero(model.actuator_trnid[:, 0] == model.joint("joint1").id)[0]
    model.actuator_forcerange[actuator] *= 2
    report = audit_model(model, samples=2)
    assert not report["reference_audit_passed"]
    assert report["inertial_mismatches"] == ["link1"]
    assert report["actuator_effort_mismatches"] == ["joint1"]


def test_base_envelope_covers_source_footprint_and_model_hash_covers_meshes(tmp_path, monkeypatch):
    import shutil

    from emet.rl import model

    bounds = mesh_bounds("base.STL")
    box = collision_boxes()["base_link"]
    center = np.fromstring(box["pos"], sep=" ")
    half = np.fromstring(box["size"], sep=" ")
    assert np.all(center[:2] - half[:2] <= bounds[0, :2])
    assert np.all(center[:2] + half[:2] >= bounds[1, :2])
    original = model.model_fingerprint()
    shutil.copytree(ROOT, tmp_path / "innate_mars")
    monkeypatch.setattr(model, "ASSETS", tmp_path)
    assert model.model_fingerprint() == original  # Portable between checkout paths.
    mesh = tmp_path / "innate_mars" / "meshes" / "base.STL"
    content = bytearray(mesh.read_bytes())
    content[0] ^= 1  # Header-only change still changes asset provenance.
    mesh.write_bytes(content)
    assert model.model_fingerprint() != original
