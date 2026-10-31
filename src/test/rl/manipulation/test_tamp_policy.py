"""Scene transfer must preserve trained robot identity and weight provenance."""

import gzip
import json
from pathlib import Path

import pytest

from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.tamp_policy import robot_identity, sha256, verify_checkpoint

pytestmark = pytest.mark.sim


@pytest.fixture
def checkpoint(tmp_path):
    baseline, fingerprint = build_manipulation_model("innate_mars", contact_cone="elliptic", contact_impratio=3)
    target, _ = build_manipulation_model("innate_mars", contact_cone="elliptic", contact_impratio=3, two_supports=True)
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    weights = b"test weights; verify_checkpoint does not deserialize models"
    (directory / "policy.zip").write_bytes(weights)
    manifest = {
        "robot": "innate_mars",
        "action_mode": "direct_velocity",
        "fingerprint": fingerprint,
        "weights_sha256": sha256(weights),
        "environment": {
            "diagnostics": False,
            "payload_asset_set": "training",
            "contact_cone": "elliptic",
            "contact_impratio": 3,
            "two_supports": False,
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    report = {"run": {"source_hashes": source_hashes()}, "checkpoints": {"test": {"manifest": manifest}}}
    raw = json.dumps(report).encode()
    archive = tmp_path / "report.json.gz"
    archive.write_bytes(gzip.compress(raw))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"full_report_archive": str(archive), "uncompressed_sha256": sha256(raw)}))
    return directory, summary, baseline, target


def test_new_support_and_primitive_payload_changes_preserve_robot(checkpoint):
    directory, evidence, baseline, target = checkpoint
    target.body_pos[target.body("destination").id] = [0.2, 0.3, 0.1]
    target.body_mass[target.body("object1").id] = 0.1
    assert robot_identity(baseline) == robot_identity(target)
    manifest, admission = verify_checkpoint(directory, target, evidence)
    assert admission["weights_sha256"] == manifest["weights_sha256"]
    assert admission["simulation_only"] and not admission["safety_promoted"]


@pytest.mark.parametrize("field", ["mass", "geometry", "mesh", "gear", "gravity", "solver", "coupling"])
def test_robot_or_physics_changes_are_rejected(checkpoint, field):
    directory, evidence, _, model = checkpoint
    if field == "mass":
        model.body_mass[model.body("base_link").id] += 0.1
    elif field == "geometry":
        model.geom_size[model.geom("link61_contact").id, 0] += 0.001
    elif field == "mesh":
        model.mesh_vert[0, 0] += 0.001
    elif field == "gear":
        model.actuator_gear[0, 0] *= 2
    elif field == "gravity":
        model.opt.gravity[2] = -8
    elif field == "solver":
        model.opt.impratio = 1
    else:
        model.eq_data[0, 0] += 0.1
    with pytest.raises(ValueError, match="compiled robot or physics mismatch"):
        verify_checkpoint(directory, model, evidence)


@pytest.mark.parametrize("change", ["weights", "manifest", "evidence", "source"])
def test_unbound_or_changed_checkpoints_are_rejected(checkpoint, change):
    directory, evidence, _, model = checkpoint
    if change == "weights":
        (directory / "policy.zip").write_bytes(b"changed")
    elif change == "manifest":
        path = directory / "manifest.json"
        value = json.loads(path.read_text())
        value["environment"]["contact_impratio"] = 1
        path.write_text(json.dumps(value))
    elif change == "evidence":
        value = json.loads(evidence.read_text())
        value["uncompressed_sha256"] = "wrong"
        evidence.write_text(json.dumps(value))
    else:
        value = json.loads(evidence.read_text())
        archive = Path(value["full_report_archive"])
        report = json.loads(gzip.decompress(archive.read_bytes()))
        report["run"]["source_hashes"]["rl/model.py"] = "changed"
        raw = json.dumps(report).encode()
        archive.write_bytes(gzip.compress(raw))
        value["uncompressed_sha256"] = sha256(raw)
        evidence.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify_checkpoint(directory, model, evidence)


@pytest.mark.parametrize(
    "module", ["rl/manipulation/feedback_policy.py", "rl/manipulation/control.py", "rl/manipulation/turn_guard.py", "rl/artifacts.py"]
)
def test_feedback_inference_source_changes_are_rejected(checkpoint, module):
    directory, evidence, _, model = checkpoint
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["environment"]["feedback_policy"] = True
    manifest_path.write_text(json.dumps(manifest))
    summary = json.loads(evidence.read_text())
    archive = Path(summary["full_report_archive"])
    report = json.loads(gzip.decompress(archive.read_bytes()))
    report["checkpoints"]["test"]["manifest"] = manifest
    report["run"]["source_hashes"][module] = "changed"
    raw = json.dumps(report).encode()
    archive.write_bytes(gzip.compress(raw))
    summary["uncompressed_sha256"] = sha256(raw)
    evidence.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="sources or runtime changed"):
        verify_checkpoint(directory, model, evidence)
