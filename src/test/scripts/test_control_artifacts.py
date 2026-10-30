"""Ignored trace storage stays checksummed, retrievable and out of Git."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from emet.rl.artifacts import PREFIX, catalog, resolve_artifact

ROOT = Path(__file__).resolve().parents[3]


def test_gzip_ignore_overrides_paper_data_negation_and_index_has_no_traces():
    path = "paper/data/whole_body_control/new-run/report.json.gz"
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert result.returncode == 0
    spec = importlib.util.spec_from_file_location("artifact_check", ROOT / "scripts/check_generated_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check(ROOT)


def test_external_artifact_resolution_verifies_hash_and_reports_missing_store(tmp_path):
    data = b"compressed trace placeholder"
    relative = PREFIX / "test/report.json.gz"
    manifest = tmp_path / PREFIX / "artifacts.json"
    manifest.parent.mkdir(parents=True)
    entry = {"path": str(relative), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest.write_text(json.dumps({"artifacts": [entry]}))
    with pytest.raises(FileNotFoundError, match="control_artifacts.py restore"):
        resolve_artifact(relative, root=tmp_path)
    store = tmp_path / "store"
    target = store / "test/report.json.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    assert resolve_artifact(relative, root=tmp_path, store=store) == target
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_artifact(relative, root=tmp_path, store=store)


def test_artifact_catalog_refuses_path_escape(tmp_path):
    manifest = tmp_path / PREFIX / "artifacts.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"artifacts": [{"path": "paper/data/whole_body_control/../../escape.gz"}]}))
    with pytest.raises(ValueError, match="escapes"):
        catalog(tmp_path)


def test_force_added_binary_is_rejected_even_if_ignore_is_bypassed(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / PREFIX / "forced.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bad")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", str(path)], check=True)
    spec = importlib.util.spec_from_file_location("artifact_check", ROOT / "scripts/check_generated_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="Generated control artifacts"):
        module.check(tmp_path)
