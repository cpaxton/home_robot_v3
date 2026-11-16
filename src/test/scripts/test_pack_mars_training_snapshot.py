"""Compute snapshots contain indirect test inputs and no generated traces."""

import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pack_snapshot", ROOT / "scripts/pack_mars_training_snapshot.py")
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)


@pytest.fixture
def repo(tmp_path):
    for name in packer.REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n" if name != ".gitignore" else "logs/\n")
    for name in ("paper/data/whole_body_control/generated.json.gz", "src/emet/assets/robot/native.STL"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Snapshot test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    secret = tmp_path / "src/untracked-secret.txt"
    secret.write_text("must not be copied")
    return tmp_path


def test_snapshot_includes_bridge_and_resolved_assets_and_checksums(repo):
    output = repo / "logs/snapshot.tar"
    manifest = packer.build_snapshot(repo, output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert set(packer.REQUIRED) <= set(names)
        assert "src/emet/assets/robot/native.STL" in names
        assert not any(name.endswith(".gz") or "secret" in name or name.startswith("logs/") for name in names)
        assert json.load(archive.extractfile("snapshot-manifest.json")) == manifest
        for entry in manifest["files"]:
            assert hashlib.sha256(archive.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    with pytest.raises(ValueError, match="fresh snapshot"):
        packer.build_snapshot(repo, output)


@pytest.mark.parametrize("failure", ["missing_bridge", "lfs_pointer"])
def test_snapshot_fails_before_publish_for_incomplete_inputs(repo, failure):
    if failure == "missing_bridge":
        (repo / "src/innate_mars_bridge/innate_mars_bridge/ros/camera.py").unlink()
    else:
        (repo / "src/emet/assets/robot/native.STL").write_text("version https://git-lfs.github.com/spec/v1\n")
    output = repo / "logs/snapshot.tar"
    with pytest.raises(ValueError, match="Missing required|Unresolved LFS"):
        packer.build_snapshot(repo, output)
    assert not output.exists() and not output.with_suffix(".tar.partial").exists()
