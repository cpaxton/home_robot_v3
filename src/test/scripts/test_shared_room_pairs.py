# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def assets():
    root = Path(__file__).resolve().parents[3]
    with patch.object(sys, "path", [str(root / "scripts"), *sys.path]):
        return runpy.run_path(str(root / "scripts/run_shared_room_pairs.py"))["grounding_assets"]


def test_detector_control_does_not_require_sam(tmp_path):
    assert assets()({"main": tmp_path}, ["main"], "yoloe") == {}


@pytest.mark.parametrize("change", ["untracked", "tracked", "staged"])
def test_frozen_source_rejects_executable_source_changes(tmp_path, change):
    root = Path(__file__).resolve().parents[3]
    with patch.object(sys, "path", [str(root / "scripts"), *sys.path]):
        module = runpy.run_path(str(root / "scripts/run_shared_room_pairs.py"))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    source = tmp_path / "src"
    source.mkdir()
    tracked = source / "tracked.py"
    tracked.write_text("# committed source\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD")
    module["assert_frozen_source"](tmp_path, revision)
    if change == "untracked":
        (source / "sitecustomize.py").write_text("raise RuntimeError('must not execute')\n")
    else:
        tracked.write_text("# changed executable source\n")
        if change == "staged":
            git("add", ".")
    with pytest.raises(RuntimeError, match="Frozen source changed"):
        module["assert_frozen_source"](tmp_path, revision)


def test_vlm_requires_complete_named_checkpoint_not_partial_download(tmp_path):
    (tmp_path / "sam2.1_hiera_small.pt123.tmp").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="no in-episode downloads"):
        assets()({"candidate": tmp_path}, ["candidate"], "vlm")


def test_only_selected_arms_require_assets_and_record_resolved_hash(tmp_path):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"fixture")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "sam2.1_hiera_small.pt").symlink_to(source)
    result = assets()({"main": tmp_path / "absent", "candidate": candidate}, ["candidate"], "vlm")
    assert result == {"candidate": {"path": str(source), "sha256": hashlib.sha256(b"fixture").hexdigest()}}


def test_explicit_model_authority_is_written_to_room_config_and_manifest(tmp_path, monkeypatch):
    import json

    root = Path(__file__).resolve().parents[3]
    with patch.object(sys, "path", [str(root / "scripts"), *sys.path]):
        run = runpy.run_path(str(root / "scripts/run_shared_room_pairs.py"))["run"]
    monkeypatch.setitem(run.__globals__, "git", lambda root, *args: "frozen" if args[0] == "rev-parse" else "")

    def stop_before_sim(*args, **kwargs):
        raise RuntimeError("test stops before simulation")

    monkeypatch.setattr(run.__globals__["subprocess"], "run", stop_before_sim)
    output = tmp_path / "out"
    args = SimpleNamespace(
        main=root,
        candidate=root,
        output=output,
        arms=["candidate"],
        grounding_backend="yoloe",
        decision_policy="model_final",
        episodes=["fixture"],
        seed=0,
        scan_steps=4,
    )
    with pytest.raises(RuntimeError, match="test stops"):
        run(args)
    assert "agentic_decision_policy: model_final" in (output / "fixture-candidate/config.yaml").read_text()
    assert json.loads((output / "manifest.json").read_text())["decision_policy"] == "model_final"


def test_untracked_selected_source_aborts_before_manifest_or_launch(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    with patch.object(sys, "path", [str(root / "scripts"), *sys.path]):
        run = runpy.run_path(str(root / "scripts/run_shared_room_pairs.py"))["run"]

    def git(checkout, *args):
        if args[0] == "rev-parse":
            return "frozen"
        if args[0] == "ls-files" and checkout == tmp_path / "candidate":
            return "src/sitecustomize.py"
        return ""

    monkeypatch.setitem(run.__globals__, "git", git)
    with patch.object(run.__globals__["subprocess"], "run") as launch:
        args = SimpleNamespace(
            main=tmp_path / "main",
            candidate=tmp_path / "candidate",
            output=tmp_path / "out",
            arms=["candidate"],
            grounding_backend="yoloe",
        )
        with pytest.raises(RuntimeError, match="untracked files"):
            run(args)
        launch.assert_not_called()
    assert not args.output.exists()
