# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import hashlib
import runpy
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
