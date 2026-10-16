# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import hashlib
import runpy
from pathlib import Path

import pytest


def assets():
    root = Path(__file__).resolve().parents[3]
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
