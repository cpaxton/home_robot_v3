# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Paths and environment defaults for the Habitat EQA harness."""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_cache() -> Path:
    raw = os.environ.get("XDG_CACHE_HOME", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".cache"


def default_habitat_eqa_data_dir() -> Path:
    """Directory for HM-EQA CSV/JSON and OpenEQA files (GraphEQA layout)."""
    raw = os.environ.get("HABITAT_EQA_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _xdg_cache() / "habitat_eqa" / "data"


def default_hm3d_scene_dir() -> Path:
    """HM3D scene dataset root (train split for HM-EQA)."""
    raw = os.environ.get("HM3D_SCENE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _xdg_cache() / "habitat_eqa" / "hm3d" / "train"


def questions_csv_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_habitat_eqa_data_dir()) / "questions.csv"


def scene_init_poses_csv_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_habitat_eqa_data_dir()) / "scene_init_poses.csv"


def openeqa_json_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_habitat_eqa_data_dir()) / "open-eqa-v0.json"


def hm3d_scene_glb_path(scene_id: str, hm3d_root: Path | None = None) -> Path:
    """Resolve ``<hm3d_root>/<scene_id>/<scene_id>.basis.glb`` (HM3D train layout)."""
    root = hm3d_root or default_hm3d_scene_dir()
    return root / scene_id / f"{scene_id}.basis.glb"
