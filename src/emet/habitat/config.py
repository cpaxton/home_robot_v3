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


def default_hm3d_data_path() -> Path:
    """Root passed to ``habitat_sim.utils.datasets_download --data-path``."""
    raw = os.environ.get("HM3D_DATA_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _xdg_cache() / "habitat_eqa" / "hm3d"


def default_hm3d_scene_dir() -> Path:
    """HM3D train split directory (HM-EQA scenes).

    After ``datasets_download``, scenes live under
    ``<HM3D_DATA_PATH>/scene_datasets/hm3d/train/<scene_id>/``.
    """
    raw = os.environ.get("HM3D_SCENE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return default_hm3d_data_path() / "scene_datasets" / "hm3d" / "train"


def questions_csv_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_habitat_eqa_data_dir()) / "questions.csv"


def scene_init_poses_csv_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_habitat_eqa_data_dir()) / "scene_init_poses.csv"


def openeqa_json_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_habitat_eqa_data_dir()) / "open-eqa-v0.json"


def hm3d_scene_short_name(scene_id: str) -> str:
    """HM3D mesh basename (e.g. ``00004-VqCaAuuoeWk`` → ``VqCaAuuoeWk``)."""
    if "-" in scene_id:
        return scene_id.split("-", 1)[1]
    return scene_id


def hm3d_scene_glb_path(scene_id: str, hm3d_root: Path | None = None) -> Path:
    """Resolve ``<train>/<scene_id>/<short_id>.basis.glb`` (Habitat HM3D layout)."""
    root = hm3d_root or default_hm3d_scene_dir()
    scene_dir = root / scene_id
    short = hm3d_scene_short_name(scene_id)
    candidates = [
        scene_dir / f"{short}.basis.glb",
        scene_dir / f"{scene_id}.basis.glb",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def hm3d_scene_navmesh_path(scene_id: str, hm3d_root: Path | None = None) -> Path:
    """Resolve HM3D navmesh next to the scene ``.basis.glb``."""
    glb = hm3d_scene_glb_path(scene_id, hm3d_root)
    return glb.parent / f"{glb.stem}.navmesh"


def hm3d_scene_semantic_glb_path(scene_id: str, hm3d_root: Path | None = None) -> Path:
    """Resolve ``<short_id>.semantic.glb`` for an HM3D scene."""
    glb = hm3d_scene_glb_path(scene_id, hm3d_root)
    short = hm3d_scene_short_name(scene_id)
    return glb.parent / f"{short}.semantic.glb"
