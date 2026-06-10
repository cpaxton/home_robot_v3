# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Paths and environment defaults for the SQA3D benchmark."""

from __future__ import annotations

import os
from pathlib import Path

SQA3D_SPLITS = ("train", "val", "test")

ZENODO_SQA_TASK_URL = "https://zenodo.org/record/7792397/files/sqa_task.zip?download=1"
ZENODO_LOCALIZATION_URL = "https://zenodo.org/record/7792397/files/localization_task.zip?download=1"


def _xdg_cache() -> Path:
    raw = os.environ.get("XDG_CACHE_HOME", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".cache"


def default_sqa3d_data_dir() -> Path:
    """Root directory for SQA3D JSON annotations (``sqa_task/``, ``localization_task/``)."""
    raw = os.environ.get("SQA3D_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _xdg_cache() / "sqa3d" / "data"


def sqa_task_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or default_sqa3d_data_dir()) / "sqa_task"


def localization_task_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or default_sqa3d_data_dir()) / "localization_task"


def balanced_dir(data_dir: Path | None = None) -> Path:
    return sqa_task_dir(data_dir) / "balanced"


def _resolve_split_json(data_dir: Path | None, filename: str) -> Path:
    """Zenodo layout first, then flat dir (CI fixtures)."""
    root = data_dir or default_sqa3d_data_dir()
    standard = balanced_dir(data_dir) / filename
    if standard.is_file():
        return standard
    flat = root / filename
    if flat.is_file():
        return flat
    return standard


def questions_json_path(split: str, data_dir: Path | None = None) -> Path:
    if split not in SQA3D_SPLITS:
        raise ValueError(f"split must be one of {SQA3D_SPLITS}, got {split!r}")
    return _resolve_split_json(data_dir, f"v1_balanced_questions_{split}_scannetv2.json")


def annotations_json_path(split: str, data_dir: Path | None = None) -> Path:
    if split not in SQA3D_SPLITS:
        raise ValueError(f"split must be one of {SQA3D_SPLITS}, got {split!r}")
    return _resolve_split_json(data_dir, f"v1_balanced_sqa_annotations_{split}_scannetv2.json")


def localization_json_path(split: str, data_dir: Path | None = None) -> Path:
    if split not in SQA3D_SPLITS:
        raise ValueError(f"split must be one of {SQA3D_SPLITS}, got {split!r}")
    loc_root = localization_task_dir(data_dir)
    for sub in ("balanced", ""):
        base = loc_root / sub if sub else loc_root
        path = base / f"v1_balanced_localization_{split}_scannetv2.json"
        if path.is_file():
            return path
    return loc_root / "balanced" / f"v1_balanced_localization_{split}_scannetv2.json"


def answer_dict_path(data_dir: Path | None = None) -> Path:
    return sqa_task_dir(data_dir) / "answer_dict.json"
