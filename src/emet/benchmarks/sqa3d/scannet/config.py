# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ScanNet on-disk layout for SQA3D embodied replay."""

from __future__ import annotations

import os
from pathlib import Path

SCANNET_MESH_SUFFIX = "_vh_clean_2.ply"
SCANNET_SENS_SUFFIX = ".sens"
SQA3D_MIN_FILETYPES = ("_vh_clean_2.ply", ".txt")
SQA3D_SENS_FILETYPES = ("_vh_clean_2.ply", ".txt", ".sens")


def _xdg_cache() -> Path:
    raw = os.environ.get("XDG_CACHE_HOME", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".cache"


def default_scannet_root() -> Path:
    """Root with ``scans/<scene_id>/`` (output of ``download-scannet.py``)."""
    raw = os.environ.get("SCANNET_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _xdg_cache() / "scannet"


def default_download_script() -> Path:
    env = os.environ.get("SCANNET_DOWNLOAD_SCRIPT", "").strip()
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[5]
    return repo / "scripts" / "scannet" / "download-scannet.py"


def scene_scan_dir(scene_id: str, scannet_root: Path | None = None) -> Path:
    root = scannet_root or default_scannet_root()
    for sub in ("scans", "scans_test"):
        candidate = root / sub / scene_id
        if candidate.is_dir():
            return candidate
    return root / "scans" / scene_id


def scene_mesh_path(scene_id: str, scannet_root: Path | None = None) -> Path:
    return scene_scan_dir(scene_id, scannet_root) / f"{scene_id}{SCANNET_MESH_SUFFIX}"


def scene_sens_path(scene_id: str, scannet_root: Path | None = None) -> Path:
    return scene_scan_dir(scene_id, scannet_root) / f"{scene_id}{SCANNET_SENS_SUFFIX}"


def scene_meta_txt_path(scene_id: str, scannet_root: Path | None = None) -> Path:
    return scene_scan_dir(scene_id, scannet_root) / f"{scene_id}.txt"


def scene_assets_present(scene_id: str, scannet_root: Path | None = None) -> bool:
    return scene_mesh_path(scene_id, scannet_root).is_file()


def scene_sens_present(scene_id: str, scannet_root: Path | None = None) -> bool:
    return scene_sens_path(scene_id, scannet_root).is_file()


def scene_replay_assets_present(
    scene_id: str,
    scannet_root: Path | None = None,
    *,
    replay_mode: str = "auto",
) -> bool:
    if not scene_assets_present(scene_id, scannet_root):
        return False
    if replay_mode == "mesh":
        return True
    if replay_mode == "sens":
        return scene_sens_present(scene_id, scannet_root)
    return True


def filter_questions_with_scannet(
    questions: list,
    scannet_root: Path | None = None,
    *,
    replay_mode: str = "auto",
) -> list:
    """Keep questions with required ScanNet replay assets under ``SCANNET_ROOT``."""
    root = scannet_root or default_scannet_root()
    return [
        q
        for q in questions
        if scene_replay_assets_present(q.scene_id, root, replay_mode=replay_mode)
    ]


def count_scannet_scenes_on_disk(
    scene_ids: list[str],
    scannet_root: Path | None = None,
) -> tuple[int, int]:
    root = scannet_root or default_scannet_root()
    present = sum(1 for s in scene_ids if scene_assets_present(s, root))
    return present, len(scene_ids)


def collect_sqa3d_scene_ids(
    split: str = "val",
    *,
    data_dir: Path | None = None,
    limit: int | None = None,
) -> list[str]:
    from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions

    qs = load_sqa3d_questions(split, data_dir=data_dir)
    scenes = sorted({q.scene_id for q in qs if q.scene_id})
    if limit is not None:
        scenes = scenes[:limit]
    return scenes


def collect_scenes_from_question_slice(
    split: str,
    question_start: int,
    question_end: int,
    *,
    data_dir: Path | None = None,
) -> list[str]:
    """Unique ``scene_id`` values for questions ``[question_start:question_end)`` in split order."""
    from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions

    qs = load_sqa3d_questions(split, data_dir=data_dir)
    subset = qs[question_start:question_end]
    return sorted({q.scene_id for q in subset if q.scene_id})
