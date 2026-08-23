# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""HuggingFace Hub push/pull helpers for scene map caches.

Repo id: ``EMET_SCENE_MAP_HF_REPO`` (e.g. ``org/emet-scene-maps``).
Layout in the hub dataset: ``maps/<key>/manifest.json``, ``voxel_map.pkl``, …
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from emet.eval.scene_map_cache import (
    has_cached_map,
    scene_cache_dir,
    scene_cache_root,
)

_ENV_HF_REPO = "EMET_SCENE_MAP_HF_REPO"


def default_hf_repo_id() -> str | None:
    raw = os.environ.get(_ENV_HF_REPO, "").strip()
    return raw or None


def pull_scene_map(
    key: str,
    *,
    dest: Path | str | None = None,
    repo_id: str | None = None,
) -> Path | None:
    """Download ``maps/<key>/`` from the HF dataset repo into the local cache.

    Returns the local directory when ``has_cached_map`` is true after pull; else None.
    Raises only on programming errors; network / missing-repo failures return None.
    """
    rid = (repo_id or default_hf_repo_id() or "").strip()
    if not rid:
        return None
    out = Path(dest) if dest is not None else scene_cache_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        return None

    prefix = f"maps/{key}/"
    try:
        files = list_repo_files(rid, repo_type="dataset")
    except Exception:
        return None
    matched = [f for f in files if f.startswith(prefix)]
    if not matched:
        return None
    try:
        for rel in matched:
            local_name = rel[len(prefix) :]
            if not local_name or local_name.endswith("/"):
                continue
            target = out / local_name
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=rid,
                filename=rel,
                repo_type="dataset",
            )
            shutil.copy2(downloaded, target)
    except Exception:
        return None
    return out if has_cached_map(out) else None


def push_scene_map(
    key: str,
    *,
    src: Path | str | None = None,
    repo_id: str | None = None,
    private: bool = True,
) -> bool:
    """Upload a local cache directory to ``maps/<key>/`` on the HF dataset repo.

    Creates the dataset repo if missing. Returns True on success.
    """
    rid = (repo_id or default_hf_repo_id() or "").strip()
    if not rid:
        raise ValueError(f"Set {_ENV_HF_REPO} (e.g. org/emet-scene-maps) before pushing scene maps")
    src_dir = Path(src) if src is not None else scene_cache_dir(key)
    if not has_cached_map(src_dir):
        raise FileNotFoundError(f"incomplete scene cache at {src_dir} (need manifest + voxel pickle)")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(rid, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(src_dir),
        path_in_repo=f"maps/{key}",
        repo_id=rid,
        repo_type="dataset",
    )
    return True


def list_local_cache_keys(*, root: Path | str | None = None) -> list[str]:
    """Keys under the local cache root that look complete."""
    base = scene_cache_root(root)
    if not base.is_dir():
        return []
    keys: list[str] = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and has_cached_map(child):
            keys.append(child.name)
    return keys
