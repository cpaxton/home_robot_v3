# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Reusable per-scene baseline map cache (graph + voxel) for OVMM / dynamic explore.

Baselines are built once with perfect-depth / GT mapping and stored under
``~/.cache/emet/scene_maps/<key>/`` (override ``EMET_SCENE_MAP_CACHE_DIR``).
Consumers load via ``ensure_cached_map`` and skip rotate/explore when a cache hit
is available.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from emet.memory.format import MANIFEST_FILENAME, VOXEL_PICKLE_FILENAME

CACHE_META_FILENAME = "cache_meta.json"
BUILD_MODE_GT = "gt"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "emet" / "scene_maps"

# Residual explore steps after loading a cached map (usually 0).
_ENV_USE_CACHE = "EMET_USE_SCENE_MAP_CACHE"
_ENV_CACHE_DIR = "EMET_SCENE_MAP_CACHE_DIR"
_ENV_HF_REPO = "EMET_SCENE_MAP_HF_REPO"
_ENV_AUTO_PULL = "EMET_SCENE_MAP_CACHE_AUTO_PULL"


def scene_map_cache_enabled(*, override: bool | None = None) -> bool:
    """Return whether consumers should try the scene map cache.

    ``override`` wins when not None. Else ``EMET_USE_SCENE_MAP_CACHE`` (default on:
    unset / empty / 1 / true / yes).
    """
    if override is not None:
        return bool(override)
    raw = os.environ.get(_ENV_USE_CACHE, "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def scene_cache_root(root: Path | str | None = None) -> Path:
    """Resolve the scene-map cache root directory."""
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get(_ENV_CACHE_DIR, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CACHE_ROOT.resolve()


def _slug(value: Any) -> str:
    s = str(value if value is not None else "").strip().lower()
    out: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", "/", ".", ":"):
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def scene_cache_key(
    sim_cfg: Any,
    *,
    build_mode: str = BUILD_MODE_GT,
) -> str:
    """Canonical cache key from a sim launch config (+ build mode tag).

    Robocasa: ``robocasa_{task}_s{style}_l{layout}_seed{seed}_{robot}_gt``
    MolmoSpaces: ``molmo_{scene}_{split}_idx{index}_{robot}_gt``
    Default table: ``default_{robot}_seed{seed}_gt``
    """
    mode = _slug(build_mode) or BUILD_MODE_GT
    kind = str(getattr(sim_cfg, "kind", "") or "").strip().lower()
    robot = _slug(getattr(sim_cfg, "robot", "stretch"))
    seed = int(getattr(sim_cfg, "seed", 0) or 0)

    if kind == "robocasa":
        task = _slug(getattr(sim_cfg, "robocasa_task", "PickPlaceCounterToCabinet"))
        style = int(getattr(sim_cfg, "robocasa_style", 1) or 1)
        layout = int(getattr(sim_cfg, "robocasa_layout", 1) or 1)
        return f"robocasa_{task}_s{style}_l{layout}_seed{seed}_{robot}_{mode}"

    if kind == "molmospaces":
        scene = _slug(getattr(sim_cfg, "scene", "ithor"))
        split = _slug(getattr(sim_cfg, "split", "train"))
        index = int(getattr(sim_cfg, "index", 0) or 0)
        return f"molmo_{scene}_{split}_idx{index}_{robot}_{mode}"

    # default_mujoco / unknown
    scene_path = getattr(sim_cfg, "scene_path", None)
    if scene_path:
        return f"default_{robot}_{_slug(Path(scene_path).stem)}_seed{seed}_{mode}"
    return f"default_{robot}_seed{seed}_{mode}"


def scene_cache_dir(key: str, *, root: Path | str | None = None) -> Path:
    """Directory for one scene cache entry: ``<root>/<key>/``."""
    return scene_cache_root(root) / _slug(key).replace("__", "_")


def has_cached_map(cache_dir: Path | str) -> bool:
    """True when ``manifest.json`` and ``voxel_map.pkl`` are present."""
    d = Path(cache_dir)
    return (d / MANIFEST_FILENAME).is_file() and (d / VOXEL_PICKLE_FILENAME).is_file()


def _git_sha(repo_root: Path | None = None) -> str | None:
    try:
        root = repo_root or Path(__file__).resolve().parents[3]
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except Exception:
        return None


def write_cache_metadata(
    cache_dir: Path | str,
    sim_cfg: Any,
    *,
    build_params: dict[str, Any] | None = None,
    key: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Write ``cache_meta.json`` alongside the exported map."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    resolved_key = key or scene_cache_key(sim_cfg)
    sim_fields: dict[str, Any] = {
        "kind": getattr(sim_cfg, "kind", None),
        "robot": getattr(sim_cfg, "robot", None),
        "seed": getattr(sim_cfg, "seed", None),
    }
    for attr in (
        "robocasa_task",
        "robocasa_style",
        "robocasa_layout",
        "scene",
        "split",
        "index",
        "scene_path",
    ):
        if hasattr(sim_cfg, attr):
            sim_fields[attr] = getattr(sim_cfg, attr)
    meta = {
        "key": resolved_key,
        "build_mode": BUILD_MODE_GT,
        "created_unix": time.time(),
        "emet_git_sha": _git_sha(repo_root),
        "sim": sim_fields,
        "build_params": dict(build_params or {}),
        "has_manifest": (d / MANIFEST_FILENAME).is_file(),
        "has_voxel_pickle": (d / VOXEL_PICKLE_FILENAME).is_file(),
    }
    path = d / CACHE_META_FILENAME
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def try_pull_cached_map(key: str, *, dest: Path | None = None) -> Path | None:
    """Best-effort HuggingFace pull for ``key``. Returns local dir on success."""
    if os.environ.get(_ENV_AUTO_PULL, "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    repo_id = os.environ.get(_ENV_HF_REPO, "").strip()
    if not repo_id:
        return None
    try:
        from emet.eval.scene_map_cache_hub import pull_scene_map

        return pull_scene_map(key, dest=dest, repo_id=repo_id)
    except Exception:
        return None


def ensure_cached_map(
    key: str,
    *,
    root: Path | str | None = None,
    try_download: bool = True,
) -> Path | None:
    """Return local cache dir if present (optionally after HF pull); else ``None``."""
    d = scene_cache_dir(key, root=root)
    if has_cached_map(d):
        return d
    if try_download:
        pulled = try_pull_cached_map(key, dest=d)
        if pulled is not None and has_cached_map(pulled):
            return Path(pulled)
    return None


def resolve_scene_cache_for_sim(
    sim_cfg: Any,
    *,
    enabled: bool | None = None,
    root: Path | str | None = None,
    try_download: bool = True,
) -> Path | None:
    """Convenience: key from ``sim_cfg`` then ``ensure_cached_map`` when enabled."""
    if not scene_map_cache_enabled(override=enabled):
        return None
    return ensure_cached_map(
        scene_cache_key(sim_cfg),
        root=root,
        try_download=try_download,
    )
