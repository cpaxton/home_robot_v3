# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# MolmoSpaces integration: scene/robot name mapping and runner env discovery.
# The actual sim runs in a separate venv (molmo-spaces requires mujoco 3.4, numpy>=2.2).
# See docs/molmospaces.md.

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# Robots supported by MolmoSpaces (from molmo_spaces_constants / their assets).
# rby1 / rby1m are Rainbow Robotics "Galaxea R1" family. Emet's merge-scene path only wires robots
# with vendored MJCFs; see :func:`emet_molmospaces_merge_robot_keys_with_mjcf`.
MOLMOSPACES_ROBOT_IDS = [
    "rby1",
    "rby1m",
    "franka_droid",
    "franka_cap",
    "floating_rum",
    "floating_robotiq",
    "franka_fr3",
]

DEFAULT_MOLMOSPACES_ROBOT = "rby1"

# Scene names used by MolmoSpaces get_scenes(scene_name, split).
# ithor = MSCrafted, procthor-10k = MSProc, procthor-objaverse = MSProcObja, holodeck-objaverse = MSMultiType.
MOLMOSPACES_SCENE_NAMES = [
    "ithor",
    "procthor-10k",
    "procthor-objaverse",
    "holodeck-objaverse",
]

MOLMOSPACES_SPLITS = ("train", "val", "test")

# Hello Stretch has no MJCF on the emet-molmospaces merge path; reject instead of substituting rby1.
_STRETCH_LIKE_ROBOT_TOKENS = frozenset({"stretch", "hello_stretch", "hellostretch"})


def normalize_molmospaces_cli_robot_token(robot: str) -> str:
    """Lowercase CLI robot token with hyphens mapped to underscores (``rb-y1`` → ``rb_y1``)."""
    return str(robot).strip().lower().replace("-", "_")


def is_stretch_like_robot_token(robot: str) -> bool:
    """True for Hello Stretch CLI names that have no MolmoSpaces merge MJCF."""
    return normalize_molmospaces_cli_robot_token(robot) in _STRETCH_LIKE_ROBOT_TOKENS


def emet_molmospaces_merge_robot_keys_with_mjcf() -> tuple[str, ...]:
    """Robot ids that resolve to an on-disk MJCF for ``emet-molmospaces merge-scene`` (vendored under emet)."""
    from emet.utils.assets import get_robot_mjcf_path

    keys = ("rby1", "galaxea_r1", "rb_y1", "innate_mars", "maurice")
    out: list[str] = []
    for k in keys:
        p = get_robot_mjcf_path(k)
        if p is not None and p.is_file() and k not in out:
            out.append(k)
    return tuple(out)


def validate_robot_for_molmospaces_merge(robot: str) -> str:
    """Return normalized robot id for ``merge-scene --robot``.

    Raises:
        ValueError: Stretch-like names (no bundled Stretch-in-room MJCF) or unknown robots without MJCF.
    """
    from emet.utils.assets import get_robot_mjcf_path

    key = normalize_molmospaces_cli_robot_token(robot)
    if key in _STRETCH_LIKE_ROBOT_TOKENS:
        emet_keys = ", ".join(emet_molmospaces_merge_robot_keys_with_mjcf()) or "(none found)"
        molmo = ", ".join(MOLMOSPACES_ROBOT_IDS)
        raise ValueError(
            "MolmoSpaces merged scenes do not include a Hello Robot Stretch MJCF on this path. "
            f"Use a robot with a vendored merge MJCF (e.g. {emet_keys}; try innate_mars for a different mobile model). "
            "For `emet serve mujoco --molmospaces-scene …`, omit `--robot` to default to rby1. "
            f"(Upstream MolmoSpaces asset robot ids include: {molmo}; emet merge-scene only wires bundled MJCFs.)"
        )
    mjcf = get_robot_mjcf_path(key)
    if mjcf is None or not mjcf.is_file():
        emet_keys = ", ".join(emet_molmospaces_merge_robot_keys_with_mjcf()) or "(none)"
        raise ValueError(
            f"No vendored MJCF for MolmoSpaces merge with --robot {robot!r}. "
            f"Use one of: {emet_keys}."
        )
    return key


def default_molmospaces_assets_dir() -> Path:
    """Default ``MLSPACES_ASSETS_DIR``: user cache (XDG_CACHE_HOME or ``~/.cache``), not under the venv."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache) / "molmospaces" / "assets"
    return Path.home() / ".cache" / "molmospaces" / "assets"


def default_molmospaces_cache_dir() -> Path:
    """Default ``MLSPACES_CACHE_DIR``: sibling of :func:`default_molmospaces_assets_dir`.

    ``molmospaces_resources.ResourceManager`` requires ``symlink_dir`` (from ``MLSPACES_ASSETS_DIR``)
    and ``cache_dir`` to differ and neither may contain the other.
    """
    return companion_cache_dir_for_assets(default_molmospaces_assets_dir())


def companion_cache_dir_for_assets(assets_path: Path) -> Path:
    """Directory for extracted archives next to *assets_path* (``…/resource_cache`` beside ``…/assets``)."""
    return assets_path.resolve().parent / "resource_cache"


def ensure_molmospaces_assets_dir_env(env: dict[str, str] | None = None) -> Path:
    """If ``MLSPACES_ASSETS_DIR`` is unset, set it to :func:`default_molmospaces_assets_dir` and mkdir.

    If ``MLSPACES_CACHE_DIR`` is unset, set it to :func:`companion_cache_dir_for_assets` for the
    resolved assets path. Upstream forbids using the same path for both (and forbids nesting).

    Pass a subprocess *env* dict (e.g. ``os.environ.copy()``) so ``emet`` forwards the same defaults
    to ``emet-molmospaces`` before MolmoSpaces imports resolve paths.
    """
    key = "MLSPACES_ASSETS_DIR"
    cache_key = "MLSPACES_CACHE_DIR"

    def _ensure_default_cache_dir(assets_path: Path) -> None:
        companion = companion_cache_dir_for_assets(assets_path)
        if env is None:
            if not os.environ.get(cache_key, "").strip():
                os.environ[cache_key] = str(companion)
                companion.mkdir(parents=True, exist_ok=True)
            return
        cur_cache = (env.get(cache_key) or os.environ.get(cache_key, "")).strip()
        if not cur_cache:
            env[cache_key] = str(companion)
            companion.mkdir(parents=True, exist_ok=True)

    if env is None:
        cur = os.environ.get(key, "").strip()
        if not cur:
            path = default_molmospaces_assets_dir()
            os.environ[key] = str(path)
            path.mkdir(parents=True, exist_ok=True)
            _ensure_default_cache_dir(path)
            return path
        path = Path(cur)
        _ensure_default_cache_dir(path)
        return path
    cur = (env.get(key) or os.environ.get(key, "")).strip()
    if not cur:
        path = default_molmospaces_assets_dir()
        env[key] = str(path)
        path.mkdir(parents=True, exist_ok=True)
        _ensure_default_cache_dir(path)
        return path
    path = Path(cur)
    _ensure_default_cache_dir(path)
    return path


def _scenes_objects_symlink_for_root(root: Path) -> None:
    """``<root>/scenes/objects`` → ``<root>/objects`` so ``../objects`` from scene XML resolves."""
    target = root / "objects"
    if not target.is_dir():
        return
    link = root / "scenes" / "objects"
    if link.is_symlink():
        try:
            if link.resolve() == target.resolve():
                return
        except OSError:
            pass
        try:
            link.unlink()
        except OSError:
            return
    elif link.is_dir():
        try:
            shutil.rmtree(link)
        except OSError:
            return
    elif link.exists():
        try:
            link.unlink()
        except OSError:
            return
    try:
        (root / "scenes").mkdir(parents=True, exist_ok=True)
        link.symlink_to(target.resolve(), target_is_directory=True)
    except OSError:
        pass


def _flatten_single_versioned_child(parent: Path) -> None:
    """If *parent* has exactly one subdirectory (e.g. THOR version), symlink each child up.

    Cache layout is ``objects/thor/<version>/Kitchen Objects/...`` while MJCF and asset links
    often reference ``objects/thor/Kitchen Objects/...``. The assets tree is already flat; the
    raw cache tree needs sibling symlinks next to the version folder.
    """
    if not parent.is_dir():
        return
    subs = [p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"]
    if len(subs) != 1:
        return
    vroot = subs[0]
    for child in vroot.iterdir():
        link = parent / child.name
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        except OSError:
            pass


def ensure_molmo_asset_layout_symlinks() -> None:
    """Link ``scenes/objects`` → ``objects`` under MolmoSpaces roots (assets and cache).

    Scene MJCF uses paths like ``../objects/thor/...``. From ``scenes/<dataset>/`` that resolves to
    ``scenes/objects/...``, not the real ``objects/`` tree at the dataset root. MolmoSpaces installs
    meshes under ``<MLSPACES_ASSETS_DIR>/objects`` and (for GLOBAL link strategy) under
    ``<MLSPACES_CACHE_DIR>/objects``. Absolute paths in merged MJCF may point under *cache*; we must
    symlink ``<cache>/scenes/objects`` → ``<cache>/objects`` as well as under assets.

    Under ``.../objects/thor``, the cache often keeps a single version directory (e.g. ``20251117``)
    with ``Kitchen Objects`` inside; MJCF may reference ``thor/Kitchen Objects`` without the version
    segment — mirror each version child up next to that folder when there is exactly one version dir.
    """
    assets = ensure_molmospaces_assets_dir_env()
    _scenes_objects_symlink_for_root(assets)
    _flatten_single_versioned_child(assets / "objects" / "thor")
    cache_raw = os.environ.get("MLSPACES_CACHE_DIR", "").strip()
    if cache_raw:
        cache = Path(cache_raw)
        if cache.resolve() != assets.resolve():
            _scenes_objects_symlink_for_root(cache)
            _flatten_single_versioned_child(cache / "objects" / "thor")


def galaxea_r1_assets_directory() -> Path:
    """Directory containing packaged ``galaxea_r1.xml``.

    MolmoSpaces merge writes a top-level wrapper MJCF that includes the scene and this robot.
    That file must live in this directory (not under ``/tmp``): MuJoCo resolves ``assetdir="meshes"``
    for the included robot XML relative to the main file path, so a merge under ``/tmp`` breaks
    mesh loading for Galaxea R1 (rby1).
    """
    import emet

    d = Path(emet.__file__).resolve().parent / "assets" / "robot" / "galaxea_r1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _python_can_import_molmo_spaces(py: Path) -> bool:
    """True if this interpreter can import the MolmoSpaces APIs the emet-molmospaces wrapper uses."""
    try:
        r = subprocess.run(
            [
                str(py),
                "-c",
                "from molmo_spaces.molmo_spaces_constants import get_scenes; "
                "from molmo_spaces.utils.lazy_loading_utils import "
                "install_scene_with_objects_and_grasps_from_path",
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def build_molmospaces_wrapper_command(args: list[str]) -> list[str] | None:
    """Return full argv to run the MolmoSpaces wrapper, or None if unavailable.

    **Important:** Prefer the repo's ``.venv-molmospaces`` when it passes the same import check
    as the wrapper (``get_scenes`` + ``install_scene_with_objects_and_grasps_from_path``).
    ``MOLMOSPACES_PYTHON`` is only used if it passes the same check (so a stale conda override
    cannot shadow a working project venv).
    """
    import shutil

    root = _project_root()
    local_py = root / ".venv-molmospaces" / "bin" / "python"
    local_exe = root / ".venv-molmospaces" / "bin" / "emet-molmospaces"

    # 1) Repo-local MolmoSpaces venv (authoritative after ./install.sh --molmospaces)
    if local_py.exists() and _python_can_import_molmo_spaces(local_py):
        if local_exe.exists():
            return [str(local_exe)] + args
        return [str(local_py), "-m", "emet_molmospaces"] + args

    # 2) Explicit override — only if molmo_spaces actually imports there
    env_py = os.environ.get("MOLMOSPACES_PYTHON")
    if env_py:
        p = Path(env_py).resolve()
        if p.exists() and _python_can_import_molmo_spaces(p):
            exe = p.parent / "emet-molmospaces"
            if exe.exists():
                return [str(exe)] + args
            return [str(p), "-m", "emet_molmospaces"] + args

    # 3) PATH — only if sibling python passes the same MolmoSpaces API import check
    which = shutil.which("emet-molmospaces")
    if which:
        w = Path(which)
        sibling_py = w.parent / "python"
        if sibling_py.exists() and _python_can_import_molmo_spaces(sibling_py):
            return [str(w)] + args

    return None


def get_molmospaces_wrapper_exe() -> Path | None:
    """Return path to the ``emet-molmospaces`` executable if installed as a script, else None."""
    cmd = build_molmospaces_wrapper_command([])
    if cmd is None:
        return None
    if Path(cmd[0]).name == "emet-molmospaces":
        return Path(cmd[0])
    return None
