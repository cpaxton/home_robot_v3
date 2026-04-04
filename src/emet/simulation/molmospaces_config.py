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
import subprocess
from pathlib import Path

# Robots supported by MolmoSpaces (from molmo_spaces_constants / their assets).
# rby1 / rby1m are Rainbow Robotics "Galaxea R1" family.
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


def default_molmospaces_assets_dir() -> Path:
    """Default ``MLSPACES_ASSETS_DIR``: user cache (XDG_CACHE_HOME or ``~/.cache``), not under the venv."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache) / "molmospaces" / "assets"
    return Path.home() / ".cache" / "molmospaces" / "assets"


def ensure_molmospaces_assets_dir_env(env: dict[str, str] | None = None) -> Path:
    """If ``MLSPACES_ASSETS_DIR`` is unset, set it to :func:`default_molmospaces_assets_dir` and mkdir.

    Pass a subprocess *env* dict (e.g. ``os.environ.copy()``) so ``emet`` forwards the same default
    to ``emet-molmospaces`` before MolmoSpaces imports resolve ``ASSETS_DIR``.
    """
    key = "MLSPACES_ASSETS_DIR"
    if env is None:
        cur = os.environ.get(key, "").strip()
        if not cur:
            path = default_molmospaces_assets_dir()
            os.environ[key] = str(path)
            path.mkdir(parents=True, exist_ok=True)
            return path
        return Path(cur)
    cur = (env.get(key) or os.environ.get(key, "")).strip()
    if not cur:
        path = default_molmospaces_assets_dir()
        env[key] = str(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(cur)


def ensure_molmo_asset_layout_symlinks() -> None:
    """Link ``<MLSPACES_ASSETS_DIR>/scenes/objects`` → ``.../objects`` when possible.

    Scene MJCF under ``scenes/<dataset>/`` uses paths like ``../objects/thor/...``. That resolves to
    ``scenes/objects/...`` beside ``scenes/<dataset>``, not the real tree at the asset-root
    ``objects/``. Symlinking ``scenes/objects`` to the root ``objects`` directory makes those
    relative paths load the installed THOR meshes.
    """
    root = ensure_molmospaces_assets_dir_env()
    target = root / "objects"
    if not target.is_dir():
        return
    link = root / "scenes" / "objects"
    if link.exists() or link.is_symlink():
        return
    try:
        (root / "scenes").mkdir(parents=True, exist_ok=True)
        link.symlink_to(target.resolve(), target_is_directory=True)
    except OSError:
        pass


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
