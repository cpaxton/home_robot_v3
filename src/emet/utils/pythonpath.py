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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Rewrite PYTHONPATH / sys.path so this checkout's ``.venv`` wins over the shell.

Why
---
``emet serve`` / ``emet run`` spawn a child with a copy of the parent environment.
Two parent environments commonly break that child:

1. **ROS ``cv2``.** Sourcing Humble/Jazzy/Noetic puts ``/opt/ros/...`` on
   ``PYTHONPATH``. ``import cv2`` then hits a stub without ``resize`` /
   ``imencode``, and sim image threads die.

2. **Mixed venv ABI.** A Python 3.10 ``.venv`` that still contains
   ``.venv/lib/python3.12/site-packages`` (partial ``uv`` / copied tree).
   Globbing ``python*/site-packages`` prepends *both*. scipy/numpy then load
   3.12 wheels into a 3.10 process (``undefined symbol``, API-version errors).

The ABI case cannot be fixed inside ``mujoco_server`` after start:
``emet.simulation.mujoco_server`` does ``import numpy`` at **module load**,
before ``ensure_venv_site_packages_first()`` in ``serve()``. The child's
``PYTHONPATH`` must already be tagged correctly at ``Popen`` time.

What to call
------------
``sanitize_emet_subprocess_env(env)``
    Copy ``env`` (default ``os.environ``), drop ROS entries, prepend ``src/``
    plus **this venv's** ``python{tag}/site-packages`` (tag from
    ``.venv/pyvenv.cfg`` ``version_info``). Pass as ``Popen(..., env=...)``.
    Used by ``emet.cli_cmds.bootstrap._run_module`` (every ``emet serve`` /
    ``emet run …`` child) and ``emet.app.robots_cli`` camera-preview spawn.

``ensure_venv_site_packages_first()``
    Apply the same rewrite to *this* process (``os.environ`` + ``sys.path``)
    and unload a stub ``cv2``. Does **not** un-import numpy/scipy already
    loaded from a bad ``PYTHONPATH``. Used by ``mujoco_server.serve`` (before
    the OpenCV check) and ``opencv_import.assert_cv2_is_real_opencv``.

Operator page: ``docs/pythonpath.md``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _venv_python() -> Path | None:
    """Repo ``.venv/bin/python`` if present (robots_cli execs this for children)."""
    root = _repo_root()
    for name in ("python", "python3"):
        p = root / ".venv" / "bin" / name
        if p.exists():
            return p
    return None


def _venv_python_tag() -> str | None:
    """``major.minor`` of *this* ``.venv`` (e.g. ``3.10``), from ``pyvenv.cfg``.

    Used so we prepend ``lib/python3.10/site-packages`` and not a leftover
    ``python3.12`` dir in the same venv. ``None`` if ``version_info`` is missing.
    """
    cfg = _repo_root() / ".venv" / "pyvenv.cfg"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version_info"):
                parts = line.split("=", 1)[1].strip().split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return f"{parts[0]}.{parts[1]}"
    except Exception:  # noqa: BLE001
        return None
    return None


def _venv_site_packages_paths() -> list[str]:
    """Venv site-packages to put first: the active interpreter only.

    With a tag, return ``.venv/lib/python{tag}/site-packages`` (or ``[]`` if
    that dir is missing). Do **not** glob every ``python*/site-packages`` —
    that is how a stray 3.12 tree leaked ABI-mismatched scipy into a 3.10
    ``emet serve mujoco`` child.

    Glob fallback is only when ``pyvenv.cfg`` has no ``version_info``.
    """
    lib = _repo_root() / ".venv" / "lib"
    if not lib.is_dir():
        return []
    tag = _venv_python_tag()
    if tag:
        path = lib / f"python{tag}" / "site-packages"
        return [str(path)] if path.is_dir() else []
    return [str(p) for p in sorted(lib.glob("python*/site-packages")) if p.is_dir()]


def _is_ros_or_conflicting_pythonpath_entry(path: str) -> bool:
    """True for empty entries and ROS distro paths that shadow venv ``cv2``."""
    p = path.replace("\\", "/")
    if not p:
        return True
    if "/opt/ros/" in p:
        return True
    if "/ros/noetic/" in p or "/ros/humble/" in p or "/ros/jazzy/" in p:
        return True
    if p.endswith("/ros2/lib/python3.10/site-packages") or "/ros2/lib/python" in p:
        return True
    return False


def sanitize_emet_subprocess_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Env for a child that must import this repo's numpy/OpenCV, not ROS/other CPythons.

    Copy ``env`` (or ``os.environ``): strip ROS ``PYTHONPATH`` entries, prepend
    ``<repo>/src`` then tagged venv site-packages. Pass to ``Popen`` / ``call``.

    Required at spawn for ``python -m emet.simulation.mujoco_server`` because
    that module imports numpy at load time — in-process
    :func:`ensure_venv_site_packages_first` runs too late.

    Callers: ``emet.cli_cmds.bootstrap._run_module`` (``emet serve``,
    ``emet run …``), ``emet.app.robots_cli.robots_preview_cameras``.
    """
    out = (env or os.environ).copy()
    pp = out.get("PYTHONPATH", "")
    filtered = [p for p in pp.split(os.pathsep) if p and not _is_ros_or_conflicting_pythonpath_entry(p)]

    prepend: list[str] = []
    src = _repo_root() / "src"
    if src.is_dir():
        prepend.append(str(src))
    prepend.extend(_venv_site_packages_paths())

    if prepend:
        merged = prepend + [p for p in filtered if p not in prepend]
        out["PYTHONPATH"] = os.pathsep.join(merged)
    elif filtered:
        out["PYTHONPATH"] = os.pathsep.join(filtered)
    elif "PYTHONPATH" in out:
        del out["PYTHONPATH"]
    return out


def ensure_venv_site_packages_first() -> None:
    """Fix *this* process: apply :func:`sanitize_emet_subprocess_env`, front-load ``sys.path``, drop stub ``cv2``.

    Use when the current interpreter already started (``mujoco_server.serve``,
    ``assert_cv2_is_real_opencv``). Cannot un-import numpy/scipy loaded from a
    bad ``PYTHONPATH`` — those children need :func:`sanitize_emet_subprocess_env`
    on the parent ``Popen``.
    """
    sanitized = sanitize_emet_subprocess_env()
    os.environ.update(sanitized)

    for sp in reversed(_venv_site_packages_paths()):
        if sp not in sys.path:
            sys.path.insert(0, sp)
    src = str(_repo_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    mod = sys.modules.get("cv2")
    if mod is not None and not hasattr(mod, "resize"):
        del sys.modules["cv2"]
