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

"""Sanitize PYTHONPATH / sys.path so the project venv wins.

Two failure modes this module exists for:

* **ROS ``cv2``:** ``PYTHONPATH`` includes ``/opt/ros/...`` so ``import cv2`` is a stub
  without ``resize`` / ``imencode``.
* **Mixed interpreter site-packages:** a 3.10 venv that still contains
  ``.venv/lib/python3.12/site-packages`` would glob-prepend ABI-mismatched scipy/numpy
  and break ``emet serve mujoco`` on import.

Docs: ``docs/pythonpath.md``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _venv_python() -> Path | None:
    root = _repo_root()
    for name in ("python", "python3"):
        p = root / ".venv" / "bin" / name
        if p.exists():
            return p
    return None


def _venv_python_tag() -> str | None:
    """Active venv interpreter tag (``3.10``) from ``pyvenv.cfg`` ``version_info``."""
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
    lib = _repo_root() / ".venv" / "lib"
    if not lib.is_dir():
        return []
    tag = _venv_python_tag()
    if tag:
        # Only the venv's own interpreter version. A stray ``python3.12`` dir in a
        # 3.10 venv would otherwise leak ABI-mismatched wheels (broken scipy import).
        path = lib / f"python{tag}" / "site-packages"
        return [str(path)] if path.is_dir() else []
    return [str(p) for p in sorted(lib.glob("python*/site-packages")) if p.is_dir()]


def _is_ros_or_conflicting_pythonpath_entry(path: str) -> bool:
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
    """Return env with ROS entries stripped from PYTHONPATH and project src/venv prepended.

    Venv prepend is the active interpreter's ``python{tag}/site-packages`` only
    (see :func:`_venv_site_packages_paths`). Use this for every ``Popen`` of
    ``emet.simulation.mujoco_server`` / ``uv run emet serve``.
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
    """Mutate process env + sys.path so venv imports win; drop a broken preloaded cv2."""
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
