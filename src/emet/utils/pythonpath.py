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

"""Sanitize PYTHONPATH / sys.path so project venv wins over ROS-shaded packages (e.g. broken cv2)."""

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


def _venv_site_packages_paths() -> list[str]:
    lib = _repo_root() / ".venv" / "lib"
    if not lib.is_dir():
        return []
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
    """Return env with ROS entries removed from PYTHONPATH and project src/venv prepended."""
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
