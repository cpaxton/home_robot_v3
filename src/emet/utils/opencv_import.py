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
#
# Copyright (c) Hello Robot, Inc. All rights reserved.

"""Detect a broken or shadowed ``cv2`` (not real OpenCV) before sim / ZMQ image paths run."""

from __future__ import annotations

import importlib
import sys


def _cv2_missing_attrs(cv2_mod: object) -> list[str]:
    required_attrs = ("resize", "imencode", "INTER_AREA", "IMWRITE_JPEG_QUALITY")
    return [a for a in required_attrs if not hasattr(cv2_mod, a)]


def _sys_path_has_ros_prefix() -> bool:
    return any(isinstance(p, str) and "/opt/ros/" in p for p in sys.path)


def _drop_ros_entries_from_sys_path() -> int:
    """Remove ``/opt/ros/...`` entries from ``sys.path`` (ROS often shadows ``cv2``). Returns count removed."""
    before = list(sys.path)
    sys.path[:] = [p for p in before if not (isinstance(p, str) and "/opt/ros/" in p)]
    return len(before) - len(sys.path)


def _purge_cv2_from_sys_modules() -> None:
    for k in list(sys.modules):
        if k == "cv2" or k.startswith("cv2."):
            del sys.modules[k]


def assert_cv2_is_real_opencv() -> None:
    """Raise ImportError if ``cv2`` is not usable OpenCV (e.g. ROS stub shadowing opencv-python).

    Common case: ``PYTHONPATH`` includes ``/opt/ros/.../dist-packages`` so ``import cv2`` resolves
    to a minimal ``cv2`` without ``resize`` / ``imencode``. We strip ``/opt/ros/`` from ``sys.path``
    once and re-import when that pattern is detected; otherwise unset ``PYTHONPATH`` or reinstall
    ``opencv-python`` in the project venv.
    """
    import cv2

    missing = _cv2_missing_attrs(cv2)
    if not missing:
        return

    if _sys_path_has_ros_prefix():
        n = _drop_ros_entries_from_sys_path()
        if n:
            _purge_cv2_from_sys_modules()
            importlib.invalidate_caches()
            import cv2 as cv2_again

            missing = _cv2_missing_attrs(cv2_again)
            if not missing:
                sys.modules["cv2"] = cv2_again
                return

    loc = getattr(cv2, "__file__", repr(cv2))
    py = sys.executable
    raise ImportError(
        "The Python module named 'cv2' is not OpenCV (or is a broken build). "
        f"Missing attributes: {missing}. cv2.__file__ = {loc!r}\n\n"
        "This usually happens when ROS or another environment prepends a different 'cv2' "
        "ahead of opencv-python in your venv.\n\n"
        "Fix:\n"
        "  • From the repo:  uv run emet serve mujoco ...   (uses project .venv first)\n"
        "  • Or:  env -u PYTHONPATH uv run emet serve mujoco ...   (ignore inherited PYTHONPATH)\n"
        "  • Or:  unset PYTHONPATH   then activate .venv and run again\n"
        '  • Check:  uv run python -c "import cv2; print(cv2.__file__)"  '
        "(should be under .venv/.../cv2/)\n"
        "  • Reinstall OpenCV in the venv:  uv pip install --reinstall opencv-python\n\n"
        f"Current interpreter: {py}"
    )
