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

"""Detect a broken or shadowed ``cv2`` (not real OpenCV) before sim / ZMQ image paths run.

ROS ``PYTHONPATH`` is the usual culprit; mixed venv interpreter tags are a
separate scipy/numpy ABI issue. Both: ``docs/pythonpath.md``.
"""

from __future__ import annotations

import sys


def assert_cv2_is_real_opencv() -> None:
    """Raise ImportError if ``cv2`` is not usable OpenCV (e.g. ROS stub shadowing opencv-python).

    Common case: ``PYTHONPATH`` includes ``/opt/ros/.../dist-packages`` so ``import cv2`` resolves
    to a minimal ``cv2`` without ``resize`` / ``imencode``. Fix: run with ``uv run emet ...`` from
    the repo, or unset ``PYTHONPATH`` / put the project ``.venv`` site-packages first.
    See ``docs/pythonpath.md``.
    """
    from emet.utils.pythonpath import ensure_venv_site_packages_first

    ensure_venv_site_packages_first()
    import cv2

    required_attrs = ("resize", "imencode", "INTER_AREA", "IMWRITE_JPEG_QUALITY")
    missing = [a for a in required_attrs if not hasattr(cv2, a)]
    if missing:
        loc = getattr(cv2, "__file__", repr(cv2))
        py = sys.executable
        raise ImportError(
            "The Python module named 'cv2' is not OpenCV (or is a broken build). "
            f"Missing attributes: {missing}. cv2.__file__ = {loc!r}\n\n"
            "This usually happens when ROS or another environment prepends a different 'cv2' "
            "ahead of opencv-contrib-python in your venv.\n\n"
            "Fix:\n"
            "  • From the repo:  uv run emet serve mujoco ...   (uses project .venv first)\n"
            "  • Or:  unset PYTHONPATH   then activate .venv and run again\n"
            '  • Check:  uv run python -c "import cv2; print(cv2.__file__)"  '
            "(should be under .venv/.../cv2/)\n"
            "  • Reinstall OpenCV in the venv:  uv pip install --reinstall opencv-contrib-python\n"
            "  • Do not install opencv-python and opencv-contrib-python together (breaks cv2).\n\n"
            f"Current interpreter: {py}"
        )
