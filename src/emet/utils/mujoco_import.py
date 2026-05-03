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

"""Fail fast when the MuJoCo Python package is missing (optional ``sim`` extra)."""

from __future__ import annotations

import os


def assert_mujoco_available() -> None:
    """Raise with install hints if ``mujoco`` is missing or GL backend init fails."""
    mujoco_gl = os.environ.get("MUJOCO_GL")
    try:
        import mujoco  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The mujoco package is not installed. From the repo root run:\n"
            "  uv sync\n"
            "For base deps only (no dev/sim/SAM-2):  uv sync --no-default-groups\n"
            "For robocasa/robosuite clones + sim pip deps:  ./install.sh --sim -y  (or: emet install sim)\n"
        ) from e
    except Exception as e:
        # MuJoCo loads GL at import time; wrong or broken MUJOCO_GL surfaces as AttributeError/OSError, not ImportError.
        if mujoco_gl and mujoco_gl.lower() == "osmesa":
            raise RuntimeError(
                "MuJoCo failed to initialize with MUJOCO_GL=osmesa (PyOpenGL could not load OSMesa / OpenGL; "
                "often ``AttributeError: 'NoneType' object has no attribute 'glGetError'`` when libOSMesa is missing).\n\n"
                "For headless Linux in this project, prefer leaving MUJOCO_GL unset (emet serve sets egl when appropriate) "
                "or explicitly: MUJOCO_GL=egl\n"
                "  sudo apt install libegl1-mesa libgles2-mesa\n\n"
                "If you truly need OSMesa: sudo apt install libosmesa6-dev and verify PyOpenGL can load it.\n\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        if mujoco_gl:
            raise RuntimeError(
                f"MuJoCo failed to import or initialize with MUJOCO_GL={mujoco_gl!r}. "
                "Try unsetting it, or MUJOCO_GL=egl (headless Linux) / glfw (desktop with DISPLAY).\n\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        raise
