"""Headless MuJoCo + cameras on WSL2: GPU EGL often hangs in ``mujoco.Renderer``; Mesa LLVMpipe is reliable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

WSL_HEADLESS_CAMERA_GL_WARNING = (
    "WSL2 kernel detected (headless, cameras on, no DISPLAY): set LIBGL_ALWAYS_SOFTWARE=1 "
    "for Mesa LLVMpipe; unset or use LIBGL_ALWAYS_SOFTWARE=0 to try GPU EGL instead."
)


def enable_software_gl_for_wsl_headless_cameras(*, headless: bool, no_cameras: bool) -> bool:
    """
    If this looks like WSL2, ``DISPLAY`` is unset, we need offscreen cameras, and the user did not
    configure ``LIBGL_ALWAYS_SOFTWARE``, set ``LIBGL_ALWAYS_SOFTWARE=1`` for Mesa LLVMpipe.

    Returns True when this function set the variable.
    """
    if sys.platform != "linux" or not headless or no_cameras or os.environ.get("DISPLAY"):
        return False
    if "LIBGL_ALWAYS_SOFTWARE" in os.environ:
        return False
    try:
        osrelease = Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False
    if "microsoft" not in osrelease or "wsl" not in osrelease:
        return False
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return True
