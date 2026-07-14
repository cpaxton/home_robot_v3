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
# This source code is licensed under the LICENSE file in the
# root directory of this source tree.

"""TTY diagnostics for head-camera frames (empty / black PNG / dtype issues)."""

from __future__ import annotations

import os

import numpy as np
from termcolor import colored

from emet.agent.env_flags import env_agent_camera_debug

_DISCORD_BGR_ENV = "EMET_DISCORD_IMAGES_BGR"  # "1"/"true" => raw OpenCV BGR matrices (JPEG not via from_jpg)


def discord_pil_bgr() -> bool:
    """For :func:`emet.utils.image.ndarray_hwc_to_pil_rgb_u8` when uploading to Discord.

    ZMQ JPEG decode (:func:`emet.utils.compression.from_jpg`) and MuJoCo ``Renderer.render`` buffers
    are **RGB**. Default is False (do not apply ``cv2.COLOR_BGR2RGB``) so Discord PNGs match pixel
    order. Legacy paths that feed raw ``cv2.imdecode`` buffers without converting to RGB should set
    ``EMET_DISCORD_IMAGES_BGR=1``.
    """
    v = os.environ.get(_DISCORD_BGR_ENV, "0").strip().lower()
    return v in ("1", "true", "yes", "on", "bgr", "y")


def print_camera_frame_diagnostics(
    where: str,
    arr: np.ndarray | None,
    *,
    force: bool = False,
) -> None:
    """Print one line (or a short block) of array stats. Respects :func:`env_agent_camera_debug` unless force.

    *force* is used when the caller is already in verbose ( tool ) mode.
    """
    if not force and not env_agent_camera_debug():
        return
    if arr is None:
        print(
            colored(f"[camera debug] {where}:", "magenta"),
            "no array (None)",
            flush=True,
        )
        return
    a = np.asarray(arr)
    if a.size == 0:
        print(
            colored(f"[camera debug] {where}:", "magenta"),
            "empty array",
            flush=True,
        )
        return
    flat = a.astype(np.float32).ravel()
    mn, mx = float(np.nanmin(flat)), float(np.nanmax(flat))
    mean = float(np.nanmean(flat))
    st = f"shape={a.shape} dtype={a.dtype} min={mn:.4g} max={mx:.4g} mean={mean:.4g} contiguous={a.flags.c_contiguous}"
    if a.ndim == 3 and a.shape[2] >= 3:
        cmeans = [float(np.nanmean(a[..., c])) for c in range(3)]
        st += f" ch_mean=[{cmeans[0]:.2f}, {cmeans[1]:.2f}, {cmeans[2]:.2f}]"
    print(colored(f"[camera debug] {where}:", "magenta"), st, flush=True)
    if mx < 1.0 and a.dtype in (np.uint8,):
        print(
            colored("  (hint)", "yellow"),
            "all pixel values are < 1; uint8 this usually means a black/empty buffer.",
            flush=True,
        )
    if mx < 1e-6 and a.size > 0:
        print(
            colored("  (hint)", "red"),
            "Array is numerically all-zero. Check that the sim is rendering (camera, EGL/GL) and ZMQ is publishing rgb.",
            flush=True,
        )
    if a.dtype in (np.float32, np.float64) and mx <= 1.0 + 1e-3 and mean < 0.01:
        print(
            colored("  (hint)", "yellow"),
            "Float image near black; if values should be in 0-255, dtype/range may be wrong.",
            flush=True,
        )
    if not rgb_frame_is_usable(a):
        print(
            colored("  (hint)", "yellow"),
            "Frame fails usability gate (near-white / near-black / near-constant); prefer live RGB.",
            flush=True,
        )


def rgb_frame_is_usable(
    arr: np.ndarray | None,
    *,
    min_std: float = 8.0,
    max_mean_white: float = 245.0,
    min_mean_black: float = 12.0,
) -> bool:
    """True when an HWC RGB crop/frame has enough contrast to be useful for Discord/chat.

    Rejects near-white, near-black, and near-constant images (common bad graph crops).
    """
    if arr is None:
        return False
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[-1] < 3 or a.size == 0:
        return False
    rgb = a[..., :3]
    if rgb.dtype != np.uint8:
        mx = float(np.nanmax(rgb)) if rgb.size else 0.0
        if mx <= 1.0 + 1e-6:
            rgb = (np.clip(rgb.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    flat = rgb.astype(np.float32).ravel()
    mean = float(np.nanmean(flat))
    std = float(np.nanstd(flat))
    if mean >= max_mean_white or mean <= min_mean_black:
        return False
    if std < min_std:
        return False
    return True
