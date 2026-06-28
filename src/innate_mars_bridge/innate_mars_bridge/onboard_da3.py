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

"""Optional Depth Anything 3 on the Mars Jetson (publish metric depth over ZMQ).

When ``EMET_MARS_ONBOARD_DA3=1`` and ``depth-anything-3`` + ``torch`` are installed on the robot,
the ZMQ bridge runs stereo/mono DA3 on head cameras and fills the ``depth`` field in full
observations. Workstations using ``depth_source: auto`` (``dynav_innate_mars.yaml``) then skip
local DA3 and save GPU compute.

Deploy: ``emet deploy --with-da3`` or ``emet mars start --deploy --onboard-da3``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_ONBOARD_ENV = "EMET_MARS_ONBOARD_DA3"


def onboard_da3_enabled() -> bool:
    return os.environ.get(_ONBOARD_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


class OnboardDA3Depth:
    """Lazy DA3 wrapper for innate_mars_bridge (uses ``emet.perception.depth`` when on PYTHONPATH)."""

    def __init__(self) -> None:
        self._estimator: Any = None
        self._step = 0
        self._infer_every_n = max(1, _env_int("EMET_MARS_DA3_INFER_EVERY_N", 2))
        self._use_stereo = os.environ.get("EMET_MARS_DA3_STEREO", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._sky_fraction = _env_float("EMET_MARS_DA3_IGNORE_SKY_FRACTION_TOP", 0.16)
        self._speckle_open_kernel = _env_int("EMET_MARS_DA3_SPECKLE_OPEN_KERNEL", 3)
        self._speckle_open_iterations = _env_int("EMET_MARS_DA3_SPECKLE_OPEN_ITERATIONS", 1)
        self._last_depth: np.ndarray | None = None
        self._load_error: str | None = None

    def _lazy_estimator(self) -> Any:
        if self._estimator is not None:
            return self._estimator
        if self._load_error is not None:
            return None
        try:
            from emet.perception.depth.da3_estimator import create_da3_estimator_from_parameters
        except ImportError as exc:
            self._load_error = (
                f"Onboard DA3: emet.perception.depth not importable ({exc}). "
                "Run `emet deploy --with-da3` on the robot checkout."
            )
            return None
        params = {
            "depth_source": "da3",
            "da3_model_id": os.environ.get("EMET_MARS_DA3_MODEL_ID", "depth-anything/DA3-SMALL"),
            "da3_process_res": _env_int("EMET_MARS_DA3_PROCESS_RES", 378),
            "da3_clip_max_m": _env_float("EMET_MARS_DA3_CLIP_MAX_M", 4.0),
            "da3_use_amp": os.environ.get("EMET_MARS_DA3_USE_AMP", "1").strip().lower()
            not in ("0", "false", "no", "off"),
        }
        device = os.environ.get("EMET_MARS_DA3_DEVICE", "cuda")
        self._estimator = create_da3_estimator_from_parameters(params, device=device)
        return self._estimator

    def infer_depth_meters(
        self,
        rgb_left: np.ndarray,
        *,
        rgb_right: np.ndarray | None = None,
        camera_K_left: np.ndarray | None = None,
        camera_pose_left: np.ndarray | None = None,
        camera_K_right: np.ndarray | None = None,
        camera_pose_right: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return H×W float32 depth in meters, or None if DA3 unavailable."""
        est = self._lazy_estimator()
        if est is None:
            return None

        self._step += 1
        run_full = self._infer_every_n <= 1 or (self._step - 1) % self._infer_every_n == 0
        if not run_full and self._last_depth is not None and self._last_depth.shape[:2] == rgb_left.shape[:2]:
            return np.asarray(self._last_depth, dtype=np.float32).copy()

        depth: np.ndarray | None = None
        if (
            self._use_stereo
            and rgb_right is not None
            and rgb_right.size > 0
            and rgb_right.shape[:2] == rgb_left.shape[:2]
        ):
            depth = est.infer_stereo(
                rgb_left,
                rgb_right,
                intrinsics_left=camera_K_left,
                extrinsics_w2c_left=camera_pose_left,
                intrinsics_right=camera_K_right,
                extrinsics_w2c_right=camera_pose_right,
            )
        else:
            depth = est.infer(
                rgb_left,
                intrinsics=camera_K_left,
                extrinsics_w2c=camera_pose_left,
            )

        if depth is None:
            return None

        if self._sky_fraction > 0.0:
            from emet.perception.depth.da3_estimator import apply_da3_sky_row_mask

            depth = apply_da3_sky_row_mask(np.asarray(depth, dtype=np.float32), self._sky_fraction)

        if self._speckle_open_kernel > 0:
            from emet.perception.depth.da3_estimator import apply_depth_speckle_filter

            depth = apply_depth_speckle_filter(
                np.asarray(depth, dtype=np.float32),
                open_kernel=self._speckle_open_kernel,
                open_iterations=self._speckle_open_iterations,
            )

        self._last_depth = np.asarray(depth, dtype=np.float32).copy()
        return self._last_depth

    @property
    def load_error(self) -> str | None:
        return self._load_error


def create_onboard_da3_from_env() -> OnboardDA3Depth | None:
    if not onboard_da3_enabled():
        return None
    return OnboardDA3Depth()
