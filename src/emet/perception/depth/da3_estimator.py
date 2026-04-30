# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Depth Anything 3 inference for DynaMem (install optional extra: ``pip install -e '.[da3]'``)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from emet.utils.logger import Logger

logger = Logger(__name__)


def resize_depth_to_match_rgb(depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Resize a depth map to ``rgb.shape[:2]`` using linear interpolation."""
    if depth.shape[0] == rgb.shape[0] and depth.shape[1] == rgb.shape[1]:
        return depth.astype(np.float32, copy=False)
    return cv2.resize(
        depth.astype(np.float32),
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


class DA3DepthEstimator:
    """Runs Depth Anything 3 and returns metric-ish depth in meters (H, W) float32."""

    def __init__(
        self,
        *,
        model_id: str = "depth-anything/DA3METRIC-LARGE",
        device: str = "cuda",
        process_res: int = 504,
    ) -> None:
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "depth_anything_3 is required for DA3 depth. Install with: uv sync --extra da3 "
                "or pip install 'depth-anything-3>=0.1.1'"
            ) from e

        self._DepthAnything3 = DepthAnything3
        self._model_id = model_id
        self._device = device
        self._process_res = int(process_res)
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading Depth Anything 3 model {self._model_id!r} on {self._device!r}…")
        self._model = self._DepthAnything3.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()

    def infer(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: np.ndarray | None = None,
        extrinsics_w2c: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return depth in meters, same height/width as *rgb* (float32)."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3 uint8/float; got {rgb.shape}")
        self._ensure_model()
        kwargs: dict[str, Any] = {
            "image": [rgb],
            "process_res": self._process_res,
        }
        if intrinsics is not None and extrinsics_w2c is not None:
            k = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
            ext = np.asarray(extrinsics_w2c, dtype=np.float32).reshape(4, 4)
            kwargs["intrinsics"] = k.reshape(1, 3, 3)
            kwargs["extrinsics"] = ext.reshape(1, 4, 4)
            kwargs["align_to_input_ext_scale"] = True

        pred = self._model.inference(**kwargs)
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth, 0.0, None)
        return resize_depth_to_match_rgb(depth, rgb)


def create_da3_estimator_from_parameters(parameters: Any, *, device: str) -> DA3DepthEstimator | None:
    """Build a DA3 estimator from DynaMem-style ``Parameters`` (or dict-like)."""
    src = str(parameters.get("depth_source", "sensor")).lower()
    if src not in ("da3", "auto"):
        return None
    model_id = str(parameters.get("da3_model_id", "depth-anything/DA3METRIC-LARGE"))
    process_res = int(parameters.get("da3_process_res", 504))
    da3_dev = parameters.get("da3_device", None)
    dev = str(da3_dev).strip() if da3_dev is not None else device
    return DA3DepthEstimator(model_id=model_id, device=dev, process_res=process_res)
