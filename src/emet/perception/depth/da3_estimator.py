# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Depth Anything 3 inference for DynaMem (``depth-anything-3`` is a core dependency)."""

from __future__ import annotations

import sys
import types
from typing import Any

import cv2
import numpy as np

from emet.utils.logger import Logger

logger = Logger(__name__)


def _stub_gsplat_if_missing() -> None:
    """``depth_anything_3.api`` imports export helpers that load ``gs_renderer``, which tries ``import gsplat``.

    That optional dependency is only for 3D Gaussian splatting export/rendering—not for depth inference.
    Pre-register a minimal stub so import succeeds without ByteDance's warning spam when gsplat is omitted.
    """
    try:
        import gsplat  # noqa: F401
        return
    except ImportError:
        pass
    if "gsplat" in sys.modules:
        return
    stub = types.ModuleType("gsplat")

    def rasterization(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError(
            "gsplat is not installed; Depth Anything 3 Gaussian splatting export is unavailable."
        )

    stub.rasterization = rasterization  # type: ignore[attr-defined]
    sys.modules["gsplat"] = stub


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
        _stub_gsplat_if_missing()
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "depth_anything_3 is required for DA3 depth. Install dependencies with `uv sync` "
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

    def infer_stereo(
        self,
        rgb_left: np.ndarray,
        rgb_right: np.ndarray,
        *,
        intrinsics_left: np.ndarray | None = None,
        extrinsics_w2c_left: np.ndarray | None = None,
        intrinsics_right: np.ndarray | None = None,
        extrinsics_w2c_right: np.ndarray | None = None,
    ) -> np.ndarray:
        """Two-view DA3; returns depth aligned to *rgb_left* (reference view = first).

        Some checkpoints (e.g. DA3METRIC-LARGE) omit ``cam_enc`` in the network; passing stacked
        intrinsics/extrinsics then crashes inside ``depth_anything_3``. We retry without camera
        tensors, then fall back to monocular depth on the left image.
        """
        if rgb_left.shape[:2] != rgb_right.shape[:2]:
            raise ValueError("Stereo RGB frames must match in HxW.")
        self._ensure_model()

        kwargs_images_only: dict[str, Any] = {
            "image": [rgb_left, rgb_right],
            "process_res": self._process_res,
            "ref_view_strategy": "first",
        }
        kwargs_with_cam: dict[str, Any] = dict(kwargs_images_only)
        have_cam = (
            intrinsics_left is not None
            and extrinsics_w2c_left is not None
            and intrinsics_right is not None
            and extrinsics_w2c_right is not None
        )
        if have_cam:
            kl = np.asarray(intrinsics_left, dtype=np.float32).reshape(3, 3)
            kr = np.asarray(intrinsics_right, dtype=np.float32).reshape(3, 3)
            el = np.asarray(extrinsics_w2c_left, dtype=np.float32).reshape(4, 4)
            er = np.asarray(extrinsics_w2c_right, dtype=np.float32).reshape(4, 4)
            kwargs_with_cam["intrinsics"] = np.stack([kl, kr], axis=0)
            kwargs_with_cam["extrinsics"] = np.stack([el, er], axis=0)
            kwargs_with_cam["align_to_input_ext_scale"] = True

        # Prefer two-view RGB without poses first: DA3METRIC-LARGE often has cam_enc=None and crashes
        # when passing stacked intrinsics/extrinsics (see depth_anything_3.model.da3.forward).
        attempts: list[tuple[str, dict[str, Any]]] = [("stereo_RGB_only", kwargs_images_only)]
        if have_cam:
            attempts.append(("stereo+intrinsics+extrinsics", kwargs_with_cam))

        last_err: BaseException | None = None
        for label, kwargs in attempts:
            try:
                pred = self._model.inference(**kwargs)
                depth = np.asarray(pred.depth[0], dtype=np.float32)
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                depth = np.clip(depth, 0.0, None)
                return resize_depth_to_match_rgb(depth, rgb_left)
            except (TypeError, RuntimeError) as e:
                last_err = e
                logger.warning("DA3 %s inference failed (%s); trying fallback.", label, e)

        logger.warning(
            "DA3 stereo unavailable (%s); using monocular depth on the left image.",
            last_err,
        )
        return self.infer(
            rgb_left,
            intrinsics=intrinsics_left,
            extrinsics_w2c=extrinsics_w2c_left,
        )


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


def resolve_depth_map(
    depth_source: str,
    est: DA3DepthEstimator | None,
    rgb: np.ndarray,
    sensor_depth: np.ndarray | None,
    camera_K: np.ndarray | None,
    camera_pose: np.ndarray | None,
    rgb_right: np.ndarray | None = None,
    camera_K_right: np.ndarray | None = None,
    camera_pose_right: np.ndarray | None = None,
) -> np.ndarray | None:
    """Resolve depth the same way as :meth:`DynamemController._resolve_depth_map` (sensor / da3 / auto).

    Shared with CLI debug tooling so visualization matches mapping.
    """
    mode = str(depth_source).lower()
    if mode == "sensor":
        return sensor_depth

    k_ok = camera_K is not None and np.asarray(camera_K).shape == (3, 3)
    p_ok = camera_pose is not None and np.asarray(camera_pose).shape == (4, 4)
    k_use = np.asarray(camera_K, dtype=np.float32) if k_ok else None
    p_use = np.asarray(camera_pose, dtype=np.float32) if p_ok else None

    kr_ok = camera_K_right is not None and np.asarray(camera_K_right).shape == (3, 3)
    pr_ok = camera_pose_right is not None and np.asarray(camera_pose_right).shape == (4, 4)
    k_r = np.asarray(camera_K_right, dtype=np.float32) if kr_ok else None
    p_r = np.asarray(camera_pose_right, dtype=np.float32) if pr_ok else None

    if mode == "auto":
        if sensor_depth is not None and np.asarray(sensor_depth).size > 0:
            return np.asarray(sensor_depth, dtype=np.float32)

    if est is None:
        raise RuntimeError("depth_source requires DA3 but estimator is None.")

    rgb_r_arr = np.asarray(rgb_right) if rgb_right is not None else None
    use_stereo = (
        rgb_r_arr is not None
        and rgb_r_arr.size > 0
        and k_use is not None
        and p_use is not None
        and k_r is not None
        and p_r is not None
    )
    if use_stereo and rgb.shape[:2] == rgb_r_arr.shape[:2]:
        return est.infer_stereo(
            rgb,
            rgb_r_arr,
            intrinsics_left=k_use,
            extrinsics_w2c_left=p_use,
            intrinsics_right=k_r,
            extrinsics_w2c_right=p_r,
        )

    if mode == "da3":
        return est.infer(rgb, intrinsics=k_use, extrinsics_w2c=p_use)

    return est.infer(rgb, intrinsics=k_use, extrinsics_w2c=p_use)
