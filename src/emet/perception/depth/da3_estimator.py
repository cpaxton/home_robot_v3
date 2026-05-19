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
import torch

from emet.utils.logger import Logger

logger = Logger(__name__)

_MISSING = object()


def apply_da3_sky_row_mask(depth: np.ndarray, fraction_top: float) -> np.ndarray:
    """Zero the top *fraction_top* of image rows in a depth map (HxW).

    Textureless sky/ceiling often gets spurious finite depths from DA3 stereo; unprojection then
    paints tall vertical sheets. Zeros are removed downstream by ``min_depth`` filtering in voxel code.
    """
    if fraction_top <= 0.0 or depth.ndim != 2:
        return depth
    h = int(depth.shape[0])
    n = int(round(float(fraction_top) * h))
    if n <= 0:
        return depth
    out = np.array(depth, dtype=np.float32, copy=True)
    out[:n, :] = 0.0
    return out


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
        model_id: str = "depth-anything/DA3-SMALL",
        device: str = "cuda",
        process_res: int = 378,
        clip_output_max_m: float = 6.0,
        use_amp: bool = False,
        torch_compile: bool = False,
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
        self._clip_output_max_m = float(clip_output_max_m)
        self._use_amp = bool(use_amp)
        self._torch_compile = bool(torch_compile)
        self._torch_compile_tried = False
        self._model: Any = None

    def _maybe_torch_compile(self) -> None:
        if not self._torch_compile or self._torch_compile_tried or self._model is None:
            return
        self._torch_compile_tried = True
        try:
            self._model = torch.compile(self._model)  # type: ignore[assignment]
            logger.info("DA3: torch.compile enabled on the loaded model.")
        except Exception as e:
            logger.warning("DA3: torch.compile failed (%s); using eager inference.", e)

    def _model_inference(self, **kwargs: Any) -> Any:
        use_cuda_amp = (
            self._use_amp
            and str(self._device).startswith("cuda")
            and torch.cuda.is_available()
        )
        if use_cuda_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._model.inference(**kwargs)
        if self._use_amp and str(self._device).startswith("mps"):
            # MPS supports float16 autocast; omit if unsupported on older torch.
            try:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    return self._model.inference(**kwargs)
            except Exception as e:
                logger.warning("DA3: MPS autocast failed (%s); using eager inference.", e)
        return self._model.inference(**kwargs)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading Depth Anything 3 model {self._model_id!r} on {self._device!r}…")
        self._model = self._DepthAnything3.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()
        self._maybe_torch_compile()

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

        pred = self._model_inference(**kwargs)
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=self._clip_output_max_m, neginf=0.0)
        depth = np.clip(depth, 0.0, self._clip_output_max_m)
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
                pred = self._model_inference(**kwargs)
                depth = np.asarray(pred.depth[0], dtype=np.float32)
                depth = np.nan_to_num(depth, nan=0.0, posinf=self._clip_output_max_m, neginf=0.0)
                depth = np.clip(depth, 0.0, self._clip_output_max_m)
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
    model_id = str(parameters.get("da3_model_id", "depth-anything/DA3-SMALL"))
    process_res = int(parameters.get("da3_process_res", 378))
    da3_dev = parameters.get("da3_device", None)
    dev = str(da3_dev).strip() if da3_dev is not None else device
    md = float(parameters.get("max_depth", 2.5))
    clip_raw = parameters.get("da3_clip_max_m", None)
    clip_m = float(clip_raw) if clip_raw is not None else max(md + 1.0, 4.0)
    raw_amp = parameters.get("da3_use_amp", _MISSING)
    if raw_amp is _MISSING:
        use_amp = str(dev).startswith("cuda") and torch.cuda.is_available()
    else:
        use_amp = bool(raw_amp)
    torch_compile = bool(parameters.get("da3_torch_compile", False))
    return DA3DepthEstimator(
        model_id=model_id,
        device=dev,
        process_res=process_res,
        clip_output_max_m=clip_m,
        use_amp=use_amp,
        torch_compile=torch_compile,
    )


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
    *,
    da3_use_stereo: bool = False,
) -> np.ndarray | None:
    """Resolve depth the same way as :meth:`DynamemController._resolve_depth_map` (sensor / da3 / auto).

    Shared with CLI debug tooling so visualization matches mapping.

    When *da3_use_stereo* is false, DA3 always uses monocular :meth:`DA3DepthEstimator.infer` on *rgb* even if
    stereo RGB and intrinsics are present (dynav ``da3_stereo: false``).
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
        bool(da3_use_stereo)
        and rgb_r_arr is not None
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


def resolve_depth_map_uses_observation_sensor_only(
    depth_source: str,
    sensor_depth: np.ndarray | None,
) -> bool:
    """True when :func:`resolve_depth_map` returns raw ``sensor_depth`` (not DA3 / stereo inference).

    Used to skip ``da3_ignore_sky_fraction_top`` masking on real / simulator depth.
    """
    mode = str(depth_source).lower()
    if mode == "sensor":
        return True
    if mode == "auto" and sensor_depth is not None and np.asarray(sensor_depth).size > 0:
        return True
    return False
