# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Build GraphEQA labels and 3D anchors from robot Observations (RGB-D + pose).
# Uses a vision-language model for open-vocabulary object names; world xyz from
# depth median in camera frame transformed by camera_pose (no sim ground truth).

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from emet.core.interfaces import Observations
from emet.utils.logger import Logger

logger = Logger(__name__)


def world_xyz_median_from_depth(obs: Observations) -> np.ndarray:
    """
    Robust scene point in world frame: median of valid head-camera depth points.
    Falls back to camera optical center if depth or intrinsics missing.
    """
    if obs.camera_pose is None:
        raise ValueError("Observations.camera_pose is required for world xyz")

    if obs.depth is None or obs.camera_K is None:
        return np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)

    obs.compute_xyz(scaling=1.0)
    pts_world = obs.get_xyz_in_world_frame(scaling=1.0)
    if pts_world is None:
        return np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)

    flat = pts_world.reshape(-1, 3)
    d = obs.depth.reshape(-1)
    valid = (d > 0.05) & (d < 10.0) & np.isfinite(d)
    sel = flat[valid]
    if sel.shape[0] == 0:
        return np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)
    return np.median(sel, axis=0).astype(np.float64)


def parse_comma_separated_labels(text: str, max_labels: int = 16) -> List[str]:
    """Parse model output into a clean label list."""
    cleaned = (
        text.replace(";", ",")
        .replace("\n", ",")
        .replace(".", ",")
    )
    out: List[str] = []
    for part in cleaned.split(","):
        s = part.strip().strip("-•").strip()
        if not s or len(s) > 64:
            continue
        out.append(s)
        if len(out) >= max_labels:
            break
    return out


class SensorGraphBuilder:
    """
    Produces object labels (VLM) and a world-frame anchor xyz from Observations.

    If ``cpu_only`` or no GPU, skips loading Qwen2.5-VL and relies on
    ``voxel_labels`` / fallback ``["object"]``.
    """

    def __init__(
        self,
        *,
        perception_client: Optional[Callable[..., str]] = None,
        use_voxel_fallback: bool = True,
        device: str = "cuda",
        cpu_only: bool = False,
    ):
        self._perception = perception_client
        self.use_voxel_fallback = use_voxel_fallback
        self._device = device
        self.cpu_only = cpu_only
        self._lazy_vl_client: Optional[Callable[..., str]] = None

    def _get_default_vl_client(self) -> Optional[Callable[..., str]]:
        if self.cpu_only:
            return None
        try:
            from emet.llms.qwen_client import Qwen25VLClient
        except ImportError as e:
            logger.warning(f"Qwen25VLClient unavailable ({e}); using voxel/fallback labels")
            return None
        dev = self._device
        if dev not in ("cuda", "mps"):
            dev = "cuda"
        try:
            return Qwen25VLClient(
                prompt=None,
                model_size="3B",
                quantization="int4",
                max_tokens=128,
                num_beams=1,
                device=dev,
            )
        except Exception as e:
            logger.warning(f"Could not load Qwen2.5-VL ({e}); using voxel/fallback labels")
            return None

    def _client(self) -> Optional[Callable[..., str]]:
        if self._perception is not None:
            return self._perception
        if self.cpu_only:
            return None
        if self._lazy_vl_client is None:
            self._lazy_vl_client = self._get_default_vl_client()
        return self._lazy_vl_client

    def labels_from_observation(
        self,
        obs: Observations,
        voxel_labels: Optional[List[str]] = None,
    ) -> List[str]:
        client = self._client()
        if client is None:
            if voxel_labels:
                return list(voxel_labels)
            return ["object"]

        prompt = (
            "List visible distinct objects in this indoor scene. "
            "Reply with comma-separated short nouns only (max 12), no sentences."
        )
        try:
            out = client([prompt, obs.rgb])
            if not isinstance(out, str):
                out = str(out)
            labels = parse_comma_separated_labels(out)
            if labels:
                return labels
        except Exception as e:
            logger.warning(f"Perception VLM failed ({e})")

        if voxel_labels and self.use_voxel_fallback:
            return list(voxel_labels)
        return ["object"]

    def world_xyz_for_observation(self, obs: Observations) -> np.ndarray:
        return world_xyz_median_from_depth(obs)
