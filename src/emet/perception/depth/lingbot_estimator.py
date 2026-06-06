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

"""LingBot-Map depth + pose estimator for DynaMem (subprocess in .venv-lingbot-map)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from emet.perception.depth.da3_estimator import resize_depth_to_match_rgb
from emet.perception.depth.lingbot_subprocess import LingBotRollingBuffer, LingBotSubprocessClient
from emet.utils.logger import Logger

logger = Logger(__name__)


def create_lingbot_estimator_from_parameters(parameters: dict[str, Any]) -> LingBotRollingBuffer:
    import os

    ckpt = (
        parameters.get("lingbot_checkpoint")
        or parameters.get("lingbot_model_path")
        or os.environ.get("LINGBOT_MAP_CHECKPOINT")
    )
    if not ckpt:
        raise ValueError("lingbot checkpoint missing: set lingbot_checkpoint in YAML or LINGBOT_MAP_CHECKPOINT")
    infer_every_n = int(parameters.get("lingbot_infer_every_n", 2) or 2)
    keyframe_interval = parameters.get("lingbot_keyframe_interval")
    kf = int(keyframe_interval) if keyframe_interval is not None else 2
    sdpa_env = os.environ.get("LINGBOT_MAP_USE_SDPA", "1").strip().lower()
    use_sdpa_default = sdpa_env not in ("0", "false", "no", "off")
    use_sdpa = bool(parameters.get("lingbot_use_sdpa", use_sdpa_default))
    client = LingBotSubprocessClient(
        checkpoint=Path(str(ckpt)) if ckpt else None,
        keyframe_interval=kf,
        use_sdpa=use_sdpa,
    )
    return LingBotRollingBuffer(client, infer_every_n=infer_every_n)


class LingBotDepthEstimator:
    """Stateful wrapper around :class:`LingBotRollingBuffer`."""

    def __init__(self, buffer: LingBotRollingBuffer, *, use_lingbot_pose: bool = True) -> None:
        self._buffer = buffer
        self.use_lingbot_pose = use_lingbot_pose
        self.last_camera_pose: np.ndarray | None = None
        self.last_camera_K: np.ndarray | None = None

    def infer(
        self,
        rgb: np.ndarray,
        *,
        camera_K: np.ndarray | None = None,
        camera_pose: np.ndarray | None = None,
        force: bool = False,
    ) -> np.ndarray | None:
        depth, pose, K = self._buffer.update(
            rgb,
            camera_K=camera_K,
            camera_pose=camera_pose,
            force=force,
        )
        if self.use_lingbot_pose and pose is not None:
            self.last_camera_pose = pose
        if K is not None:
            self.last_camera_K = K
        if depth is None:
            return None
        return resize_depth_to_match_rgb(depth, rgb)

    def close(self) -> None:
        self._buffer.cleanup()
