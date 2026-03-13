# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# Minimal Instance, InstanceView, InstanceMemory so mapping/controller/voxel imports run.
# See docs/plans/MAPPING_REFACTOR.md. process_instances_for_env is a no-op unless extended.

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class InstanceView:
    """Minimal view of an instance (cropped image, mask, bounds)."""

    def __init__(
        self,
        bounds: torch.Tensor | None = None,
        cropped_image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        score: float = 0.0,
    ):
        self.bounds = bounds if bounds is not None else torch.zeros(2, 3)
        self.cropped_image = cropped_image if cropped_image is not None else torch.zeros(3, 32, 32)
        self.mask = mask if mask is not None else torch.zeros(1, 32, 32)
        self.score = score

    def get_pose(self) -> np.ndarray:
        return np.eye(4)

    def get_image(self) -> np.ndarray:
        return self.cropped_image.permute(1, 2, 0).detach().cpu().numpy()


class Instance:
    """Minimal detected object instance (global_id, point_cloud, bounds, views)."""

    def __init__(
        self,
        global_id: int = 0,
        instance_id: int | None = None,
        point_cloud: torch.Tensor | None = None,
        point_cloud_rgb: torch.Tensor | None = None,
        bounds: torch.Tensor | None = None,
        category_id: int = -1,
    ):
        self.global_id = global_id
        self.id = instance_id if instance_id is not None else global_id
        self.point_cloud = point_cloud if point_cloud is not None else torch.zeros(0, 3)
        self.point_cloud_rgb = point_cloud_rgb if point_cloud_rgb is not None else torch.zeros(0, 3)
        self.bounds = bounds if bounds is not None else torch.zeros(2, 3)
        self.category_id = category_id
        self._views: list[InstanceView] = []

    def get_best_view(self) -> InstanceView:
        if self._views:
            return self._views[0]
        return InstanceView()

    def show_best_view(self) -> None:
        pass


class InstanceMemory:
    """Minimal instance memory: per-env dict of instances; process/associate are no-ops."""

    def __init__(self, num_envs: int = 1, encoder: Any = None, **kwargs: Any):
        self.num_envs = num_envs
        self.encoder = encoder
        self.instances: dict[int, dict[int, Instance]] = {i: {} for i in range(num_envs)}

    def reset(self) -> None:
        for i in range(self.num_envs):
            self.instances[i] = {}

    def process_instances_for_env(
        self,
        env_id: int = 0,
        instance_seg: torch.Tensor | None = None,
        point_cloud: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        cam_to_world: torch.Tensor | None = None,
        instance_classes: torch.Tensor | None = None,
        instance_scores: torch.Tensor | None = None,
        background_instance_labels: list[int] | None = None,
        valid_points: torch.Tensor | None = None,
        pose: Any | None = None,
    ) -> None:
        """No-op: minimal impl so voxel/mapping code paths run."""
        pass

    def associate_instances_to_memory(self) -> None:
        """No-op."""
        pass

    def global_box_compression_and_nms(self, env_id: int = 0) -> None:
        """No-op."""
        pass

    def pop_global_instance(self, env_id: int = 0, global_instance_id: int = 0) -> None:
        """Remove one instance from the env dict."""
        self.instances[env_id].pop(global_instance_id, None)
