# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Instance and InstanceView restored from home_robot_v2 (emet.mapping.instance.core).
InstanceMemory remains a minimal stub so controller/stretch imports work without
pulling full instance_map dependencies. Add debug via __repr__ and optional logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from emet.utils.point_cloud_torch import get_bounds

logger = logging.getLogger(__name__)


@dataclass
class InstanceView:
    """
    Stores information about a single view of a single instance.
    Restored from home_robot_v2 mapping/instance/core.py.
    """

    # Required: 2D and 3D bbox
    bbox: Tensor
    """[2,2] or [4,] bounding box of instance in the current image"""
    bounds: Tensor
    """[3, 2] xyz mins and maxes"""
    timestep: int
    """Timestep at which this view was recorded"""

    # Optional view/description
    text_description: Optional[str] = None
    cropped_image: Optional[Tensor] = None
    """Cropped image of instance (can be [C,H,W] or [H,W,C])"""
    embedding: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    image_instance_id: Optional[int] = None
    visual_feat: Optional[Tensor] = None

    # Detection
    global_instance_id: Optional[int] = None
    category_id: Optional[int] = None
    score: Optional[float] = None

    # 3D
    point_cloud: Optional[Tensor] = None
    point_cloud_rgb: Optional[Tensor] = None
    point_cloud_features: Optional[Tensor] = None
    cam_to_world: Optional[Tensor] = None
    """[4,4] camera space to world space"""
    pose: Optional[Tensor] = None
    """Base pose of the robot when this view was collected"""

    def __repr__(self) -> str:
        cam = "present" if self.cam_to_world is not None else "None"
        return (
            f"InstanceView(timestep={self.timestep}, category_id={self.category_id}, "
            f"bounds={getattr(self.bounds, 'shape', None)}, cam_to_world={cam})"
        )

    def get_pose(self) -> Optional[Tensor]:
        """Returns the position from which we captured this instance view."""
        return self.pose

    @cached_property
    def object_coverage(self) -> float:
        if self.mask is None:
            return 0.0
        return float(self.mask.sum()) / max(self.mask.numel(), 1)

    def show(self, backend: str = "folder", **backend_kwargs: Any) -> None:
        assert backend in ["folder"], backend
        self._show_folder(**backend_kwargs)

    def _show_folder(self, folder_path: Optional[Union[Path, str]] = None) -> None:
        import os

        import cv2

        if folder_path is None:
            folder_path = "."
        if self.cropped_image is None or self.mask is None:
            logger.debug("InstanceView._show_folder: cropped_image or mask is None")
            return
        full_image = self.cropped_image
        if full_image.dim() == 3 and full_image.shape[0] == 3:
            full_image = full_image.permute(1, 2, 0)
        full_image = (full_image.float().clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        if full_image.shape[-1] != 3:
            full_image = np.broadcast_to(full_image[..., None], (*full_image.shape, 3))
        mask_np = self.mask.cpu().numpy().astype(np.uint8)
        if mask_np.shape != full_image.shape[:2]:
            mask_np = np.broadcast_to(mask_np, full_image.shape[:2])
        mask_vis = np.zeros_like(full_image)
        mask_vis[:, :] = (0, 0, 255)
        mask_vis = cv2.bitwise_and(mask_vis, mask_vis, mask=mask_np)
        masked_image = cv2.addWeighted(mask_vis, 1, full_image, 1, 0)
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(
            f"{folder_path}/{self.timestep}_{self.image_instance_id}_cat_{self.category_id}.png",
            cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR),
        )

    def get_image(self) -> np.ndarray:
        """Convert image to showable numpy [H,W,3] uint8."""
        if self.cropped_image is None:
            return np.zeros((32, 32, 3), dtype=np.uint8)
        img = self.cropped_image.cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if img.shape[-1] != 3:
            img = np.broadcast_to(img[..., None], (*img.shape, 3))
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _dummy_instance_view() -> InstanceView:
    """Return a minimal InstanceView when an instance has no views (for get_best_view)."""
    return InstanceView(
        bbox=torch.zeros(2, 2),
        bounds=torch.zeros(3, 2),
        timestep=-1,
    )


@dataclass
class Instance:
    """
    A single instance found in the environment, composed of a list of InstanceView objects.
    Restored from home_robot_v2 mapping/instance/core.py.
    """

    name: Optional[str] = None
    global_id: Optional[int] = None
    category_id: Optional[int] = None
    point_cloud: Optional[Tensor] = None
    point_cloud_rgb: Optional[Tensor] = None
    point_cloud_features: Optional[Tensor] = None
    bounds: Optional[Tensor] = None
    instance_views: List[InstanceView] = field(default_factory=list)
    score: Optional[float] = None
    score_aggregation_method: str = "max"

    def __repr__(self) -> str:
        n_views = len(self.instance_views)
        return (
            f"Instance(global_id={self.global_id}, category_id={self.category_id}, "
            f"views={n_views}, bounds={getattr(self.bounds, 'shape', None)})"
        )

    @property
    def id(self) -> int:
        """Unique global id of the instance."""
        return self.global_id if self.global_id is not None else -1

    def get_category_id(self) -> Optional[int]:
        if self.category_id is None:
            return None
        if isinstance(self.category_id, torch.Tensor):
            return int(self.category_id.item())
        return int(self.category_id)

    def get_image_embedding(
        self,
        aggregation_method: str = "max",
        normalize: bool = True,
        use_visual_feat: bool = False,
    ) -> Any:
        """Combined image embedding across all views."""
        if use_visual_feat:
            view_embeddings = [v.visual_feat for v in self.instance_views]
        else:
            view_embeddings = [v.embedding for v in self.instance_views]
        view_embeddings = [e for e in view_embeddings if e is not None]
        if len(view_embeddings) == 0:
            return None
        view_embeddings_t = torch.cat(view_embeddings, dim=0)
        if aggregation_method == "concatenate":
            emb = view_embeddings_t
        elif aggregation_method == "max":
            emb = view_embeddings_t.max(dim=0).values
        elif aggregation_method == "mean":
            emb = view_embeddings_t.mean(dim=0)
        else:
            raise RuntimeError(
                f"Unsupported aggregation method {aggregation_method}. Options: max, mean."
            )
        if normalize and emb is not None:
            emb = emb / (emb.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        return emb

    def get_best_view(self, metric: str = "area") -> InstanceView:
        """Get best view by area or update_time. Returns dummy view if no views (with debug log)."""
        if not self.instance_views:
            logger.debug("Instance.get_best_view: no instance_views for global_id=%s", self.global_id)
            return _dummy_instance_view()
        best_view: Optional[InstanceView] = None
        if metric == "area":
            best_area = 0.0
            for view in self.instance_views:
                if view.bbox is not None and view.bbox.numel() >= 4:
                    if view.bbox.dim() == 2:
                        area = float(
                            (view.bbox[1, 1] - view.bbox[0, 1])
                            * (view.bbox[1, 0] - view.bbox[0, 0])
                        )
                    else:
                        area = 0.0
                elif view.cropped_image is not None:
                    h, w = view.cropped_image.shape[:2] if view.cropped_image.dim() >= 2 else (0, 0)
                    area = h * w
                else:
                    continue
                if area > best_area:
                    best_area = area
                    best_view = view
        elif metric == "update_time":
            best_view = self.instance_views[-1]
        else:
            raise NotImplementedError(f"metric {metric!r} not supported")
        return best_view if best_view is not None else _dummy_instance_view()

    def get_instance_id(self) -> Optional[int]:
        return self.global_id

    def get_center(self) -> Optional[Tensor]:
        """Center of the instance in 3D (mean xy, actual mean z)."""
        if self.point_cloud is None or self.point_cloud.numel() == 0:
            return None
        xyz = self.point_cloud.mean(dim=0)
        xy = xyz[:2]
        dists = torch.norm(self.point_cloud[:, :2] - xy, dim=1)
        idx = dists.argmin()
        center = self.point_cloud[idx].clone()
        center[2] = xyz[2]
        return center

    def get_median(self) -> Optional[Tensor]:
        if self.point_cloud is None or self.point_cloud.numel() == 0:
            return None
        return self.point_cloud.median(dim=0).values

    def get_closest_point(self, xyz: Union[Tensor, np.ndarray]) -> Optional[Tensor]:
        if self.point_cloud is None or self.point_cloud.numel() == 0:
            return None
        if isinstance(xyz, np.ndarray):
            xyz = torch.as_tensor(xyz, device=self.point_cloud.device, dtype=self.point_cloud.dtype)
        dists = torch.norm(self.point_cloud - xyz, dim=1)
        return self.point_cloud[dists.argmin()]

    def show_best_view(
        self,
        metric: str = "area",
        title: Optional[str] = None,
    ) -> None:
        """Show the best view (cv2.imshow). No-op if no views or headless."""
        best_view = self.get_best_view(metric=metric)
        if best_view.timestep < 0:
            logger.debug("Instance.show_best_view: no views to show for global_id=%s", self.global_id)
            return
        try:
            import cv2

            image = best_view.get_image()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            win = title or f"Instance {self.global_id}"
            cv2.imshow(win, image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            logger.debug("Instance.show_best_view failed (e.g. no display): %s", e)

    def add_instance_view(self, instance_view: InstanceView) -> None:
        if len(self.instance_views) == 0:
            self.category_id = instance_view.category_id
            self.instance_views.append(instance_view)
            self.bounds = instance_view.bounds
            self.point_cloud = instance_view.point_cloud
            self.point_cloud_rgb = instance_view.point_cloud_rgb
            self.point_cloud_features = instance_view.point_cloud_features
            self.score = instance_view.score
        else:
            if instance_view.point_cloud is not None and self.point_cloud is not None:
                self.point_cloud = torch.cat([self.point_cloud, instance_view.point_cloud], dim=0)
            if instance_view.point_cloud_rgb is not None and self.point_cloud_rgb is not None:
                self.point_cloud_rgb = torch.cat(
                    [self.point_cloud_rgb, instance_view.point_cloud_rgb], dim=0
                )
            if (
                instance_view.point_cloud_features is not None
                and self.point_cloud_features is not None
            ):
                self.point_cloud_features = torch.cat(
                    [self.point_cloud_features, instance_view.point_cloud_features], dim=0
                )
            if self.score is None:
                self.score = instance_view.score
            elif self.score_aggregation_method == "max" and instance_view.score is not None:
                self.score = max(self.score, instance_view.score)
            elif self.score_aggregation_method == "mean" and instance_view.score is not None:
                n = len(self.instance_views)
                self.score = (self.score * n + instance_view.score) / (n + 1)
            self.instance_views.append(instance_view)
            if self.point_cloud is not None and self.point_cloud.numel() > 0:
                self.bounds = get_bounds(self.point_cloud)


class InstanceMemory:
    """
    Minimal instance memory stub: per-env dict of instances.
    process_instances_for_env / associate_instances_to_memory are no-ops so controller
    and stretch imports work without full instance_map (InstanceMemory) dependencies.
    """

    def __init__(self, num_envs: int = 1, encoder: Any = None, **kwargs: Any) -> None:
        self.num_envs = num_envs
        self.encoder = encoder
        self.instances: dict[int, dict[int, Instance]] = {i: {} for i in range(num_envs)}

    def __len__(self) -> int:
        """Total number of instances across all envs (for len(self) / voxel_map compatibility)."""
        return sum(len(env) for env in self.instances.values())

    def reset(self) -> None:
        for i in range(self.num_envs):
            self.instances[i] = {}

    def process_instances_for_env(
        self,
        env_id: int = 0,
        instance_seg: Optional[Tensor] = None,
        point_cloud: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        cam_to_world: Optional[Tensor] = None,
        instance_classes: Optional[Tensor] = None,
        instance_scores: Optional[Tensor] = None,
        background_instance_labels: Optional[List[int]] = None,
        valid_points: Optional[Tensor] = None,
        pose: Any = None,
        **kwargs: Any,
    ) -> None:
        """No-op: minimal impl so voxel/mapping code paths run."""
        pass

    def associate_instances_to_memory(self) -> None:
        """No-op."""
        pass

    def global_box_compression_and_nms(self, env_id: int = 0, **kwargs: Any) -> Any:
        """No-op; return empty list for compatibility."""
        return []

    def pop_global_instance(
        self,
        env_id: int = 0,
        global_instance_id: int = 0,
        skip_reindex: bool = False,
    ) -> Optional[Instance]:
        """Remove and return one instance from the env dict."""
        return self.instances[env_id].pop(global_instance_id, None)
