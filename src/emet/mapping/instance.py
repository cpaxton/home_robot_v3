# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Instance-level memory: tracks segmented objects across observations with bounding boxes,
# point clouds, cropped views, and optional visual embeddings.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class InstanceView:
    """A single observation/crop of an instance from one camera viewpoint."""

    def __init__(
        self,
        cropped_image: Tensor,
        mask: Tensor,
        bbox_xyxy: Tensor,
        cam_to_world: Tensor,
        embedding: Optional[Tensor] = None,
        score: float = 1.0,
        bounds: Optional[Tensor] = None,
        timestep: int = 0,
    ):
        self.cropped_image = cropped_image  # (H_crop, W_crop, 3) uint8 or float
        self.mask = mask  # (H_crop, W_crop) or (H_crop, W_crop, 1) binary
        self.bbox_xyxy = bbox_xyxy  # (4,) in original image coords
        self.cam_to_world = cam_to_world  # (4, 4) camera-to-world transform
        self.embedding = embedding  # (D,) visual embedding (CLIP/SigLIP/DINOv3)
        self.score = score
        self.bounds = bounds  # (3, 2) min/max xyz in world frame
        self.timestep = timestep

    def get_image(self) -> np.ndarray:
        """Return the cropped image as a uint8 numpy HWC array."""
        img = self.cropped_image
        if isinstance(img, Tensor):
            img = img.detach().cpu()
            if img.dtype == torch.float32 or img.dtype == torch.float16:
                img = (img * 255).clamp(0, 255).byte()
            img = img.numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.uint8)

    def get_pose(self) -> Tensor:
        """Return the camera-to-world transform for this view."""
        return self.cam_to_world


class Instance:
    """A tracked object instance in the map, aggregated across multiple views."""

    def __init__(
        self,
        instance_id: int,
        global_id: int,
        category_id: Union[Tensor, int] = -1,
        score: float = 1.0,
    ):
        self.id = instance_id
        self.global_id = global_id
        if isinstance(category_id, int):
            category_id = torch.tensor(category_id, dtype=torch.long)
        self.category_id = category_id
        self.score = score

        self.instance_views: List[InstanceView] = []
        self.point_cloud: Optional[Tensor] = None  # (N, 3) world xyz
        self.point_cloud_rgb: Optional[Tensor] = None  # (N, 3) colors
        self.bounds: Optional[Tensor] = None  # (3, 2) axis-aligned bbox [min, max]

        self._embedding_cache: Optional[Tensor] = None

    def add_view(self, view: InstanceView) -> None:
        """Add an observation of this instance."""
        self.instance_views.append(view)
        self._embedding_cache = None

    def get_best_view(self) -> InstanceView:
        """Return the view with the highest score (or the latest if tied)."""
        if not self.instance_views:
            raise ValueError("Instance has no views")
        return max(self.instance_views, key=lambda v: (v.score, v.timestep))

    def get_instance_id(self) -> int:
        return self.global_id

    def get_center(self) -> Tensor:
        """Mean of the 3D point cloud."""
        if self.point_cloud is None or self.point_cloud.shape[0] == 0:
            return torch.zeros(3)
        return self.point_cloud.mean(dim=0)

    def get_median(self) -> np.ndarray:
        """Median of the 3D point cloud."""
        if self.point_cloud is None or self.point_cloud.shape[0] == 0:
            return np.zeros(3)
        pc = self.point_cloud.detach().cpu().numpy()
        return np.median(pc, axis=0)

    def get_closest_point(self, xyz: np.ndarray) -> np.ndarray:
        """Return the point in the cloud closest to the given position."""
        if self.point_cloud is None or self.point_cloud.shape[0] == 0:
            return np.zeros(3)
        pc = self.point_cloud.detach().cpu().numpy()
        dists = np.linalg.norm(pc - xyz, axis=1)
        return pc[np.argmin(dists)]

    def get_image_embedding(
        self, aggregation_method: str = "mean", normalize: bool = True
    ) -> Tensor:
        """Aggregate visual embeddings across all views."""
        embeddings = [v.embedding for v in self.instance_views if v.embedding is not None]
        if not embeddings:
            return torch.zeros(1)
        stacked = torch.stack(embeddings)
        if aggregation_method == "mean":
            result = stacked.mean(dim=0)
        elif aggregation_method == "max":
            result = stacked.max(dim=0).values
        else:
            result = stacked.mean(dim=0)
        if normalize:
            result = result / (result.norm(dim=-1, keepdim=True) + 1e-8)
        return result

    def show_best_view(self, title: Optional[str] = None) -> None:
        """Display the best view using matplotlib."""
        import matplotlib.pyplot as plt

        view = self.get_best_view()
        img = view.get_image()
        plt.figure()
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()

    def update_bounds(self) -> None:
        """Recompute axis-aligned bounding box from point cloud."""
        if self.point_cloud is not None and self.point_cloud.shape[0] > 0:
            self.bounds = torch.stack(
                [self.point_cloud.min(dim=0).values, self.point_cloud.max(dim=0).values], dim=1
            )

    def __repr__(self) -> str:
        n_views = len(self.instance_views)
        n_pts = self.point_cloud.shape[0] if self.point_cloud is not None else 0
        return (
            f"Instance(id={self.id}, global_id={self.global_id}, "
            f"cat={int(self.category_id.item())}, views={n_views}, points={n_pts})"
        )


class InstanceMemory:
    """Manages a set of object instances across multiple observations.

    Tracks per-frame instance segmentations, associates them to persistent objects
    using IoU-based matching, and maintains 3D point clouds and cropped views.
    """

    def __init__(
        self,
        num_envs: int = 1,
        encoder: Any = None,
        du_scale: int = 1,
        instance_association: str = "bbox_iou",
        log_dir_overwrite_ok: bool = True,
        mask_cropped_instances: str = "False",
        min_pixels_for_instance_view: int = 100,
        min_instance_thickness: float = 0.01,
        min_instance_vol: float = 1e-6,
        max_instance_vol: float = 10.0,
        min_instance_height: float = 0.1,
        max_instance_height: float = 1.8,
        min_percent_for_instance_view: float = 0.2,
        open_vocab_cat_map_file: Optional[str] = None,
        use_visual_feat: bool = False,
    ):
        self.num_envs = num_envs
        self.encoder = encoder
        self.du_scale = du_scale
        self.instance_association = instance_association
        self.mask_cropped_instances = mask_cropped_instances.lower() in ("true", "1", "yes") if isinstance(mask_cropped_instances, str) else bool(mask_cropped_instances)
        self.min_pixels_for_instance_view = min_pixels_for_instance_view
        self.min_instance_thickness = min_instance_thickness
        self.min_instance_vol = min_instance_vol
        self.max_instance_vol = max_instance_vol
        self.min_instance_height = min_instance_height
        self.max_instance_height = max_instance_height
        self.min_percent_for_instance_view = min_percent_for_instance_view
        self.use_visual_feat = use_visual_feat

        self._next_global_id = 0
        self.instances: Dict[int, Dict[int, Instance]] = {}
        self.reset()

    def reset(self) -> None:
        """Clear all tracked instances."""
        self.instances = {env_id: {} for env_id in range(self.num_envs)}
        self._next_global_id = 0
        self._unassociated: Dict[int, List[_PendingInstance]] = {
            env_id: [] for env_id in range(self.num_envs)
        }

    def _allocate_global_id(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    def process_instances_for_env(
        self,
        env_id: int,
        instance_seg: Tensor,
        point_cloud: Tensor,
        image: Tensor,
        cam_to_world: Tensor,
        instance_classes: Optional[Tensor] = None,
        instance_scores: Optional[Tensor] = None,
        background_instance_labels: Optional[List[int]] = None,
        valid_points: Optional[Tensor] = None,
        pose: Optional[Tensor] = None,
    ) -> None:
        """Process a single frame's instance segmentation and stage for association.

        Args:
            env_id: environment index (usually 0)
            instance_seg: (H, W) integer mask with instance IDs
            point_cloud: (H, W, 3) world-frame xyz
            image: (3, H, W) RGB tensor
            cam_to_world: (4, 4) camera pose
            instance_classes: (K,) per-instance category IDs
            instance_scores: (K,) per-instance confidence
            background_instance_labels: labels to ignore (e.g. [-1])
            valid_points: (H, W) bool mask for valid depth
            pose: (3,) robot base pose xyt
        """
        if background_instance_labels is None:
            background_instance_labels = [-1]

        H, W = instance_seg.shape
        unique_ids = instance_seg.unique().tolist()

        pending = []
        for inst_id in unique_ids:
            if inst_id in background_instance_labels:
                continue

            mask_2d = instance_seg == inst_id
            n_pixels = mask_2d.sum().item()
            if n_pixels < self.min_pixels_for_instance_view:
                continue

            total_pixels = H * W
            if n_pixels / total_pixels < self.min_percent_for_instance_view:
                continue

            # Compute 3D points for this instance
            if valid_points is not None:
                inst_valid = mask_2d & valid_points
            else:
                inst_valid = mask_2d
            pts_3d = point_cloud[inst_valid]
            if pts_3d.shape[0] < 3:
                continue

            # Volume/height filtering
            mins = pts_3d.min(dim=0).values
            maxs = pts_3d.max(dim=0).values
            extent = maxs - mins
            vol = extent.prod().item()
            height = extent[2].item()
            thickness = extent[:2].min().item()

            if vol < self.min_instance_vol or vol > self.max_instance_vol:
                continue
            if height < self.min_instance_height or height > self.max_instance_height:
                continue
            if thickness < self.min_instance_thickness:
                continue

            # Crop the image
            ys, xs = torch.where(mask_2d)
            y0, y1 = ys.min().item(), ys.max().item() + 1
            x0, x1 = xs.min().item(), xs.max().item() + 1
            bbox = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)

            # image is (3, H, W)
            crop = image[:, y0:y1, x0:x1].permute(1, 2, 0).float()
            crop_mask = mask_2d[y0:y1, x0:x1].unsqueeze(-1).float()

            if self.mask_cropped_instances:
                crop = crop * crop_mask

            # Category and score
            cat_id = torch.tensor(-1, dtype=torch.long)
            score = 1.0
            if instance_classes is not None and inst_id < len(instance_classes):
                cat_id = instance_classes[inst_id].long()
            if instance_scores is not None and inst_id < len(instance_scores):
                score = float(instance_scores[inst_id])

            # Compute embedding if encoder is available
            embedding = None
            if self.encoder is not None and self.use_visual_feat:
                try:
                    img_np = crop.clamp(0, 255).byte().cpu().numpy()
                    embedding = self.encoder.encode_image(img_np).squeeze(0).cpu()
                except Exception as e:
                    logger.debug("Failed to encode instance crop: %s", e)

            bounds = torch.stack([mins, maxs], dim=1)  # (3, 2)

            view = InstanceView(
                cropped_image=crop,
                mask=crop_mask,
                bbox_xyxy=bbox,
                cam_to_world=cam_to_world,
                embedding=embedding,
                score=score,
                bounds=bounds,
            )

            pts_rgb = None
            if image.shape[0] == 3:
                img_hw3 = image.permute(1, 2, 0).float()
                pts_rgb = img_hw3[inst_valid]

            pending.append(
                _PendingInstance(
                    local_id=inst_id,
                    view=view,
                    point_cloud=pts_3d,
                    point_cloud_rgb=pts_rgb,
                    category_id=cat_id,
                    score=score,
                    bounds=bounds,
                )
            )

        self._unassociated[env_id] = pending

    def associate_instances_to_memory(self) -> None:
        """Match pending instances to existing tracked instances or create new ones."""
        for env_id in range(self.num_envs):
            pending = self._unassociated.get(env_id, [])
            existing = self.instances[env_id]

            for p in pending:
                best_match_id = None
                best_iou = 0.0

                if self.instance_association == "bbox_iou" and existing:
                    for gid, inst in existing.items():
                        if inst.bounds is None or p.bounds is None:
                            continue
                        iou = _bbox3d_iou(inst.bounds, p.bounds)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_id = gid

                if best_match_id is not None and best_iou > 0.1:
                    inst = existing[best_match_id]
                    inst.add_view(p.view)
                    # Merge point clouds
                    if inst.point_cloud is not None and p.point_cloud is not None:
                        inst.point_cloud = torch.cat([inst.point_cloud, p.point_cloud], dim=0)
                        if inst.point_cloud_rgb is not None and p.point_cloud_rgb is not None:
                            inst.point_cloud_rgb = torch.cat(
                                [inst.point_cloud_rgb, p.point_cloud_rgb], dim=0
                            )
                    elif p.point_cloud is not None:
                        inst.point_cloud = p.point_cloud
                        inst.point_cloud_rgb = p.point_cloud_rgb
                    inst.update_bounds()
                    inst.score = max(inst.score, p.score)
                else:
                    gid = self._allocate_global_id()
                    inst = Instance(
                        instance_id=gid,
                        global_id=gid,
                        category_id=p.category_id,
                        score=p.score,
                    )
                    inst.add_view(p.view)
                    inst.point_cloud = p.point_cloud
                    inst.point_cloud_rgb = p.point_cloud_rgb
                    inst.bounds = p.bounds
                    existing[gid] = inst

            self._unassociated[env_id] = []

    def global_box_compression_and_nms(self, env_id: int = 0) -> None:
        """Merge overlapping instances via 3D NMS and remove duplicates."""
        instances = self.instances[env_id]
        if len(instances) < 2:
            return

        gids = list(instances.keys())
        to_remove = set()

        for i, gid_a in enumerate(gids):
            if gid_a in to_remove:
                continue
            for gid_b in gids[i + 1 :]:
                if gid_b in to_remove:
                    continue
                inst_a = instances[gid_a]
                inst_b = instances[gid_b]
                if inst_a.bounds is None or inst_b.bounds is None:
                    continue
                iou = _bbox3d_iou(inst_a.bounds, inst_b.bounds)
                if iou > 0.5:
                    # Keep the one with more views / higher score
                    if len(inst_a.instance_views) >= len(inst_b.instance_views):
                        to_remove.add(gid_b)
                    else:
                        to_remove.add(gid_a)
                        break

        for gid in to_remove:
            instances.pop(gid, None)

    def pop_global_instance(self, env_id: int, global_instance_id: int) -> Optional[Instance]:
        """Remove and return an instance by global ID."""
        return self.instances.get(env_id, {}).pop(global_instance_id, None)

    def get_all_instances(self, env_id: int = 0) -> List[Instance]:
        return list(self.instances.get(env_id, {}).values())


@dataclass
class _PendingInstance:
    """Temporary container for an instance detected in a single frame, before association."""

    local_id: int
    view: InstanceView
    point_cloud: Optional[Tensor]
    point_cloud_rgb: Optional[Tensor]
    category_id: Tensor
    score: float
    bounds: Optional[Tensor]


def _bbox3d_iou(bounds_a: Tensor, bounds_b: Tensor) -> float:
    """Compute IoU between two axis-aligned 3D bounding boxes.

    Each bounds tensor is (3, 2) with columns [min, max].
    """
    mins_a, maxs_a = bounds_a[:, 0], bounds_a[:, 1]
    mins_b, maxs_b = bounds_b[:, 0], bounds_b[:, 1]

    inter_mins = torch.max(mins_a, mins_b)
    inter_maxs = torch.min(maxs_a, maxs_b)
    inter_extent = (inter_maxs - inter_mins).clamp(min=0)
    inter_vol = inter_extent.prod().item()

    vol_a = (maxs_a - mins_a).clamp(min=0).prod().item()
    vol_b = (maxs_b - mins_b).clamp(min=0).prod().item()

    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol
