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
from typing import Any

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
    text_description: str | None = None
    cropped_image: Tensor | None = None
    """Cropped image of instance (can be [C,H,W] or [H,W,C])"""
    embedding: Tensor | None = None
    mask: Tensor | None = None
    image_instance_id: int | None = None
    visual_feat: Tensor | None = None

    # Detection
    global_instance_id: int | None = None
    category_id: int | None = None
    score: float | None = None

    # 3D
    point_cloud: Tensor | None = None
    point_cloud_rgb: Tensor | None = None
    point_cloud_features: Tensor | None = None
    cam_to_world: Tensor | None = None
    """[4,4] camera space to world space"""
    pose: Tensor | None = None
    """Base pose of the robot when this view was collected"""

    def __repr__(self) -> str:
        cam = "present" if self.cam_to_world is not None else "None"
        return (
            f"InstanceView(timestep={self.timestep}, category_id={self.category_id}, "
            f"bounds={getattr(self.bounds, 'shape', None)}, cam_to_world={cam})"
        )

    def get_pose(self) -> Tensor | None:
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

    def _show_folder(self, folder_path: Path | str | None = None) -> None:
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


def _xyz_tensor_to_np3(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            return np.asarray(x.detach().cpu().numpy(), dtype=np.float64).reshape(3)
        return np.asarray(x, dtype=np.float64).reshape(3)
    except Exception:
        return None


def graph_label_for_instance_xyz(
    xyz: np.ndarray,
    graph_memory: Any,
    *,
    max_xy_m: float = 0.55,
) -> str | None:
    """Nearest GraphEQA object-node label in the XY plane (for Rerun when detector names are missing)."""
    get_nodes = getattr(graph_memory, "get_nodes", None)
    if get_nodes is None:
        return None
    best_lab: str | None = None
    best_d = float(max_xy_m)
    for node in get_nodes():
        if getattr(node, "is_viewpoint", False):
            continue
        labels = getattr(node, "labels", None) or []
        if not labels:
            continue
        nxyz = getattr(node, "xyz", None)
        if nxyz is None:
            continue
        nxy = np.asarray(nxyz, dtype=np.float64).reshape(3)[:2]
        d = float(np.linalg.norm(xyz[:2] - nxy))
        if d < best_d:
            primary = str(labels[0]).strip()
            if primary and not primary.startswith("obj_"):
                best_d = d
                best_lab = primary
    return best_lab


def instance_display_label(
    instance: Instance,
    *,
    semantic_sensor: Any | None = None,
    detection_model: Any | None = None,
    graph_memory: Any | None = None,
    class_names: dict[int, str] | None = None,
    graph_match_xy_m: float = 0.55,
) -> str:
    """Human-readable label for visualization (detector / graph class, not ``obj_{id}``)."""
    raw_name = getattr(instance, "name", None)
    if raw_name and str(raw_name).strip():
        name = str(raw_name).strip()
        if not name.startswith("obj_"):
            return name.replace(" ", "_")

    cid = instance.get_category_id() if hasattr(instance, "get_category_id") else None
    if cid is None:
        try:
            best = instance.get_best_view()
            view_cid = getattr(best, "category_id", None)
            if view_cid is not None:
                cid = int(view_cid.item()) if hasattr(view_cid, "item") else int(view_cid)
        except (ValueError, TypeError, AttributeError):
            pass

    if class_names and cid is not None and int(cid) in class_names:
        return str(class_names[int(cid)]).replace(" ", "_")

    if semantic_sensor is not None and getattr(semantic_sensor, "is_semantic", lambda: False)():
        if cid is not None:
            return semantic_sensor.get_class_name_for_id(cid).replace(" ", "_")

    if detection_model is not None and cid is not None:
        from emet.memory.graph_eqa.instance_observations import label_for_detection_category

        lab = label_for_detection_category(detection_model, int(cid))
        if lab and not lab.startswith("object_"):
            return lab.replace(" ", "_")

    try:
        best = instance.get_best_view()
        cap = getattr(best, "text_description", None)
        if cap and str(cap).strip():
            short = str(cap).strip().split(",")[0].split(".")[0][:48]
            return short.replace(" ", "_")
    except (ValueError, AttributeError):
        pass

    if graph_memory is not None:
        xyz = _xyz_tensor_to_np3(instance.get_center() if hasattr(instance, "get_center") else None)
        if xyz is not None:
            matched = graph_label_for_instance_xyz(xyz, graph_memory, max_xy_m=graph_match_xy_m)
            if matched:
                return matched.replace(" ", "_")

    if detection_model is not None and cid is not None:
        from emet.memory.graph_eqa.instance_observations import label_for_detection_category

        return label_for_detection_category(detection_model, int(cid)).replace(" ", "_")

    gid = getattr(instance, "global_id", None) or getattr(instance, "id", None)
    return f"id_{gid}" if gid is not None else "object"


def instances_to_text(
    instances: list[Instance],
    class_names: dict[int, str] | None = None,
    include_bounds: bool = True,
    include_caption: bool = True,
    include_moved: bool = True,
) -> str:
    """Format instance memory as human-readable text for logging or dumping.

    Args:
        instances: List of Instance from e.g. voxel_map.get_instances().
        class_names: Optional mapping category_id -> name (e.g. from semantic sensor).
        include_bounds: If True, append xyz min/max per instance.
        include_caption: If True, append best view caption when present.
        include_moved: If True, append moved_since_last when association is used.

    Returns:
        Multi-line string suitable for print or log.
    """
    if not instances:
        return "No instances in memory."
    lines = [f"Instance memory ({len(instances)} objects):"]
    for inst in instances:
        cid = inst.get_category_id()
        name = (class_names or {}).get(cid, None) if cid is not None else None
        label = name if name else f"id_{inst.id}"
        score_str = f" score={inst.score:.2f}" if inst.score is not None else ""
        line = f"  {inst.id}: {label}{score_str}"
        if include_bounds and inst.bounds is not None:
            b = inst.bounds.cpu().numpy() if isinstance(inst.bounds, Tensor) else inst.bounds
            line += f"  x=[{b[0, 0]:.2f},{b[0, 1]:.2f}] y=[{b[1, 0]:.2f},{b[1, 1]:.2f}] z=[{b[2, 0]:.2f},{b[2, 1]:.2f}]"
        if include_caption:
            best = inst.get_best_view()
            if getattr(best, "text_description", None):
                cap = best.text_description
                line += f"  caption={cap[:60]!r}..." if len(cap) > 60 else f"  caption={cap!r}"
        if include_moved and getattr(inst, "moved_since_last", False):
            line += "  moved"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class Instance:
    """
    A single instance found in the environment, composed of a list of InstanceView objects.
    Restored from home_robot_v2 mapping/instance/core.py.
    """

    name: str | None = None
    global_id: int | None = None
    category_id: int | None = None
    point_cloud: Tensor | None = None
    point_cloud_rgb: Tensor | None = None
    point_cloud_features: Tensor | None = None
    bounds: Tensor | None = None
    instance_views: list[InstanceView] = field(default_factory=list)
    score: float | None = None
    score_aggregation_method: str = "max"

    # Movement tracking (updated when association adds views)
    last_center: Tensor | None = None
    """Center of instance at last update; used to set moved_since_last."""
    moved_since_last: bool = False
    """True if center moved beyond threshold since last association."""

    # Cache for the aggregated image embedding. get_image_embedding re-concats ALL
    # views every call, which is O(views) per call and O(views^2) per association
    # pass (the associate loop calls it once per (view x global_instance)). Cache
    # the aggregated result and invalidate on add_instance_view.
    _cached_embedding: Any = None
    _cached_embedding_use_visual: bool | None = None

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

    def get_category_id(self) -> int | None:
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
        """Combined image embedding across all views (cached; invalidated on view add)."""
        if (
            self._cached_embedding is not None
            and self._cached_embedding_use_visual == use_visual_feat
            and aggregation_method == "mean"
            and normalize
        ):
            return self._cached_embedding
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
            raise RuntimeError(f"Unsupported aggregation method {aggregation_method}. Options: max, mean.")
        if normalize and emb is not None:
            emb = emb / (emb.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        if aggregation_method == "mean" and normalize:
            self._cached_embedding = emb
            self._cached_embedding_use_visual = use_visual_feat
        return emb

    def get_best_view(self, metric: str = "area") -> InstanceView:
        """Get best view by area or update_time. Returns dummy view if no views (with debug log)."""
        if not self.instance_views:
            logger.debug("Instance.get_best_view: no instance_views for global_id=%s", self.global_id)
            return _dummy_instance_view()
        best_view: InstanceView | None = None
        if metric == "area":
            best_area = 0.0
            for view in self.instance_views:
                if view.bbox is not None and view.bbox.numel() >= 4:
                    if view.bbox.dim() == 2:
                        area = float((view.bbox[1, 1] - view.bbox[0, 1]) * (view.bbox[1, 0] - view.bbox[0, 0]))
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

    def get_instance_id(self) -> int | None:
        return self.global_id

    def get_center(self) -> Tensor | None:
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

    def get_median(self) -> Tensor | None:
        if self.point_cloud is None or self.point_cloud.numel() == 0:
            return None
        return self.point_cloud.median(dim=0).values

    def get_closest_point(self, xyz: Tensor | np.ndarray) -> Tensor | None:
        if self.point_cloud is None or self.point_cloud.numel() == 0:
            return None
        if isinstance(xyz, np.ndarray):
            xyz = torch.as_tensor(xyz, device=self.point_cloud.device, dtype=self.point_cloud.dtype)
        dists = torch.norm(self.point_cloud - xyz, dim=1)
        return self.point_cloud[dists.argmin()]

    def show_best_view(
        self,
        metric: str = "area",
        title: str | None = None,
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
        self._cached_embedding = None
        self._cached_embedding_use_visual = None
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
                self.point_cloud_rgb = torch.cat([self.point_cloud_rgb, instance_view.point_cloud_rgb], dim=0)
            if instance_view.point_cloud_features is not None and self.point_cloud_features is not None:
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
        # Update last_center for movement tracking (moved_since_last set by caller when merging)
        if self.point_cloud is not None and self.point_cloud.numel() > 0:
            new_center = self.get_center()
            if new_center is not None:
                self.last_center = new_center.clone()


def _cropped_image_to_caption_input(cropped_image: Tensor) -> np.ndarray:
    """Convert InstanceView.cropped_image (C,H,W or H,W,C; 0-1 or 0-255) to numpy [H,W,3] uint8 for captioners."""
    img = cropped_image.cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if img.dtype == np.float32 or img.dtype == np.float64 or img.max() <= 1.0:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[-1] != 3:
        img = np.broadcast_to(img[..., None], (*img.shape[:2], 3))
    return img


def _process_instances_single_frame(
    env_instances: dict[int, Instance],
    instance_seg: Tensor,
    point_cloud: Tensor,
    image: Tensor | None,
    cam_to_world: Tensor | None,
    instance_classes: Tensor | None,
    instance_scores: Tensor | None,
    background_instance_labels: list[int],
    valid_points: Tensor | None,
    pose: Any,
    timestep: int,
    min_points: int = 3,
) -> None:
    """Build Instance + InstanceView per detection and store in env_instances (replaces content)."""
    H, W = instance_seg.shape
    if point_cloud.shape[0] != H or point_cloud.shape[1] != W:
        logger.warning(
            "process_instances: point_cloud shape %s does not match seg %s",
            point_cloud.shape,
            (H, W),
        )
        return
    bg = set(background_instance_labels or [-1])
    unique_ids = [x for x in instance_seg.unique().cpu().tolist() if x not in bg]
    env_instances.clear()
    pc_flat = point_cloud.reshape(-1, 3)
    img_chw = image  # (C, H, W)
    valid_2d = (
        valid_points
        if valid_points is not None
        else torch.ones_like(instance_seg, dtype=torch.bool, device=instance_seg.device)
    )
    for global_id, instance_id in enumerate(sorted(unique_ids)):
        mask_2d = (instance_seg == instance_id) & valid_2d
        mask_flat = mask_2d.flatten()
        pts = pc_flat[mask_flat]
        if pts.shape[0] < min_points:
            continue
        if pts.shape[0] < 2:
            continue
        bounds = get_bounds(pts)
        ys, xs = torch.where(mask_2d)
        x_min, x_max = xs.min().item(), xs.max().item()
        y_min, y_max = ys.min().item(), ys.max().item()
        bbox = torch.tensor(
            [[float(x_min), float(y_min)], [float(x_max), float(y_max)]],
            device=instance_seg.device,
            dtype=torch.float32,
        )
        category_id: int | None = None
        if instance_classes is not None and instance_id < instance_classes.shape[0]:
            category_id = int(instance_classes[instance_id].item())
        score: float | None = None
        if instance_scores is not None and instance_id < instance_scores.shape[0]:
            score = float(instance_scores[instance_id].item())
        point_cloud_rgb: Tensor | None = None
        if img_chw is not None and img_chw.shape[0] >= 3:
            rgb_hw3 = img_chw[:3].permute(1, 2, 0)
            point_cloud_rgb = rgb_hw3.reshape(-1, 3)[mask_flat].to(pts.dtype)
            if point_cloud_rgb.shape[0] != pts.shape[0]:
                point_cloud_rgb = None
        cropped_image: Tensor | None = None
        mask_1hw: Tensor | None = None
        if img_chw is not None:
            y0, y1 = max(0, y_min), min(H, y_max + 1)
            x0, x1 = max(0, x_min), min(W, x_max + 1)
            cropped_image = img_chw[:, y0:y1, x0:x1].clone()
            mask_crop = mask_2d[y0:y1, x0:x1].unsqueeze(0).float()
            mask_1hw = mask_crop
        if mask_1hw is None:
            mask_1hw = mask_2d.unsqueeze(0).float()
        view = InstanceView(
            bbox=bbox,
            bounds=bounds,
            timestep=timestep,
            cropped_image=cropped_image,
            mask=mask_1hw,
            point_cloud=pts,
            point_cloud_rgb=point_cloud_rgb,
            category_id=category_id,
            score=score,
            cam_to_world=cam_to_world,
            pose=pose
            if isinstance(pose, Tensor)
            else (torch.tensor(pose, dtype=torch.float32) if pose is not None else None),
            image_instance_id=int(instance_id),
            global_instance_id=global_id,
        )
        inst = Instance(
            global_id=global_id,
            category_id=category_id,
            bounds=bounds,
            point_cloud=pts,
            point_cloud_rgb=point_cloud_rgb,
            instance_views=[view],
            score=score,
        )
        env_instances[global_id] = inst


class InstanceMemory:
    """
    Instance memory: per-env dict of instances.
    process_instances_for_env builds Instance + InstanceView from segmentation and
    point cloud; optionally runs captioning on new views and associates across frames
    (de-duplication) when use_association is True.
    """

    def __init__(self, num_envs: int = 1, encoder: Any = None, **kwargs: Any) -> None:
        self.num_envs = num_envs
        self.encoder = encoder
        self.instances: dict[int, dict[int, Instance]] = {i: {} for i in range(num_envs)}
        self._timestep = 0
        self._next_global_id = 0
        self.captioner: Any = kwargs.get("captioner")
        self.use_association: bool = bool(kwargs.get("use_association", False))
        self.association_distance_m: float = float(kwargs.get("association_distance_m", 0.15))
        self.move_threshold_m: float = float(kwargs.get("move_threshold_m", 0.05))

    def __len__(self) -> int:
        """Total number of instances across all envs (for len(self) / voxel_map compatibility)."""
        return sum(len(env) for env in self.instances.values())

    def reset(self) -> None:
        for i in range(self.num_envs):
            self.instances[i] = {}
        self._timestep = 0
        self._next_global_id = 0

    def _caption_views(self, instances_dict: dict[int, Instance]) -> None:
        """Run captioner on views that have cropped_image and no text_description."""
        if not self.captioner or not hasattr(self.captioner, "caption_image"):
            return
        for inst in instances_dict.values():
            for view in inst.instance_views:
                if view.text_description or view.cropped_image is None:
                    continue
                try:
                    arr = _cropped_image_to_caption_input(view.cropped_image)
                    view.text_description = self.captioner.caption_image(arr)
                except Exception as e:
                    logger.debug("Caption failed for instance view: %s", e)

    def process_instances_for_env(
        self,
        env_id: int = 0,
        instance_seg: Tensor | None = None,
        point_cloud: Tensor | None = None,
        image: Tensor | None = None,
        cam_to_world: Tensor | None = None,
        instance_classes: Tensor | None = None,
        instance_scores: Tensor | None = None,
        background_instance_labels: list[int] | None = None,
        valid_points: Tensor | None = None,
        pose: Any = None,
        **kwargs: Any,
    ) -> None:
        """Build instances from current frame; optionally caption and associate (de-duplicate)."""
        if instance_seg is None or point_cloud is None:
            return
        env_instances = self.instances[env_id]
        if self.use_association:
            candidates: dict[int, Instance] = {}
            _process_instances_single_frame(
                env_instances=candidates,
                instance_seg=instance_seg,
                point_cloud=point_cloud,
                image=image,
                cam_to_world=cam_to_world,
                instance_classes=instance_classes,
                instance_scores=instance_scores,
                background_instance_labels=background_instance_labels or [-1],
                valid_points=valid_points,
                pose=pose,
                timestep=self._timestep,
            )
            self._caption_views(candidates)
            self._associate_candidates(env_id, candidates)
        else:
            _process_instances_single_frame(
                env_instances=env_instances,
                instance_seg=instance_seg,
                point_cloud=point_cloud,
                image=image,
                cam_to_world=cam_to_world,
                instance_classes=instance_classes,
                instance_scores=instance_scores,
                background_instance_labels=background_instance_labels or [-1],
                valid_points=valid_points,
                pose=pose,
                timestep=self._timestep,
            )
            self._caption_views(env_instances)
        self._timestep += 1

    def _associate_candidates(self, env_id: int, candidates: dict[int, Instance]) -> None:
        """Match candidates to existing instances by 3D center distance; merge or add new."""
        env = self.instances[env_id]
        for _frame_id, cand in list(candidates.items()):
            if not cand.instance_views:
                continue
            cand_center = cand.get_center()
            if cand_center is None:
                env[self._next_global_id] = cand
                cand.global_id = self._next_global_id
                self._next_global_id += 1
                continue
            best_id: int | None = None
            best_dist: float = float("inf")
            for existing_id, existing in env.items():
                ec = existing.get_center()
                if ec is None:
                    continue
                d = float(torch.norm(cand_center - ec).item())
                if d < best_dist and d < self.association_distance_m:
                    best_dist = d
                    best_id = existing_id
            if best_id is not None:
                existing = env[best_id]
                existing.moved_since_last = best_dist > self.move_threshold_m
                existing.add_instance_view(cand.instance_views[0])
            else:
                env[self._next_global_id] = cand
                cand.global_id = self._next_global_id
                self._next_global_id += 1
                if cand.last_center is None and cand.point_cloud is not None and cand.point_cloud.numel() > 0:
                    cand.last_center = cand.get_center().clone()

    def associate_instances_to_memory(self, candidates: dict[int, Instance] | None = None) -> None:
        """If candidates provided (by process_instances_for_env when use_association), already merged. No-op otherwise."""
        pass

    def global_box_compression_and_nms(self, env_id: int = 0, **kwargs: Any) -> Any:
        """No-op; return empty list for compatibility."""
        return []

    def pop_global_instance(
        self,
        env_id: int = 0,
        global_instance_id: int = 0,
        skip_reindex: bool = False,
    ) -> Instance | None:
        """Remove and return one instance from the env dict."""
        return self.instances[env_id].pop(global_instance_id, None)
