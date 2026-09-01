# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, median_filter
from torch import Tensor

from emet.core.parameters import Parameters
from emet.llms import OpenaiClient
from emet.llms.eqa_vl_settings import resolve_vl_endpoint
from emet.llms.prompts import DYNAMEM_VISUAL_GROUNDING_PROMPT
from emet.llms.vllm_factory import create_dynamem_vllm, eqa_vl_client_kwargs
from emet.llms.vllm_registry import VLLMRunConfig, default_hf_model_id, normalize_vl_family, should_share_vllm
from emet.utils.image import Camera, camera_xyz_to_global_xyz
from emet.utils.morphology import binary_dilation, get_edges
from emet.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from emet.utils.voxel import VoxelizedPointcloud, scatter3d
from emet.utils.vram_debug import print_vram_snapshot

from .dynamem_eqa import DynamemVoxelEQAMixin
from .dynamem_localize import DynamemVoxelLocalizeMixin
from .voxel import VALID_FRAMES, Frame
from .voxel import SparseVoxelMap as SparseVoxelMapBase

logger = logging.getLogger(__name__)

# Subdir under self.log for debug files (rgb, depth, descriptions) so memory root stays canonical.
DEBUG_SUBDIR = "debug"


def _map_boundary_config(parameters: Parameters | dict | None) -> tuple[int, int]:
    """Return (obstacle_barrier_cells, history_penalty_cells); default 0 (no grid-edge barrier)."""
    if parameters is None:
        return 0, 0
    if isinstance(parameters, Parameters):
        mb = parameters.get("map_boundary", {})
    elif isinstance(parameters, dict):
        mb = parameters.get("map_boundary", {})
    else:
        mb = {}
    if not isinstance(mb, dict):
        mb = {}
    obs_cells = int(mb.get("obstacle_barrier_cells", 0) or 0)
    hist_cells = int(mb.get("history_penalty_cells", 0) or 0)
    return max(0, obs_cells), max(0, hist_cells)


def _apply_map_boundary_2d(
    obstacles: Tensor,
    history_soft: Tensor | None,
    parameters: Parameters | dict | None,
) -> None:
    """Mark grid-edge obstacles and optional history penalty from ``map_boundary`` config."""
    obs_barrier, hist_penalty = _map_boundary_config(parameters)
    h, w = int(obstacles.shape[0]), int(obstacles.shape[1])
    if obs_barrier > 0:
        n = min(obs_barrier, h // 2, w // 2)
        if n > 0:
            obstacles[0:n, :] = True
            obstacles[-n:, :] = True
            obstacles[:, 0:n] = True
            obstacles[:, -n:] = True
    if history_soft is not None and hist_penalty > 0:
        n = min(hist_penalty, h // 2, w // 2)
        if n > 0:
            mx = history_soft.max().item()
            history_soft[0:n, :] = mx
            history_soft[-n:, :] = mx
            history_soft[:, 0:n] = mx
            history_soft[:, -n:] = mx


def _eqa_qwen_vl_single_client_ok(
    vl_family: str,
    eqa_vl_hf_model_id: str | None,
    vl_dev: str,
    eqa_vl_quantization: str | None,
) -> bool:
    """True when registry policy allows one local :class:`~emet.llms.base.AbstractVLLMClient` for captions + EQA."""
    resolved = eqa_vl_hf_model_id or default_hf_model_id(vl_family)
    cfg = VLLMRunConfig(vl_family, resolved, vl_dev, eqa_vl_quantization)
    return should_share_vllm(cfg, cfg)


class SparseVoxelMap(DynamemVoxelEQAMixin, DynamemVoxelLocalizeMixin, SparseVoxelMapBase):
    def __init__(
        self,
        resolution: float = 0.01,
        semantic_memory_resolution: float = 0.05,
        feature_dim: int = 3,
        grid_size: tuple[int, int] = None,
        grid_resolution: float = 0.05,
        obs_min_height: float = 0.1,
        obs_max_height: float = 1.8,
        obs_min_density: float = 10,
        smooth_kernel_size: int = 2,
        neg_obs_height: float = 0.0,
        add_local_radius_points: bool = True,
        remove_visited_from_obstacles: bool = False,
        local_radius: float = 0.25,
        min_depth: float = 0.25,
        max_depth: float = 2.5,
        pad_obstacles: int = 0,
        background_instance_label: int = -1,
        instance_memory_kwargs: dict[str, Any] = None,
        voxel_kwargs: dict[str, Any] = None,
        encoder=None,
        map_2d_device: str = "cpu",
        device: str | None = None,
        use_instance_memory: bool = False,
        use_median_filter: bool = False,
        median_filter_size: int = 5,
        median_filter_max_error: float = 0.01,
        use_derivative_filter: bool = False,
        derivative_filter_threshold: float = 0.5,
        prune_detected_objects: bool = False,
        add_local_radius_every_step: bool = False,
        min_points_per_voxel: int = 10,
        use_negative_obstacles: bool = False,
        voxel_pcd_dbscan_min_samples: int = 0,
        point_update_threshold: float = 0.9,
        detection=None,
        image_shape=(480, 360),
        log="test",
        mllm=False,
        run_eqa=False,
        parameters: Parameters | dict | None = None,
        eqa_backend: str = "qwen_vl",
        eqa_vl_model_size: str = "8B",
        eqa_vl_max_tokens: int = 512,
        eqa_vl_quantization: str | None = "int4",
        eqa_vl_hf_model_id: str | None = None,
        gemini_model: str = "gemini-2.5-flash",
        eqa_device: str | None = None,
        vl_family: str = "qwen3_vl",
        defer_eqa_vllm: bool = False,
    ):
        if voxel_kwargs is None:
            voxel_kwargs = {}
        if instance_memory_kwargs is None:
            instance_memory_kwargs = {}
        super().__init__(
            resolution=resolution,
            feature_dim=feature_dim,
            grid_size=grid_size,
            grid_resolution=grid_resolution,
            obs_min_height=obs_min_height,
            obs_max_height=obs_max_height,
            obs_min_density=obs_min_density,
            smooth_kernel_size=smooth_kernel_size,
            neg_obs_height=neg_obs_height,
            add_local_radius_points=add_local_radius_points,
            remove_visited_from_obstacles=remove_visited_from_obstacles,
            local_radius=local_radius,
            min_depth=min_depth,
            max_depth=max_depth,
            pad_obstacles=pad_obstacles,
            background_instance_label=background_instance_label,
            instance_memory_kwargs=instance_memory_kwargs,
            voxel_kwargs=voxel_kwargs,
            encoder=encoder,
            map_2d_device=map_2d_device,
            device=device,
            use_instance_memory=use_instance_memory,
            use_median_filter=use_median_filter,
            median_filter_size=median_filter_size,
            median_filter_max_error=median_filter_max_error,
            use_derivative_filter=use_derivative_filter,
            derivative_filter_threshold=derivative_filter_threshold,
            prune_detected_objects=prune_detected_objects,
            add_local_radius_every_step=add_local_radius_every_step,
            min_points_per_voxel=min_points_per_voxel,
            use_negative_obstacles=use_negative_obstacles,
            voxel_pcd_dbscan_min_samples=voxel_pcd_dbscan_min_samples,
        )

        self.point_update_threshold = point_update_threshold
        self._history_soft: Tensor | None = None
        self.semantic_memory = VoxelizedPointcloud(voxel_size=semantic_memory_resolution).to(self.device)
        self.encoder = encoder
        self.image_shape = image_shape
        self.obs_count = 0
        self.detection_model = detection
        self.log = log
        self.mllm = mllm
        self.parameters = parameters

        # Open-vocabulary scene graph (optional, enabled via config or set_scene_graph_processor)
        self._scene_graph_processor = None
        if self.mllm:
            # Used to do visual grounding task
            self.gpt_client = OpenaiClient(DYNAMEM_VISUAL_GROUNDING_PROMPT, model="gpt-4o-2024-05-13")

        self.run_eqa = run_eqa
        if isinstance(parameters, Parameters):
            _eqa_raw = parameters.get("eqa", {})
        elif isinstance(parameters, dict):
            _eqa_raw = parameters.get("eqa", {})
        else:
            _eqa_raw = {}
        _eqa_cfg: dict[str, Any] = _eqa_raw if isinstance(_eqa_raw, dict) else {}
        self._vl_client_kw = eqa_vl_client_kwargs(_eqa_cfg)
        self._vl_endpoint = resolve_vl_endpoint(self.parameters) or (
            str(_eqa_cfg.get("vl_endpoint") or "").strip() or None
        )

        self._eqa_backend = str(_eqa_cfg.get("backend", eqa_backend) or "qwen_vl").strip().lower()
        self._vl_family = str(_eqa_cfg.get("vl_family", vl_family) or "qwen3_vl").strip().lower()
        eqa_ms = str(_eqa_cfg.get("vl_model_size", eqa_vl_model_size) or "8B")
        eqa_hf = _eqa_cfg.get("vl_hf_model_id", eqa_vl_hf_model_id)
        eqa_quant = _eqa_cfg.get("vl_quantization", eqa_vl_quantization)
        gemini_m = str(_eqa_cfg.get("gemini_model", gemini_model) or "gemini-2.5-flash")
        self._eqa_max_tokens = int(_eqa_cfg.get("vl_max_tokens", eqa_vl_max_tokens) or 512)

        self._eqa_device_resolved: str | None = None
        self._eqa_pending: dict[str, Any] | None = None

        def _hf_registry_eqa_path() -> bool:
            if self._eqa_backend == "gemini":
                return True
            if self._eqa_backend != "qwen_vl":
                return False
            return normalize_vl_family(self._vl_family) in ("qwen3_vl", "qwen2_5_vl", "gemma4")

        if self.run_eqa:
            if self.parameters is None:
                raise ValueError("SparseVoxelMap run_eqa=True requires ``parameters`` (dynav config).")

            from emet.llms.eqa_vl_settings import apply_eqa_vl_runtime_settings, get_eqa_vl_int

            apply_eqa_vl_runtime_settings(self.parameters)

            self.image_descriptions: list[tuple[list[str], list[int]]] = []

            if _hf_registry_eqa_path():
                from emet.llms.prompts.eqa_prompt import EQA_PROMPT

                _vl_dev = eqa_device
                if _vl_dev is None and self.device is not None:
                    _vl_dev = str(self.device)
                if _vl_dev not in ("cuda", "cpu", "mps"):
                    _vl_dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._eqa_device_resolved = _vl_dev

                if defer_eqa_vllm:
                    self._eqa_pending = {
                        "vl_family": self._vl_family,
                        "eqa_vl_hf_model_id": eqa_hf,
                        "eqa_vl_model_size": eqa_ms,
                        "eqa_vl_max_tokens": self._eqa_max_tokens,
                        "eqa_vl_quantization": eqa_quant,
                        "gemini_model": gemini_m,
                        "vl_endpoint": self._vl_endpoint,
                    }
                    self.image_description_client = None
                    self.eqa_client = None
                    if self._eqa_backend == "gemini":
                        from emet.llms.gemini_client import GeminiClient

                        self.eqa_client = GeminiClient(EQA_PROMPT, model=gemini_m)
                elif self._eqa_backend == "gemini":
                    from emet.llms.gemini_client import GeminiClient

                    self.image_description_client = create_dynamem_vllm(
                        self._vl_family,
                        hf_model_id=eqa_hf,
                        vl_model_size=eqa_ms,
                        max_tokens=max(256, self._eqa_max_tokens),
                        device=_vl_dev,
                        quantization=eqa_quant,
                        prompt=None,
                        endpoint=self._vl_endpoint,
                        **self._vl_client_kw,
                    )
                    self.eqa_client = GeminiClient(EQA_PROMPT, model=gemini_m)
                elif self._eqa_backend == "qwen_vl":
                    if not self._vl_endpoint and not _eqa_qwen_vl_single_client_ok(
                        self._vl_family, eqa_hf, _vl_dev, eqa_quant
                    ):
                        raise ValueError(
                            "EQA configuration does not allow a single shared local VLM for captions and QA "
                            f"(vl_family={self._vl_family!r}). See emet.llms.vllm_registry."
                        )
                    shared = create_dynamem_vllm(
                        self._vl_family,
                        hf_model_id=eqa_hf,
                        vl_model_size=eqa_ms,
                        max_tokens=self._eqa_max_tokens,
                        device=_vl_dev,
                        quantization=eqa_quant,
                        prompt=None,
                        endpoint=self._vl_endpoint,
                        **self._vl_client_kw,
                    )
                    self.image_description_client = shared
                    self.eqa_client = shared
                else:
                    raise ValueError(
                        f"Unknown eqa_backend {self._eqa_backend!r}; use 'qwen_vl' or 'gemini' (see dynav_config.yaml eqa:)."
                    )
            else:
                from emet.llms.eqa_qwen import build_shared_eqa_clients

                kw = get_eqa_vl_int(self.parameters, "voxel_keyword_max_tokens", 20)
                self.image_description_client, self.eqa_client = build_shared_eqa_clients(
                    parameters=self.parameters,
                    keyword_max_tokens=kw,
                )
                self._eqa_max_tokens = get_eqa_vl_int(self.parameters, "eqa_max_tokens", 1024)

        # Attributes for EQA, If you are not running EQA module, this will stay the same.
        self._question: str | None = None
        self.relevant_objects: list | None = None

        self.history_outputs: list[str] = []

    def bind_shared_vllm_from_agent(self, client: Any) -> bool:
        """Reuse the agent's VL client for DynaMem captions/EQA when it matches deferred ``eqa:`` config.

        Returns True only when the client is an :class:`~emet.llms.base.AbstractVLLMClient` whose
        :meth:`~emet.llms.base.AbstractVLLMClient.canonical_model_key` matches the pending EQA
        run config (family, HF id, device, quant). Text routers such as ``qwen35-4B`` are VL
        clients but **must not** steal the deferred 8B caption load — caller should then
        :meth:`materialize_local_eqa_vllm`.
        """
        from emet.llms.base import AbstractVLLMClient
        from emet.llms.vllm_registry import (
            VLLMRunConfig,
            config_from_client,
            default_hf_model_id,
            should_share_vllm,
        )

        if not self.run_eqa or self._eqa_pending is None:
            return False
        if not isinstance(client, AbstractVLLMClient):
            return False
        p = self._eqa_pending
        _vl_dev = self._eqa_device_resolved or ("cuda" if torch.cuda.is_available() else "cpu")
        pending_hf = p.get("eqa_vl_hf_model_id") or default_hf_model_id(p["vl_family"])
        pending_cfg = VLLMRunConfig(
            str(p["vl_family"]),
            pending_hf,
            str(_vl_dev),
            p.get("eqa_vl_quantization"),
        )
        try:
            client_cfg = config_from_client(client)
        except TypeError:
            return False
        if not should_share_vllm(client_cfg, pending_cfg):
            return False
        self.image_description_client = client
        if self._eqa_backend == "qwen_vl":
            self.eqa_client = client
        self._eqa_pending = None
        print_vram_snapshot("voxel_dynamem_bind_shared_vllm_from_agent")
        return True

    def materialize_local_eqa_vllm(self) -> None:
        """Load a dedicated local EQA VLM when defer was used but no shared VL client was bound."""
        if not self.run_eqa or self._eqa_pending is None:
            return
        if self.image_description_client is not None:
            self._eqa_pending = None
            return
        p = self._eqa_pending
        _vl_dev = self._eqa_device_resolved or ("cuda" if torch.cuda.is_available() else "cpu")
        if self._eqa_backend == "gemini":
            self.image_description_client = create_dynamem_vllm(
                p["vl_family"],
                hf_model_id=p["eqa_vl_hf_model_id"],
                vl_model_size=p["eqa_vl_model_size"],
                max_tokens=max(256, int(p["eqa_vl_max_tokens"])),
                device=_vl_dev,
                quantization=p["eqa_vl_quantization"],
                prompt=None,
                endpoint=p.get("vl_endpoint") or self._vl_endpoint,
                **self._vl_client_kw,
            )
        elif self._eqa_backend == "qwen_vl":
            endpoint = p.get("vl_endpoint") or self._vl_endpoint
            if not endpoint and not _eqa_qwen_vl_single_client_ok(
                p["vl_family"], p["eqa_vl_hf_model_id"], _vl_dev, p["eqa_vl_quantization"]
            ):
                raise ValueError(
                    "EQA configuration does not allow a single shared local VLM for captions and QA "
                    f"(vl_family={p['vl_family']!r}). See emet.llms.vllm_registry."
                )
            shared = create_dynamem_vllm(
                p["vl_family"],
                hf_model_id=p["eqa_vl_hf_model_id"],
                vl_model_size=p["eqa_vl_model_size"],
                max_tokens=int(p["eqa_vl_max_tokens"]),
                device=_vl_dev,
                quantization=p["eqa_vl_quantization"],
                prompt=None,
                endpoint=endpoint,
                **self._vl_client_kw,
            )
            self.image_description_client = shared
            self.eqa_client = shared
        else:
            raise ValueError(
                f"Unknown eqa_backend {self._eqa_backend!r}; use 'qwen_vl' or 'gemini' (see dynav_config.yaml eqa:)."
            )
        self._eqa_pending = None
        print_vram_snapshot("voxel_dynamem_materialize_local_eqa_vllm")

    def set_scene_graph_processor(self, processor) -> None:
        """Attach a SceneGraphProcessor to update an open-vocab scene graph on each frame."""
        self._scene_graph_processor = processor

    def get_scene_graph(self):
        """Return the OpenVocabSceneGraph if a processor is attached, else None."""
        if self._scene_graph_processor is not None:
            return self._scene_graph_processor.scene_graph
        return None

    def get_2d_map(self, debug: bool = False, return_history_id: bool = False, kernel: int = 7) -> tuple[Tensor, ...]:
        """
        Get 2d map with explored area and frontiers.
        return_history_id: if True, return when each voxel was recently updated
        """

        # Is this already cached? If so we don't need to go to all this work
        if self._map2d is not None and self._history_soft is not None and self._seq == self._2d_last_updated:
            return self._map2d if not return_history_id else (*self._map2d, self._history_soft)

        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        xyz, _, counts, _ = self.voxel_pcd.get_pointcloud()
        # print(counts)
        # if xyz is not None:
        #     counts = torch.ones(xyz.shape[0])
        obs_ids = self.voxel_pcd._obs_counts
        if xyz is None:
            xyz = torch.zeros((0, 3))
            counts = torch.zeros(0)
            obs_ids = torch.zeros(0)

        device = xyz.device
        xyz = ((xyz / self.grid_resolution) + self.grid_origin + 0.5).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        # print('min_height', min_height, 'max_height', max_height)
        grid_size = self.grid_size + [max_height]
        voxels = torch.zeros(grid_size, device=device)

        # Mask out obstacles only above a certain height
        obs_mask = xyz[:, -1] < max_height
        xyz = xyz[obs_mask, :]
        counts = counts[obs_mask][:, None]
        # print(counts)
        obs_ids = obs_ids[obs_mask][:, None]

        # voxels[x_coords, y_coords, z_coords] = 1
        voxels = scatter3d(xyz, counts, grid_size)
        history_ids = scatter3d(xyz, obs_ids, grid_size, "max")

        # Compute the obstacle voxel grid based on what we've seen
        obstacle_voxels = voxels[:, :, min_height:max_height]
        obstacles_soft = torch.sum(obstacle_voxels, dim=-1)
        obstacles = obstacles_soft > self.obs_min_density

        history_ids = history_ids[:, :, min_height:max_height]
        history_soft = torch.max(history_ids, dim=-1).values
        history_soft = torch.from_numpy(maximum_filter(history_soft.float().numpy(), size=kernel))

        if self._remove_visited_from_obstacles:
            # Remove "visited" points containing observations of the robot
            obstacles *= (1 - self._visited).bool()

        if self.dilate_obstacles_kernel is not None:
            obstacles = binary_dilation(
                obstacles.float().unsqueeze(0).unsqueeze(0),
                self.dilate_obstacles_kernel,
            )[0, 0].bool()

        # Explored = any occupied XY column plus the start-pose ``local_radius`` disk.
        # Do not morphologically close this mask: open/close with kernel 3 painted over
        # real observation gaps (Stretch rotate-only / no head pan). Spawn navigability
        # is the visited disk stamped on the first observation (or ``_seed_local_radius_explored``).
        explored_soft = torch.sum(voxels, dim=-1)
        explored = explored_soft > 0
        explored = (torch.zeros_like(explored) + self._visited).to(torch.bool) | explored
        if debug:
            import matplotlib.pyplot as plt

            plt.subplot(2, 2, 1)
            plt.imshow(obstacles_soft.detach().cpu().numpy())
            plt.title("obstacles soft")
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(explored_soft.detach().cpu().numpy())
            plt.title("explored soft")
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(obstacles.detach().cpu().numpy())
            plt.title("obstacles")
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(explored.detach().cpu().numpy())
            plt.axis("off")
            plt.title("explored")
            plt.show()

        # Optional grid-edge obstacle barrier (map_boundary/obstacle_barrier_cells in dynav YAML).
        _apply_map_boundary_2d(obstacles, history_soft, self.parameters)

        # Update cache
        self._map2d = (obstacles, explored)
        self._2d_last_updated = self._seq
        self._history_soft = history_soft
        if not return_history_id:
            return obstacles, explored
        else:
            return obstacles, explored, history_soft

    def _depth_validity_mask(self, depth: Tensor) -> Tensor:
        """Pixels safe for both insertion and free-space carving.

        Clear and add must share this mask: a depth sample rejected as too noisy /
        edge-like for mapping must never be treated as evidence that geometry moved.
        """
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth, dtype=torch.float32)
        depth = depth.float()
        valid = torch.isfinite(depth) & (depth > float(self.min_depth)) & (depth < float(self.max_depth))
        if self.use_derivative_filter:
            edges = get_edges(depth, threshold=self.derivative_filter_threshold)
            valid = valid & ~edges
        if self.use_median_filter:
            median_depth = torch.from_numpy(
                median_filter(depth.detach().cpu().numpy(), size=int(self.median_filter_size))
            ).to(device=depth.device, dtype=depth.dtype)
            median_filter_error = (depth - median_depth).abs()
            valid = valid & (median_filter_error < float(self.median_filter_max_error))
        return valid.bool()

    def process_rgbd_images(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose: np.ndarray,
        *,
        base_xyt: np.ndarray | None = None,
        full_perception: bool = True,
    ):
        """
        Process rgbd images for Dynamem

        Args:
            base_xyt: Optional ``(x, y, yaw)`` in the same world frame as ``gps`` / ``compass`` from the
                robot client. When set, stamps ``_visited`` at the **base** so A* ``_navigable`` matches the
                planner start pose (camera pose alone can miss the footprint for head-mounted cameras).
            full_perception: When False, skip the expensive object-level stack (YoloE
                detection, SigLIP dense features, instance memory, semantic memory).
                Occupancy / clearance / visited still update so navigation is current.
        """
        # Keep originals for scene graph processor (before any resizing/filtering)
        original_rgb = rgb.copy()
        original_depth = depth.copy()
        original_intrinsics = intrinsics.copy()
        original_pose = pose.copy()
        _t_pr0 = time.time()

        # Log input data to debug subdir so memory root stays canonical for save_memory().
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        debug_dir = os.path.join(self.log, DEBUG_SUBDIR)
        os.makedirs(debug_dir, exist_ok=True)
        self.obs_count += 1

        cv2.imwrite(os.path.join(debug_dir, "rgb" + str(self.obs_count) + ".jpg"), rgb[:, :, [2, 1, 0]])
        np.save(os.path.join(debug_dir, "rgb" + str(self.obs_count) + ".npy"), rgb)
        np.save(os.path.join(debug_dir, "depth" + str(self.obs_count) + ".npy"), depth)
        np.save(os.path.join(debug_dir, "intrinsics" + str(self.obs_count) + ".npy"), intrinsics)
        np.save(os.path.join(debug_dir, "pose" + str(self.obs_count) + ".npy"), pose)

        base_pose_t: Tensor | None = None
        if base_xyt is not None:
            b = np.asarray(base_xyt, dtype=np.float64).ravel()
            if b.size >= 2:
                th = float(b[2]) if b.size >= 3 else 0.0
                dev = torch.device(self.map_2d_device)
                base_pose_t = torch.tensor([float(b[0]), float(b[1]), th], dtype=torch.float32, device=dev)

        # Same validity mask for clear and add: never carve free space from pixels we
        # would refuse to insert (edges / median outliers / OOR depth).
        depth_t = torch.as_tensor(depth, dtype=torch.float32)
        depth_is_valid = self._depth_validity_mask(depth_t)

        # Update obstacle map. Clearing and adding are two halves of one refresh pass;
        # they must always run together or the map loses geometry nothing re-adds.
        dbscan_min = int(getattr(self, "_voxel_pcd_dbscan_min_samples", 0) or 0)
        self.voxel_pcd.clear_points(
            depth_t,
            torch.from_numpy(intrinsics),
            torch.from_numpy(pose),
            depth_is_valid=depth_is_valid,
            min_samples_clear=dbscan_min if dbscan_min > 0 else None,
            max_depth=self.max_depth,
        )
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            print(f"[update] clear_points={time.time() - _t_pr0:.3f}s", flush=True)

        instance_image = None
        instance_classes = None
        instance_scores = None
        if full_perception and self.use_instance_memory and self.detection_model is not None:
            try:
                sem, instance, task_obs = self.detection_model.predict(
                    rgb, depth=depth, draw_instance_predictions=False
                )
                instance_image = torch.from_numpy(instance.astype(np.int64))
                instance_classes = torch.from_numpy(task_obs["instance_classes"].astype(np.int64))
                instance_scores = torch.from_numpy(task_obs["instance_scores"].astype(np.float32))
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    print(f"[update] detect={time.time() - _t_pr0:.3f}s", flush=True)
            except Exception as e:
                logger.warning("Instance detection failed in process_rgbd_images: %s", e)

        self.add(
            camera_pose=torch.Tensor(pose),
            rgb=torch.Tensor(rgb),
            depth=torch.Tensor(depth),
            camera_K=torch.Tensor(intrinsics),
            base_pose=base_pose_t,
            instance_image=instance_image,
            instance_classes=instance_classes,
            instance_scores=instance_scores,
        )
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            print(f"[update] add()={time.time() - _t_pr0:.3f}s", flush=True)

        # Add image descriptions if we want to explore intelligently
        if self.run_eqa and self.image_description_client is not None:
            self.list_objects_in_an_image(rgb)

        # Process data: reshaping images, computing xyz coordinate, depth filtering
        rgb, depth = torch.Tensor(rgb), torch.Tensor(depth)
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)

        if self.image_shape is not None:
            h, w = self.image_shape
            h_image, w_image = depth.shape
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            intrinsics = np.copy(intrinsics)
            intrinsics[0, 0] *= w / w_image
            intrinsics[1, 1] *= h / h_image
            intrinsics[0, 2] *= w / w_image
            intrinsics[1, 2] *= h / h_image

        height, width = depth.squeeze().shape
        camera = Camera.from_K(np.array(intrinsics), width=width, height=height)
        camera_xyz = camera.depth_to_xyz(np.array(depth))
        world_xyz = torch.Tensor(camera_xyz_to_global_xyz(camera_xyz, np.array(pose)))

        median_depth = torch.from_numpy(median_filter(depth, size=5))
        median_filter_error = (depth - median_depth).abs()
        valid_depth = torch.logical_and(depth < self.max_depth, depth > self.min_depth)
        valid_depth = valid_depth & (median_filter_error < 0.01).bool()
        mask = ~valid_depth
        self.update_close_map_from_view(pose, world_xyz, valid_depth)

        # Update semantic memory (skipped when encoder is None, e.g. manipulation_only mapping)
        self.semantic_memory.clear_points(
            depth,
            torch.from_numpy(intrinsics),
            torch.from_numpy(pose),
            depth_is_valid=valid_depth,
            min_samples_clear=10,
            max_depth=self.max_depth,
        )

        if full_perception and self.encoder is not None:
            with torch.no_grad():
                _t_enc = time.time()
                rgb, features = self.encoder.run_mask_siglip(rgb, self.image_shape)  # type:ignore
                rgb, features = rgb.squeeze(), features.squeeze()
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    print(f"[update] siglip_enc={time.time() - _t_enc:.3f}s", flush=True)

            valid_xyz = world_xyz[~mask]
            features = features[~mask]
            valid_rgb = rgb.permute(1, 2, 0)[~mask]
            if len(valid_xyz) != 0:
                self.add_to_semantic_memory(valid_xyz, features, valid_rgb)

        # Update open-vocab scene graph if attached
        if self._scene_graph_processor is not None:
            try:
                self._scene_graph_processor.process_frame(
                    rgb=original_rgb,
                    depth=original_depth,
                    intrinsics=original_intrinsics,
                    camera_pose=original_pose,
                    world_xyz=world_xyz,
                )
            except Exception as e:
                from emet.utils.logger import warning as _warn_colored

                _warn_colored(f"Scene graph update failed: {e}")
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            print(f"[update] process_rgbd_images_end={time.monotonic():.3f}", flush=True)

    def add_to_semantic_memory(
        self,
        valid_xyz: torch.Tensor | None,
        feature: torch.Tensor | None,
        valid_rgb: torch.Tensor | None,
        weights: torch.Tensor | None = None,
        threshold: float = 0.95,
    ):
        """
        Add pixel points into the semantic memory
        """
        # Adding all points to voxelizedPointCloud is useless and expensive, we should exclude threshold of all points
        # Adding pixel points into the semantic memory is expensive; subsample but always keep ≥1 point.
        n_keep = max(1, int((1 - threshold) * len(valid_xyz)))
        selected_indices = torch.randperm(len(valid_xyz))[:n_keep]
        if valid_xyz is not None:
            valid_xyz = valid_xyz[selected_indices]
        if feature is not None:
            feature = feature[selected_indices]
        if valid_rgb is not None:
            valid_rgb = valid_rgb[selected_indices]
        if weights is not None:
            weights = weights[selected_indices]

        valid_xyz = valid_xyz.to(self.device)
        if feature is not None:
            feature = feature.to(self.device)
        if valid_rgb is not None:
            valid_rgb = valid_rgb.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        self.semantic_memory.add(
            points=valid_xyz,
            features=feature,
            rgb=valid_rgb,
            weights=weights,
            obs_count=self.obs_count,
        )

    def add(
        self,
        camera_pose: Tensor,
        rgb: Tensor,
        xyz: Tensor | None = None,
        camera_K: Tensor | None = None,
        feats: Tensor | None = None,
        depth: Tensor | None = None,
        base_pose: Tensor | None = None,
        xyz_frame: str = "camera",
        instance_image: Tensor | None = None,
        instance_classes: Tensor | None = None,
        instance_scores: Tensor | None = None,
        **info,
    ):
        """Add this to our history of observations. Also update the current running map.

        Parameters:
            camera_pose(Tensor): [4,4] cam_to_world matrix
            rgb(Tensor): N x 3 color points
            camera_K(Tensor): [3,3] camera instrinsics matrix -- usually pinhole model
            xyz(Tensor): N x 3 point cloud points in camera coordinates
            feats(Tensor): N x D point cloud features; D == 3 for RGB is most common
            base_pose(Tensor): optional location of robot base
            instance_image(Tensor): [H,W] instance ids (e.g. -1 or 0 = background)
            instance_classes(Tensor): class id per instance
            instance_scores(Tensor): confidence per instance
        """
        # TODO: we should remove the xyz/feats maybe? just use observations as input?
        # TODO: switch to using just Obs struct?
        # Shape checking
        assert rgb.ndim == 3 or rgb.ndim == 2, f"{rgb.ndim=}: must be 2 or 3"
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        if isinstance(camera_pose, np.ndarray):
            camera_pose = torch.from_numpy(camera_pose)
        if depth is not None:
            assert rgb.shape[:-1] == depth.shape, (
                f"depth and rgb image sizes must match; got {rgb.shape=} {depth.shape=}"
            )
        assert xyz is not None or (camera_K is not None and depth is not None)
        if xyz is not None:
            assert xyz.shape[-1] == 3, "xyz must have last dimension = 3 for x, y, z position of points"
            assert rgb.shape == xyz.shape, "rgb shape must match xyz"
            # Make sure shape is correct here for xyz and any passed-in features
            if feats is not None:
                assert feats.shape[-1] == self.feature_dim, (
                    f"features must match voxel feature dimenstionality of {self.feature_dim}"
                )
                assert xyz.shape[0] == feats.shape[0], "features must be available for each point"
            else:
                pass
            if isinstance(xyz, np.ndarray):
                xyz = torch.from_numpy(xyz)
        if depth is not None:
            assert depth.ndim == 2 or xyz_frame == "world"
        if camera_K is not None:
            assert camera_K.ndim == 2, "camera intrinsics K must be a 3x3 matrix"
        assert camera_pose.ndim == 2 and camera_pose.shape[0] == 4 and camera_pose.shape[1] == 4, (
            "Camera pose must be a 4x4 matrix representing a pose in SE(3)"
        )
        assert xyz_frame in VALID_FRAMES, f"frame {xyz_frame} was not valid; should one one of {VALID_FRAMES}"

        # Apply a median filter to remove bad depth values when mapping and exploring
        # This is not strictly necessary but the idea is to clean up bad pixels
        if depth is not None and self.use_median_filter:
            median_depth = torch.from_numpy(median_filter(depth, size=self.median_filter_size))
            median_filter_error = (depth - median_depth).abs()

        # Get full_world_xyz
        if xyz is not None:
            if xyz_frame == "camera":
                full_world_xyz = (torch.cat([xyz, torch.ones_like(xyz[..., [0]])], dim=-1) @ camera_pose.T)[..., :3]
            elif xyz_frame == "world":
                full_world_xyz = xyz
            else:
                raise NotImplementedError(f"Unknown xyz_frame {xyz_frame}")
        else:
            full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
                depth=depth.unsqueeze(0).unsqueeze(1),
                pose=camera_pose.unsqueeze(0),
                inv_intrinsics=torch.linalg.inv(camera_K[:3, :3]).unsqueeze(0),
            )
        # add observations before we start changing things
        self.observations.append(
            Frame(
                camera_pose,
                camera_K,
                xyz,
                rgb,
                feats,
                depth,
                instance=instance_image,
                instance_classes=instance_classes,
                instance_scores=instance_scores,
                base_pose=base_pose,
                info=info,
                obs=None,
                full_world_xyz=full_world_xyz,
                xyz_frame=xyz_frame,
            )
        )

        valid_depth = torch.full_like(rgb[:, 0], fill_value=True, dtype=torch.bool)
        if depth is not None:
            valid_depth = (depth > self.min_depth) & (depth < self.max_depth)

            if self.use_derivative_filter:
                edges = get_edges(depth, threshold=self.derivative_filter_threshold)
                valid_depth = valid_depth & ~edges

            if self.use_median_filter:
                valid_depth = valid_depth & (median_filter_error < self.median_filter_max_error).bool()

        self.update_close_map_from_view(camera_pose, full_world_xyz, valid_depth)

        # Add instance views to memory (for UI icons / scene graph)
        if self.use_instance_memory and instance_image is not None:
            H, W = rgb.shape[0], rgb.shape[1]
            numel = full_world_xyz.numel()
            if numel == H * W * 3:
                pc = full_world_xyz.reshape(H, W, 3)
            else:
                logger.warning(
                    "full_world_xyz shape %s does not match H*W*3=%d; skipping instance update",
                    full_world_xyz.shape,
                    H * W * 3,
                )
                pc = None
            if pc is not None:
                img_chw = rgb.permute(2, 0, 1) if rgb.ndim == 3 else rgb.unsqueeze(0)
                seg = instance_image.clone()
                if seg.dtype != torch.long and seg.dtype != torch.int:
                    seg = seg.long()
                # Detection overlay_masks uses -1 for background
                self.instances.process_instances_for_env(
                    env_id=0,
                    instance_seg=seg,
                    point_cloud=pc,
                    image=img_chw,
                    cam_to_world=camera_pose,
                    instance_classes=instance_classes,
                    instance_scores=instance_scores,
                    background_instance_labels=[-1],
                    valid_points=valid_depth,
                    pose=base_pose,
                )
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    print(f"[update] process_instances={time.monotonic():.3f}", flush=True)
                self.instances.associate_instances_to_memory()
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    print(f"[update] associate_instances={time.monotonic():.3f}", flush=True)

        # Add to voxel grid
        if feats is not None:
            feats = feats[valid_depth].reshape(-1, feats.shape[-1])
        rgb = rgb[valid_depth].reshape(-1, 3)
        world_xyz = full_world_xyz.view(-1, 3)[valid_depth.flatten()]

        # TODO: weights could also be confidence, inv distance from camera, etc
        if world_xyz.nelement() > 0:
            n_keep = max(1, int((1 - self.point_update_threshold) * len(world_xyz)))
            selected_indices = torch.randperm(len(world_xyz))[:n_keep]
            if world_xyz is not None:
                world_xyz = world_xyz[selected_indices]
            if feats is not None:
                feats = feats[selected_indices]
            if rgb is not None:
                rgb = rgb[selected_indices]
            self.voxel_pcd.add(world_xyz, features=feats, rgb=rgb, weights=None)

        # Stamp the start disk once (YAML ``add_local_every_step``). Stamping every
        # rotate-in-place frame grew a blob that hid coverage holes in Rerun.
        if self._add_local_radius_points and (len(self.observations) < 2 or self._add_local_radius_every_step):
            if base_pose is not None:
                self._update_visited(base_pose.to(self.map_2d_device))
            else:
                self._update_visited(camera_pose[:3, 3].to(self.map_2d_device))

        # Increment sequence counter
        self._seq += 1

    def xy_to_grid_coords(self, xy: np.ndarray) -> np.ndarray | None:
        if not isinstance(xy, np.ndarray):
            xy = np.array(xy)
        return self.grid.xy_to_grid_coords(torch.Tensor(xy))

    def grid_coords_to_xy(self, grid_coords: np.ndarray) -> np.ndarray:
        if not isinstance(grid_coords, np.ndarray):
            grid_coords = np.array(grid_coords)
        return self.grid.grid_coords_to_xy(torch.Tensor(grid_coords))

    def grid_coords_to_xyt(self, grid_coords: np.ndarray) -> np.ndarray:
        if not isinstance(grid_coords, np.ndarray):
            grid_coords = np.array(grid_coords)
        return self.grid.grid_coords_to_xyt(torch.Tensor(grid_coords))

    def read_from_pickle(self, pickle_file_name, num_frames: int = -1):
        print("Reading from ", pickle_file_name)
        if isinstance(pickle_file_name, str):
            pickle_file_name = Path(pickle_file_name)
        assert pickle_file_name.exists(), f"No file found at {pickle_file_name}"
        # Clear any pre-load live frames so restore is the checkpoint, not a merge.
        if hasattr(self, "reset"):
            self.reset()
        with pickle_file_name.open("rb") as f:
            data = pickle.load(f)
        for i, (
            camera_pose,
            xyz,
            rgb,
            feats,
            depth,
            base_pose,
            K,
            _world_xyz,
        ) in enumerate(
            zip(
                data["camera_poses"],
                data["xyz"],
                data["rgb"],
                data["feats"],
                data["depth"],
                data["base_poses"],
                data["camera_K"],
                data["world_xyz"],
                strict=False,
            )
        ):
            # Handle the case where we dont actually want to load everything
            if num_frames > 0 and i >= num_frames:
                break

            camera_pose = self.fix_data_type(camera_pose)
            xyz = self.fix_data_type(xyz)
            rgb = self.fix_data_type(rgb)
            depth = self.fix_data_type(depth)
            intrinsics = self.fix_data_type(K)
            if feats is not None:
                feats = self.fix_data_type(feats)
            base_pose = self.fix_data_type(base_pose)
            depth_is_valid = self._depth_validity_mask(depth) if depth is not None else None
            self.voxel_pcd.clear_points(
                depth,
                intrinsics,
                camera_pose,
                depth_is_valid=depth_is_valid,
                max_depth=self.max_depth,
            )
            self.add(
                camera_pose=camera_pose,
                xyz=xyz,
                rgb=rgb,
                feats=feats,
                depth=depth,
                base_pose=base_pose,
                camera_K=K,
            )

            self.obs_count += 1
        self.semantic_memory._points = data["combined_xyz"]
        self.semantic_memory._features = data["combined_feats"]
        self.semantic_memory._weights = data["combined_weights"]
        self.semantic_memory._rgb = data["combined_rgb"]
        self.semantic_memory._obs_counts = data["obs_id"]
        self.semantic_memory._mins = self.semantic_memory._points.min(dim=0).values
        self.semantic_memory._maxs = self.semantic_memory._points.max(dim=0).values
        self.semantic_memory.obs_count = max(self.semantic_memory._obs_counts).item()
        self.semantic_memory.obs_count = max(self.semantic_memory._obs_counts).item()

    def write_to_pickle(self, filename: str | None = None) -> None:
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts.

        Args:
            filename (Optional[str], optional): Filename to write to. Defaults to None.
        """
        if filename is None:
            if not os.path.exists("debug"):
                os.mkdir("debug")
            filename = self.log + ".pkl"
        data: dict[str, Any] = {}
        data["camera_poses"] = []
        data["camera_K"] = []
        data["base_poses"] = []
        data["xyz"] = []
        data["world_xyz"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        for frame in self.observations:
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(frame.camera_pose)
            data["base_poses"].append(frame.base_pose)
            data["camera_K"].append(frame.camera_K)
            data["xyz"].append(frame.xyz)
            data["world_xyz"].append(frame.full_world_xyz)
            data["rgb"].append(frame.rgb)
            data["depth"].append(frame.depth)
            data["feats"].append(frame.feats)
            for k, v in (frame.info or {}).items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        (
            data["combined_xyz"],
            data["combined_feats"],
            data["combined_weights"],
            data["combined_rgb"],
        ) = self.semantic_memory.get_pointcloud()
        data["obs_id"] = self.semantic_memory._obs_counts
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print("write all data to", filename)

