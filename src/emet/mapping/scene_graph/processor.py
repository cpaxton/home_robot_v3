# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Scene graph processor: takes RGBD observations + segmentation and updates the
# OpenVocabSceneGraph. Designed to be called from DynaMem's process_rgbd_images
# or from a standalone pipeline.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from emet.config.agents import load_agent_config
from emet.mapping.scene_graph.open_vocab_scene_graph import (
    ObjectObservation,
    OpenVocabSceneGraph,
)
from emet.utils.logger import Logger

logger = Logger(__name__)


class SceneGraphProcessor:
    """Processes RGBD frames into scene graph updates.

    Orchestrates: segmentation -> embedding -> scene graph update.
    Can use SAM3 (primary) or SAM2+OWL (fallback) for segmentation,
    and SigLIP + DINOv3 for dual embeddings.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_name: str = "default_scene_graph",
        device: Optional[str] = None,
        text_encoder: Any = None,
        visual_encoder: Any = None,
        segmenter: Any = None,
    ):
        if config is None:
            config = load_agent_config(config_name)
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Scene graph
        sg_cfg = config.get("scene_graph", {})
        self.scene_graph = OpenVocabSceneGraph(
            dedup_visual_threshold=sg_cfg.get("dedup_visual_threshold", 0.85),
            dedup_iou_threshold=sg_cfg.get("dedup_iou_threshold", 0.3),
            max_near_distance=sg_cfg.get("max_near_distance", 1.5),
            min_on_height=sg_cfg.get("min_on_height", 0.02),
            max_on_height=sg_cfg.get("max_on_height", 0.3),
            floor_z_threshold=sg_cfg.get("floor_z_threshold", 0.05),
            staleness_horizon=sg_cfg.get("staleness_horizon", 0),
            min_observations_stable=sg_cfg.get("min_observations_stable", 2),
            min_points_per_object=sg_cfg.get("min_points_per_object", 10),
        )

        # Encoders (lazy-loaded if not provided)
        self._text_encoder = text_encoder
        self._visual_encoder = visual_encoder
        self._segmenter = segmenter
        self._step = 0

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            self._text_encoder = self._build_text_encoder()
        return self._text_encoder

    @property
    def visual_encoder(self):
        if self._visual_encoder is None:
            self._visual_encoder = self._build_visual_encoder()
        return self._visual_encoder

    @property
    def segmenter(self):
        if self._segmenter is None:
            self._segmenter = self._build_segmenter()
        return self._segmenter

    def _build_text_encoder(self):
        cfg = self.config.get("embeddings", {}).get("text_encoder", {})
        name = cfg.get("name", "siglip")
        args = {k: v for k, v in cfg.items() if k != "name"}
        args["device"] = self.device
        from emet.perception.encoders import get_encoder

        return get_encoder(name, args)

    def _build_visual_encoder(self):
        cfg = self.config.get("embeddings", {}).get("visual_encoder", {})
        name = cfg.get("name", "dinov3")
        args = {k: v for k, v in cfg.items() if k != "name"}
        args["device"] = self.device
        from emet.perception.encoders import get_encoder

        return get_encoder(name, args)

    def _build_segmenter(self):
        seg_cfg = self.config.get("segmentation", {})
        primary = seg_cfg.get("primary", "sam3")

        if primary == "sam3":
            try:
                from emet.perception.detection.sam3 import SAM3Perception

                sam3_cfg = seg_cfg.get("sam3", {})
                seg = SAM3Perception(
                    model_id=sam3_cfg.get("model_id", "facebook/sam3"),
                    device=self.device,
                    confidence_threshold=sam3_cfg.get("confidence_threshold", 0.5),
                    min_area=sam3_cfg.get("min_area", 500),
                )
                if seg.available:
                    logger.info("Using SAM3 as primary segmenter")
                    return seg
                logger.warning("SAM3 not available, falling back")
            except Exception as e:
                logger.warning(f"SAM3 import failed ({e}), falling back")

        # Fallback
        fallback = seg_cfg.get("fallback", "owlsam")
        return self._build_fallback_segmenter(fallback, seg_cfg)

    def _build_fallback_segmenter(self, name: str, seg_cfg: dict):
        if name == "owlsam":
            from emet.perception.detection.owl.owlsam_perception import OWLSAMProcessor

            owl_cfg = seg_cfg.get("owlsam", {})
            return OWLSAMProcessor(
                version=owl_cfg.get("owl_version", "owlv2-L-p14-ensemble"),
                device=self.device,
                confidence_threshold=owl_cfg.get("owl_confidence", 0.15),
            )
        elif name == "sam2":
            from emet.perception.detection.sam2 import SAM2Perception

            sam2_cfg = seg_cfg.get("sam2", {})
            return SAM2Perception(
                configuration=sam2_cfg.get("configuration", "l"),
            )
        else:
            raise ValueError(f"Unknown fallback segmenter: {name}")

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray,
        world_xyz: Optional[Tensor] = None,
        vocabulary: Optional[List[str]] = None,
    ) -> List[int]:
        """Process a single RGBD frame and update the scene graph.

        Args:
            rgb: (H, W, 3) uint8
            depth: (H, W) float meters
            intrinsics: (3, 3) camera matrix
            camera_pose: (4, 4) cam-to-world
            world_xyz: (H, W, 3) pre-computed world points; computed from depth if None
            vocabulary: text prompts for SAM3; uses config vocabulary if None

        Returns:
            List of node IDs that were updated/created
        """
        self._step += 1
        H, W = rgb.shape[:2]

        # Compute world xyz if not provided
        if world_xyz is None:
            from emet.utils.image import Camera, camera_xyz_to_global_xyz

            camera = Camera.from_K(intrinsics, width=W, height=H)
            cam_xyz = camera.depth_to_xyz(depth)
            world_xyz = torch.from_numpy(
                camera_xyz_to_global_xyz(cam_xyz, camera_pose).astype(np.float32)
            )
        elif isinstance(world_xyz, np.ndarray):
            world_xyz = torch.from_numpy(world_xyz.astype(np.float32))

        # Valid depth mask
        min_depth = self.config.get("voxel", {}).get("min_depth", 0.25)
        max_depth = self.config.get("voxel", {}).get("max_depth", 2.5)
        valid = (depth > min_depth) & (depth < max_depth)

        # Segment
        vocab = vocabulary or self.config.get("vocabulary", [])
        detections = self._run_segmentation(rgb, depth, vocab)

        if not detections:
            return []

        # Build observations
        observations = []
        for det in detections:
            mask = det["mask"]  # (H, W) bool
            inst_valid = mask & valid

            pts = world_xyz[inst_valid]
            if pts.shape[0] < 3:
                continue

            # Crop
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            crop = rgb[y0:y1, x0:x1].copy()

            # RGB for points
            pts_rgb = torch.from_numpy(rgb[inst_valid].astype(np.float32))

            # Compute embeddings
            siglip_emb = self._encode_siglip(crop)
            dinov3_emb = self._encode_dinov3(crop)

            obs = ObjectObservation(
                mask=mask,
                bbox_xyxy=np.array([x0, y0, x1, y1]),
                rgb_crop=crop,
                points_3d=pts,
                points_rgb=pts_rgb,
                camera_pose=camera_pose,
                label=det.get("label", "object"),
                score=det.get("score", 1.0),
                timestep=self._step,
                siglip_embedding=siglip_emb,
                dinov3_embedding=dinov3_emb,
            )
            observations.append(obs)

        # Update scene graph
        node_ids = self.scene_graph.add_observations_batch(observations)

        # Periodic maintenance
        if self._step % 5 == 0:
            self.scene_graph.prune_small()
            self.scene_graph.merge_duplicates()
            self.scene_graph.update_edges()

        return node_ids

    def _run_segmentation(
        self, rgb: np.ndarray, depth: np.ndarray, vocabulary: List[str]
    ) -> List[Dict]:
        """Run the segmenter and return a list of detection dicts."""
        seg = self.segmenter

        # SAM3-style: text-prompted concept segmentation
        if hasattr(seg, "segment_concepts"):
            if vocabulary:
                return seg.segment_concepts(rgb, vocabulary)
            else:
                return seg.segment_concepts(rgb)

        # Fallback: standard predict interface
        try:
            if vocabulary:
                seg.reset_vocab(vocabulary)
            sem, inst, task_obs = seg.predict(rgb, depth=depth)
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}")
            return []

        # Convert to detection list
        detections = []
        labels = task_obs.get("labels", [])
        scores = task_obs.get("instance_scores", np.array([]))
        bboxes = task_obs.get("bboxes", None)

        unique_ids = np.unique(inst)
        for idx, uid in enumerate(unique_ids):
            if uid < 0:
                continue
            mask = inst == uid
            if mask.sum() < 100:
                continue

            label = labels[int(uid)] if int(uid) < len(labels) else "object"
            score = float(scores[int(uid)]) if int(uid) < len(scores) else 1.0
            ys, xs = np.where(mask)
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

            detections.append({
                "mask": mask.astype(bool),
                "label": label,
                "score": score,
                "bbox_xyxy": bbox,
            })

        return detections

    @torch.no_grad()
    def _encode_siglip(self, crop: np.ndarray) -> Optional[Tensor]:
        """Encode a crop with the text-aligned encoder."""
        try:
            feat = self.text_encoder.encode_image(crop)
            if feat.dim() == 2:
                feat = feat.squeeze(0)
            return F.normalize(feat.unsqueeze(0), dim=-1).squeeze(0).cpu()
        except Exception as e:
            logger.debug(f"SigLIP encoding failed: {e}")
            return None

    @torch.no_grad()
    def _encode_dinov3(self, crop: np.ndarray) -> Optional[Tensor]:
        """Encode a crop with the visual similarity encoder."""
        try:
            feat = self.visual_encoder.encode_image(crop)
            if feat.dim() == 2:
                feat = feat.squeeze(0)
            return F.normalize(feat.unsqueeze(0), dim=-1).squeeze(0).cpu()
        except Exception as e:
            logger.debug(f"DINOv3 encoding failed: {e}")
            return None
