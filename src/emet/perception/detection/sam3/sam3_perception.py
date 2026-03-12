# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# SAM 3 (Segment Anything with Concepts) perception module.
# Uses HuggingFace transformers API for text-prompted concept segmentation.
# Falls back to SAM2 + OWL if SAM3 is unavailable.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from emet.core.abstract_perception import PerceptionModule
from emet.perception.detection.utils import filter_depth, overlay_masks
from emet.utils.logger import Logger

logger = Logger(__name__)

SAM3_MODEL_ID = "facebook/sam3"


class SAM3Perception(PerceptionModule):
    """Open-vocabulary concept segmentation using Meta SAM 3.

    SAM3 supports Promptable Concept Segmentation (PCS): given a text prompt
    (e.g. "cup", "red cylinder"), it detects and segments ALL instances of that
    concept in the image. It also supports class-agnostic mask generation.
    """

    def __init__(
        self,
        model_id: str = SAM3_MODEL_ID,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        min_area: int = 500,
        verbose: bool = False,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self._verbose = verbose
        self._vocabulary: List[str] = []

        try:
            from transformers import Sam3Model, Sam3Processor

            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
            self._available = True
            logger.info(f"SAM3 loaded from {model_id} on {device}")
        except Exception as e:
            logger.warning(f"SAM3 not available ({e}); will fall back to SAM2+OWL")
            self._available = False
            self.processor = None
            self.model = None

    @property
    def available(self) -> bool:
        return self._available

    def reset_vocab(self, new_vocab: List[str]) -> None:
        self._vocabulary = list(new_vocab)

    @torch.no_grad()
    def segment_concepts(
        self,
        image: np.ndarray,
        text_prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Segment all instances of given text concepts in the image.

        Args:
            image: (H, W, 3) uint8 RGB
            text_prompts: list of concept strings; if None uses self._vocabulary

        Returns:
            List of dicts with keys: 'mask' (H,W bool), 'label' (str),
            'score' (float), 'bbox_xyxy' (4,)
        """
        if not self._available:
            return []

        prompts = text_prompts or self._vocabulary
        if not prompts:
            return self._generate_all_masks(image)

        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        results = []

        for prompt in prompts:
            inputs = self.processor(
                images=pil_image, text=prompt, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            processed = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.confidence_threshold
            )

            for seg_result in processed:
                for seg_info in seg_result.get("segments_info", []):
                    mask_id = seg_info["id"]
                    score = seg_info.get("score", 1.0)
                    seg_map = seg_result["segmentation"]
                    mask = (seg_map == mask_id).cpu().numpy()

                    if mask.sum() < self.min_area:
                        continue

                    ys, xs = np.where(mask)
                    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

                    results.append({
                        "mask": mask,
                        "label": prompt,
                        "score": float(score),
                        "bbox_xyxy": bbox,
                    })

        return results

    @torch.no_grad()
    def _generate_all_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Class-agnostic mask generation (no text prompt)."""
        if not self._available:
            return []

        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        processed = self.processor.post_process_instance_segmentation(
            outputs, threshold=self.confidence_threshold
        )

        results = []
        for seg_result in processed:
            seg_map = seg_result["segmentation"]
            for seg_info in seg_result.get("segments_info", []):
                mask_id = seg_info["id"]
                score = seg_info.get("score", 1.0)
                mask = (seg_map == mask_id).cpu().numpy()
                if mask.sum() < self.min_area:
                    continue
                ys, xs = np.where(mask)
                bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                results.append({
                    "mask": mask,
                    "label": "object",
                    "score": float(score),
                    "bbox_xyxy": bbox,
                })

        return results

    def predict(
        self,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Standard PerceptionModule predict interface.

        Returns (semantic_map, instance_map, task_observations).
        """
        if rgb is None:
            raise ValueError("rgb is required")

        image = np.asarray(rgb)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)

        H, W = image.shape[:2]

        if self._vocabulary:
            detections = self.segment_concepts(image, self._vocabulary)
        else:
            detections = self.segment_concepts(image)

        if not detections:
            semantic_map = np.zeros((H, W), dtype=int)
            instance_map = -np.ones((H, W), dtype=int)
            task_obs = {
                "instance_map": instance_map,
                "instance_classes": np.array([], dtype=int),
                "instance_scores": np.array([], dtype=float),
                "semantic_frame": None,
            }
            return semantic_map, instance_map, task_obs

        masks = [d["mask"] for d in detections]
        scores = np.array([d["score"] for d in detections])

        # Build label -> class_id mapping
        label_to_id: Dict[str, int] = {}
        class_ids = []
        for d in detections:
            lbl = d["label"]
            if lbl not in label_to_id:
                label_to_id[lbl] = len(label_to_id)
            class_ids.append(label_to_id[lbl])
        class_ids = np.array(class_ids)

        if depth_threshold is not None and depth is not None:
            masks = [filter_depth(m.astype(float), depth, depth_threshold) for m in masks]

        masks_arr = np.array(masks)
        semantic_map, instance_map = overlay_masks(masks_arr, class_ids, (H, W))

        task_obs = {
            "instance_map": instance_map,
            "instance_classes": class_ids,
            "instance_scores": scores,
            "semantic_frame": None,
            "labels": [d["label"] for d in detections],
            "bboxes": np.array([d["bbox_xyxy"] for d in detections]),
        }
        return semantic_map.astype(int), instance_map.astype(int), task_obs

    def is_semantic(self) -> bool:
        return True

    def is_instance(self) -> bool:
        return True
