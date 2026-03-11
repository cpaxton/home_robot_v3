# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# DINOv3 encoder for dense visual features (geometry/appearance, not text-aligned).
# Used alongside SigLIP for dual-embedding scene graph objects.

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from emet.utils.logger import Logger

from .base_encoder import BaseImageTextEncoder

logger = Logger(__name__)

DINOV3_MODELS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vitg16": "facebook/dinov3-vitg16-pretrain-lvd1689m",
}


class Dinov3Encoder(BaseImageTextEncoder):
    """Dense visual feature encoder using Meta DINOv3.

    DINOv3 produces high-quality self-supervised features useful for visual
    similarity, object deduplication, and geometric reasoning. Unlike SigLIP/CLIP,
    these features are NOT text-aligned -- use SigLIP for text queries.
    """

    def __init__(
        self,
        version: str = "vitb16",
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.normalize = normalize
        self.version = version

        model_name = DINOV3_MODELS.get(version)
        if model_name is None:
            raise ValueError(
                f"Unknown DINOv3 version '{version}'. Choose from: {list(DINOV3_MODELS.keys())}"
            )

        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        # Cache feature dim from model config
        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> Tensor:
        """Encode an image to a single pooled feature vector.

        Args:
            image: (H, W, 3) uint8 numpy or (3, H, W) / (H, W, 3) tensor

        Returns:
            (1, D) feature tensor
        """
        img = self._prepare_image(image)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        feat = outputs.pooler_output  # (1, D)
        if self.normalize:
            feat = F.normalize(feat, dim=-1)
        return feat.float().cpu()

    @torch.no_grad()
    def encode_image_dense(
        self,
        image: Union[torch.Tensor, np.ndarray],
        output_shape: Optional[tuple] = None,
    ) -> Tensor:
        """Extract dense per-patch features, optionally interpolated to output_shape.

        Args:
            image: (H, W, 3) uint8 numpy or tensor
            output_shape: (H_out, W_out) to interpolate features to; if None returns patch grid

        Returns:
            (H_out, W_out, D) dense feature map
        """
        img = self._prepare_image(image)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)

        # last_hidden_state: (1, num_patches+1, D) -- skip CLS token
        feat = outputs.last_hidden_state[:, 1:, :]  # (1, N_patches, D)

        # Reshape to spatial grid
        patch_size = self.model.config.patch_size
        h_in = inputs["pixel_values"].shape[2]
        w_in = inputs["pixel_values"].shape[3]
        h_patches = h_in // patch_size
        w_patches = w_in // patch_size
        feat = feat.reshape(1, h_patches, w_patches, -1).permute(0, 3, 1, 2)  # (1, D, H, W)

        if output_shape is not None:
            feat = F.interpolate(feat, size=output_shape, mode="bilinear", align_corners=True)

        feat = feat.squeeze(0).permute(1, 2, 0)  # (H, W, D)
        if self.normalize:
            feat = F.normalize(feat, dim=-1)
        return feat.float().cpu()

    def encode_text(self, text: str) -> Tensor:
        """DINOv3 does not support text encoding. Returns zeros.

        For text-based queries, use SigLIP or CLIP alongside DINOv3.
        """
        logger.warning("DINOv3 does not support text encoding; returning zero vector")
        return torch.zeros(1, self.feature_dim)

    def compute_score(self, image: Tensor, text: Tensor) -> Tensor:
        """Cosine similarity (meaningful only between two image embeddings)."""
        return torch.cosine_similarity(image, text, dim=-1)

    def _prepare_image(self, image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)
        return image
