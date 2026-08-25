# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""HTTP client for remote DINOv3 embeddings (Jetson Orin offload)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from emet.utils.logger import Logger

from .base_encoder import BaseImageTextEncoder
from .dinov3_encoder import DINOV3_MODELS

logger = Logger(__name__)

_DEFAULT_FEATURE_DIM = 384  # dinov3-vits16


def resolve_dinov3_endpoint(explicit: str | None = None) -> str | None:
    for candidate in (
        explicit,
        os.environ.get("EMET_DINOV3_ENDPOINT"),
        os.environ.get("EMET_DINOV3_HOST"),
    ):
        s = (candidate or "").strip().rstrip("/")
        if s:
            if "://" not in s:
                s = f"http://{s}"
            return s
    return None


def _health_url(endpoint: str) -> str:
    return urljoin(endpoint.rstrip("/") + "/", "health")


def _embed_url(endpoint: str) -> str:
    return urljoin(endpoint.rstrip("/") + "/", "embed")


class RemoteDinov3Encoder(BaseImageTextEncoder):
    """DINOv3 via HTTP on a LAN Jetson (caliban :8002 by default)."""

    def __init__(
        self,
        version: str = "vitb16",
        device: str | None = None,
        normalize: bool = True,
        endpoint: str | None = None,
        timeout_s: float = 30.0,
        **kwargs: Any,
    ) -> None:
        del device, kwargs
        self.version = version
        self.normalize = normalize
        self.endpoint = resolve_dinov3_endpoint(endpoint)
        if not self.endpoint:
            raise ValueError("Remote DINOv3 requires EMET_DINOV3_ENDPOINT or endpoint=...")
        self.timeout_s = float(timeout_s)
        self.feature_dim = _DEFAULT_FEATURE_DIM
        self._probe_feature_dim()

    def _probe_feature_dim(self) -> None:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(_health_url(self.endpoint), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                import json

                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            logger.warning(f"Remote DINOv3 health probe failed ({exc}); using dim={self.feature_dim}")
            return
        dim = int(data.get("feature_dim") or 0)
        if dim > 0:
            self.feature_dim = dim

    def _image_to_b64(self, image: torch.Tensor | np.ndarray) -> str:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _post_embed(self, image_b64: str) -> Tensor:
        import json
        import urllib.error
        import urllib.request

        body = json.dumps({"image_b64": image_b64}).encode("utf-8")
        req = urllib.request.Request(
            _embed_url(self.endpoint),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError(f"remote DINOv3 missing embedding: {data}")
        feat = torch.tensor(emb, dtype=torch.float32).reshape(1, -1)
        if self.normalize:
            feat = F.normalize(feat, dim=-1)
        return feat

    def encode_image(self, image: torch.Tensor | np.ndarray, **kwargs: Any) -> Tensor:
        del kwargs
        return self._post_embed(self._image_to_b64(image))

    def encode_image_dense(
        self,
        image: torch.Tensor | np.ndarray,
        output_shape: tuple | None = None,
    ) -> Tensor:
        pooled = self.encode_image(image).squeeze(0)
        if output_shape is None:
            return pooled.unsqueeze(0)
        h, w = int(output_shape[0]), int(output_shape[1])
        return pooled.unsqueeze(0).unsqueeze(0).expand(h, w, -1)

    def encode_text(self, text: str) -> Tensor:
        logger.warning("DINOv3 does not support text encoding; returning zero vector")
        return torch.zeros(1, self.feature_dim)

    def compute_score(self, image: Tensor, text: Tensor) -> Tensor:
        return torch.cosine_similarity(image, text, dim=-1)


def build_dinov3_encoder(**args: Any) -> BaseImageTextEncoder:
    endpoint = resolve_dinov3_endpoint(args.pop("endpoint", None))
    if endpoint:
        return RemoteDinov3Encoder(endpoint=endpoint, **args)
    from .dinov3_encoder import Dinov3Encoder

    return Dinov3Encoder(**args)
