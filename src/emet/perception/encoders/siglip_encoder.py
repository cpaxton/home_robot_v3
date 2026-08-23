# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from emet.utils.logger import Logger, suppress_hf_hub_http_logging

from .base_encoder import BaseImageTextEncoder

# All SigLIP-family checkpoints are fixed-resolution (NOT naflex) and share the same
# vision-tower API, so MaskSiglipEncoder's per-pixel head surgery applies to every entry.
SIGLIP_CHECKPOINTS = {
    # SigLIP 1
    "base": "google/siglip-base-patch16-224",
    "so400m": "google/siglip-so400m-patch14-384",
    # SigLIP 2 (better text alignment + dense features)
    "siglip2_base": "google/siglip2-base-patch16-224",
    "siglip2_so400m": "google/siglip2-so400m-patch14-384",
    # SigLIP 2 high-resolution variants (Siglip2Encoder registry names map here)
    "siglip2_base_512": "google/siglip2-base-patch16-512",
    "siglip2_large_512": "google/siglip2-large-patch16-512",
    "siglip2_so400m_512": "google/siglip2-so400m-patch16-512",
    "siglip2_giant": "google/siglip2-giant-opt-patch16-384",
}

logger = Logger(__name__)


class SiglipEncoder(BaseImageTextEncoder):
    """Image/text feature encoder using SIGLip model.

    Referencing the following paper: https://arxiv.org/abs/2303.15343

    From the HuggingFace implementation here: https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/siglip

    Generally, these features are much better than OpenAI CLIP for open-vocabulary object detection.
    """

    def __init__(
        self,
        normalize: bool = True,
        device: str | None = None,
        version: str | None = None,
        feature_matching_threshold: float = 0.05,
        dtype: str | None = None,
        **kwargs,
    ) -> None:
        suppress_hf_hub_http_logging()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.normalize = normalize
        self.feature_matching_threshold = feature_matching_threshold

        # Weight dtype: float32 (default), float16, or bfloat16. Halves VRAM
        # (so400m: 3.5 GB -> 1.75 GB). All public outputs are cast back to fp32 so
        # stored voxel features and similarity thresholds are unaffected.
        # Do NOT use bitsandbytes int4/int8 here: MaskSiglipEncoder calls F.linear on
        # raw weight tensors (head surgery), which packed quantized weights break.
        if dtype is None:
            dtype = os.environ.get("EMET_SIGLIP_DTYPE", "").strip().lower() or "float32"
        valid_dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype {dtype}: must be one of {sorted(valid_dtypes)}")
        self.torch_dtype = valid_dtypes[dtype]

        if version is None:
            version = "base"

        if version not in SIGLIP_CHECKPOINTS:
            raise ValueError(f"Invalid version {version}: must be one of {sorted(SIGLIP_CHECKPOINTS)}")
        model_name = SIGLIP_CHECKPOINTS[version]

        # Hub models ship as .safetensors; avoid legacy pytorch_model.bin resolution errors.
        from emet.llms.hf_local import resolve_pretrained_source

        source, local_kw = resolve_pretrained_source(model_name)
        _sf = {"use_safetensors": True, **local_kw}
        self.processor = AutoProcessor.from_pretrained(source, use_fast=False, **_sf)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **local_kw)
        self.model = AutoModel.from_pretrained(source, dtype=self.torch_dtype, **_sf).to(self.device)

    def _to_model_inputs(self, inputs: dict) -> dict:
        """Move inputs to the model device, casting float tensors to the weight dtype."""
        out = {}
        for k, v in inputs.items():
            v = v.to(self.device)
            if torch.is_floating_point(v):
                v = v.to(self.torch_dtype)
            out[k] = v
        return out

    def encode_image(
        self,
        image: torch.Tensor | np.ndarray,
        image_shape=(360, 270),
        verbose: bool = False,
    ) -> torch.Tensor:
        """Encode this input image to a feature vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.uint8)

        # We should avoid using PIL image to allow parrelism

        # pil_image = Image.fromarray(image)
        # if verbose:
        #     logger.info("Encoding image", pil_image.size)
        # inputs = self.processor(images=pil_image, return_tensors="pt")

        inputs = self._to_model_inputs(self.processor(images=image, return_tensors="pt"))
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        image_features = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
        if not isinstance(image_features, torch.Tensor):
            image_features = out.last_hidden_state[:, 0]
        if self.normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    def encode_text(self, text: str) -> torch.Tensor:
        """Return feature vector for text"""
        # inputs = self.processor(text, return_tensors="pt")
        inputs = self._to_model_inputs(self.tokenizer([text], padding="max_length", return_tensors="pt"))
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        text_features = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
        if not isinstance(text_features, torch.Tensor):
            text_features = out.last_hidden_state[:, 0]
        if self.normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def classify(self, image: np.ndarray | torch.Tensor, text: str) -> torch.Tensor:
        """Classify image and text"""

        # Convert image to PIL
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)

        # Process image and text
        inputs = self._to_model_inputs(
            self.processor(images=pil_image, text=text, return_tensors="pt", padding="max_length")
        )

        # Evaluate model
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits_per_image
        probs = torch.sigmoid(logits)
        return probs

    def encode_batch_text(self, texts: list[str]) -> torch.Tensor:
        """Return feature vector for text"""
        # inputs = self.processor(text, return_tensors="pt")
        inputs = self._to_model_inputs(self.tokenizer(texts, padding="max_length", return_tensors="pt"))
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        text_features = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
        if not isinstance(text_features, torch.Tensor):
            text_features = out.last_hidden_state[:, 0]
        return text_features.float()

    def compute_score(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Compute similarity score between image and text"""
        # return torch.sigmoid((image @ text.T).sum(dim=-1))
        return torch.cosine_similarity(image, text, dim=-1)


class MaskSiglipEncoder(SiglipEncoder):
    def __init__(
        self,
        device: str | None = None,
        version: str | None = None,
        feature_matching_threshold: float = 0.12,
        dtype: str | None = None,
    ) -> None:
        """
        Extract pixel-wise features from SIGLip model
        """
        super().__init__(
            normalize=True,
            device=device,
            version=version,
            feature_matching_threshold=feature_matching_threshold,
            dtype=dtype,
        )

    def forward_one_block_(self, resblocks, x):
        x = F.linear(x, resblocks.in_proj_weight, resblocks.in_proj_bias)
        N, L, C = x.shape
        x = x.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        x = F.linear(x, resblocks.out_proj.weight, resblocks.out_proj.bias)
        q, k, v = x.tensor_split(3, dim=0)

        return v

    def extract_mask_siglip_features(self, x, image_shape):
        with torch.no_grad():
            output = self.model.vision_model(x["pixel_values"], output_hidden_states=True)
        feat = output.last_hidden_state
        feat = self.forward_one_block_(self.model.vision_model.head.attention, feat)
        feat = self.model.vision_model.head.layernorm(feat)
        feat = feat + self.model.vision_model.head.mlp(feat)
        # Do the spatial upsampling + normalize on GPU (fp32): CPU bilinear on a
        # 512-channel 4D tensor is very slow and dominated per-update wall time in
        # OVMM find evals. Only the final small feature map moves to CPU.
        with torch.no_grad():
            N, L, H, W = self.model.vision_model.embeddings.patch_embedding(x["pixel_values"]).shape
        feat = feat.reshape(N, H, W, L).permute(0, 3, 1, 2).float()
        feat = F.interpolate(feat, image_shape, mode="bilinear", align_corners=True)
        feat = F.normalize(feat, dim=1)
        return feat.permute(0, 2, 3, 1).detach().cpu()

    def run_mask_siglip(self, image, image_shape):
        """
        Run mask siglip
        Input:
            image: RGB image, shape [3, H, W]
            image_shape: desired output shape, tuple (H1, W1)
        Output:
            image: RGB image, shape [3, H1, W1]
            features: pixel-wise features, shape [H1, W1, 512]
        """
        input = self._to_model_inputs(self.processor(images=image, padding="max_length", return_tensors="pt"))
        if image_shape is not None:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = F.interpolate(image, size=image_shape, mode="bilinear", align_corners=False).squeeze()
        features = self.extract_mask_siglip_features(input, image.shape[-2:])

        return image, features

    def extract_per_pixel_features(self, x, image_shape):
        """
        Same as run_mask_siglip, but for multiple images
        """
        with torch.no_grad():
            output = self.model.vision_model(x["pixel_values"], output_hidden_states=True)
            feat = output.last_hidden_state
            feat = self.forward_one_block_(self.model.vision_model.head.attention, feat)
            feat = self.model.vision_model.head.layernorm(feat)
            feat = feat + self.model.vision_model.head.mlp(feat)
            feat = feat.detach().cpu().float()
            N, L, H, W = self.model.vision_model.embeddings.patch_embedding(x["pixel_values"]).shape
            feat = feat.reshape(N, H, W, L).permute(0, 3, 1, 2)
        features = []
        for f, size in zip(feat, image_shape, strict=False):
            f = F.interpolate(f.unsqueeze(0), size, mode="bilinear", align_corners=True)[0]
            f = F.normalize(f, dim=0).permute(1, 2, 0)
            features.append(f.detach().cpu())
        return features


_SHARED_MASK_SIGLIP: dict[tuple[str, str, str], "MaskSiglipEncoder"] = {}


def get_shared_mask_siglip_encoder(
    version: str = "so400m",
    device: str = "cuda",
    feature_matching_threshold: float = 0.12,
) -> "MaskSiglipEncoder":
    """Return a process-wide, load-once ``MaskSiglipEncoder`` keyed by (version, device).

    Batch runs (e.g. Habitat EQA) build a fresh controller per episode; without sharing,
    SigLIP weights would reload every episode (slow + VRAM churn). Mirrors the shared
    GraphEQA VLM client pattern.

    ``EMET_SIGLIP_VERSION`` overrides *version* (e.g. ``siglip2_so400m``) so eval
    scripts can A/B encoders without config edits.
    """
    env_version = os.environ.get("EMET_SIGLIP_VERSION", "").strip()
    if env_version:
        version = env_version
    dtype = os.environ.get("EMET_SIGLIP_DTYPE", "").strip().lower() or "float32"
    key = (version or "so400m", device or "cuda", dtype)
    enc = _SHARED_MASK_SIGLIP.get(key)
    if enc is None:
        enc = MaskSiglipEncoder(
            version=key[0],
            device=key[1],
            feature_matching_threshold=feature_matching_threshold,
            dtype=dtype,
        )
        _SHARED_MASK_SIGLIP[key] = enc
    return enc


def release_shared_mask_siglip_encoder() -> None:
    """Drop the shared SigLIP encoder cache and free CUDA weights."""
    encoders = list(_SHARED_MASK_SIGLIP.values())
    _SHARED_MASK_SIGLIP.clear()
    for enc in encoders:
        model = getattr(enc, "model", None)
        if model is not None:
            try:
                if hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
        try:
            enc.model = None  # type: ignore[attr-defined]
        except Exception:
            pass
    del encoders
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
