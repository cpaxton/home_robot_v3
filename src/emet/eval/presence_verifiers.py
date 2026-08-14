# Copyright (c) Chris Paxton 2026

"""Open-vocabulary presence verifiers used by offline bakeoffs and agentic EQA."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class DetectionEvidence:
    backend: str
    phrase: str
    score: float
    bbox_xyxy: tuple[int, int, int, int] | None
    latency_ms: float
    crop_siglip_sim: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PresenceDetector(Protocol):
    name: str

    def score(self, rgb: np.ndarray, phrase: str) -> DetectionEvidence: ...


def crop_bbox(
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int] | None,
    *,
    pad_fraction: float = 0.08,
) -> np.ndarray | None:
    if bbox_xyxy is None:
        return None
    image = np.asarray(rgb)
    if image.ndim != 3:
        return None
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    pad_x = int(max(1, (x1 - x0) * pad_fraction))
    pad_y = int(max(1, (y1 - y0) * pad_fraction))
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def siglip_cosine(encoder: Any, rgb: np.ndarray, phrase: str) -> float:
    image = encoder.encode_image(np.asarray(rgb, dtype=np.uint8))
    text = encoder.encode_text(phrase)
    image_np = image.detach().float().cpu().numpy().reshape(-1)
    text_np = text.detach().float().cpu().numpy().reshape(-1)
    image_np /= np.linalg.norm(image_np) + 1e-12
    text_np /= np.linalg.norm(text_np) + 1e-12
    return float(image_np @ text_np)


def dense_siglip_argmax_crop(
    encoder: Any,
    rgb: np.ndarray,
    phrase: str,
    *,
    patch_frac: float = 0.45,
) -> tuple[np.ndarray, float] | None:
    """Crop around the SigLIP dense-sim argmax patch for *phrase* (close-look).

    Returns ``(crop_rgb, max_sim)`` — a zoomed region centered on the most
    phrase-aligned patch so the VLM can read fine detail (counts, clock faces)
    that a wide frame hides. Falls back to ``None`` when the encoder / forward
    fails (callers keep the wide frame).
    """
    try:
        import torch
        import torch.nn.functional as F

        from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

        if encoder is None:
            encoder = get_shared_mask_siglip_encoder()
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3:
            return None
        text_t = encoder.encode_text(phrase).detach().float().reshape(-1)
        text_t = text_t / (text_t.norm() + 1e-12)
        inputs = encoder._to_model_inputs(encoder.processor(images=image, return_tensors="pt"))
        with torch.no_grad():
            out = encoder.model.vision_model(inputs["pixel_values"], output_hidden_states=True)
            feat = F.normalize(out.last_hidden_state.float(), dim=-1)
            sims = (feat @ text_t.to(device=feat.device, dtype=feat.dtype).reshape(-1, 1)).squeeze(-1)
        # sims layout: [batch, seq] where seq = 1 cls + patch tokens (excluding masks).
        if sims.ndim == 1:
            sims = sims.unsqueeze(0)
        # Drop the CLS token and infer a square-ish grid from the patch count.
        patches = sims[0][1:]
        max_sim = float(patches.max().item())
        if max_sim < 0.0 or patches.numel() < 4:
            return None
        argmax = int(patches.argmax().item())
        n = int(patches.numel())
        grid = int(round(n**0.5))
        if grid * grid != n:
            grid = max(1, int(round(n**0.5)))
        ph, pw = grid, max(1, n // grid)
        r, c = argmax // pw, argmax % pw
        h, w = image.shape[:2]
        # Map a patch block to image pixels (roughly patch_frac of the frame).
        cw = int(max(1, round(w * patch_frac)))
        ch = int(max(1, round(h * patch_frac)))
        cx = int(round((c + 0.5) / pw * w))
        cy = int(round((r + 0.5) / ph * h))
        x0, x1 = max(0, cx - cw // 2), min(w, cx + cw // 2)
        y0, y1 = max(0, cy - ch // 2), min(h, cy + ch // 2)
        if x1 <= x0 or y1 <= y0:
            return None
        return image[y0:y1, x0:x1].copy(), max_sim
    except Exception:
        return None


class OwlV2PresenceDetector:
    name = "owlv2"

    def __init__(
        self,
        *,
        version: str = "owlv2-B-p16",
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if model is None:
            from emet.perception.detection.owl import OwlPerception

            model = OwlPerception(
                version=version,
                device=device,
                confidence_threshold=0.0,
            )
        self.model = model

    def score(self, rgb: np.ndarray, phrase: str) -> DetectionEvidence:
        t0 = time.perf_counter()
        scores, boxes = self.model.detect_object(rgb, phrase, confidence_threshold=0.0)
        score = 0.0
        bbox = None
        if scores is not None and len(scores):
            scores_np = scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            boxes_np = boxes.detach().float().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
            index = int(np.argmax(scores_np))
            score = float(scores_np[index])
            bbox = tuple(int(round(v)) for v in boxes_np[index].tolist())
        return DetectionEvidence(
            backend=self.name,
            phrase=phrase,
            score=score,
            bbox_xyxy=bbox,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class YoloEPresenceDetector:
    name = "yoloe"

    def __init__(
        self,
        *,
        size: str = "s",
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if model is None:
            from emet.perception.detection.yoloe import YoloEPerception

            kwargs: dict[str, Any] = {
                "vocabulary": "custom",
                "class_list": ["object"],
                "size": size,
                "confidence_threshold": 0.001,
            }
            if device is not None:
                kwargs["device"] = device
            model = YoloEPerception(**kwargs)
        self.model = model

    def score(self, rgb: np.ndarray, phrase: str) -> DetectionEvidence:
        t0 = time.perf_counter()
        scores, boxes = self.model.detect_object(
            rgb,
            phrase,
            confidence_threshold=0.001,
            output_mask=False,
        )
        score = 0.0
        bbox = None
        if scores is not None and len(scores):
            scores_np = scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            boxes_np = boxes.detach().float().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
            index = int(np.argmax(scores_np))
            score = float(scores_np[index])
            bbox = tuple(int(round(v)) for v in boxes_np[index].tolist())
        return DetectionEvidence(
            backend=self.name,
            phrase=phrase,
            score=score,
            bbox_xyxy=bbox,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


def detector_crop_evidence(
    detector: PresenceDetector,
    encoder: Any,
    rgb: np.ndarray,
    phrase: str,
) -> DetectionEvidence:
    detection = detector.score(rgb, phrase)
    crop = crop_bbox(rgb, detection.bbox_xyxy)
    crop_sim = siglip_cosine(encoder, crop, phrase) if crop is not None else None
    return DetectionEvidence(
        backend=detection.backend,
        phrase=detection.phrase,
        score=detection.score,
        bbox_xyxy=detection.bbox_xyxy,
        latency_ms=detection.latency_ms,
        crop_siglip_sim=crop_sim,
    )
