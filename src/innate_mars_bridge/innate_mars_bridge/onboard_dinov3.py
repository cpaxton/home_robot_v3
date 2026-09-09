# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Optional DINOv3 head embedding on the Mars Jetson (publish over ZMQ).

When ``EMET_MARS_ONBOARD_DINOV3=1`` and ``transformers`` + ``torch`` are installed on the robot,
the ZMQ bridge runs DINOv3 vits16 on the head-left frame and adds ``dinov3_head`` to full
observations. Workstations can set ``EMET_DINOV3_ENDPOINT`` to a remote Orin instead when this
env is unset.

Deploy: ``emet mars start --deploy --onboard-dinov3``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_ONBOARD_ENV = "EMET_MARS_ONBOARD_DINOV3"


def onboard_dinov3_enabled() -> bool:
    return os.environ.get(_ONBOARD_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


class OnboardDinov3:
    """Lazy DINOv3 wrapper for innate_mars_bridge."""

    def __init__(self) -> None:
        self._encoder: Any = None
        self._step = 0
        self._infer_every_n = max(1, _env_int("EMET_MARS_DINOV3_INFER_EVERY_N", 4))
        self._version = os.environ.get("EMET_MARS_DINOV3_VERSION", "vits16").strip() or "vits16"
        self._last_embedding: list[float] | None = None
        self._load_error: str | None = None

    def _lazy_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        if self._load_error is not None:
            return None
        try:
            from emet.perception.encoders.dinov3_encoder import Dinov3Encoder
        except ImportError as exc:
            self._load_error = (
                f"Onboard DINOv3: emet.perception.encoders not importable ({exc}). "
                "Run `emet deploy --with-dinov3` on the robot checkout."
            )
            return None
        device = os.environ.get("EMET_MARS_DINOV3_DEVICE", "cuda")
        try:
            self._encoder = Dinov3Encoder(version=self._version, device=device)
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return None
        return self._encoder

    def infer_head_embedding(self, rgb: np.ndarray) -> list[float] | None:
        enc = self._lazy_encoder()
        if enc is None:
            return None
        self._step += 1
        run_full = self._infer_every_n <= 1 or (self._step - 1) % self._infer_every_n == 0
        if not run_full and self._last_embedding is not None:
            return list(self._last_embedding)

        try:
            feat = enc.encode_image(rgb)
            emb = feat.squeeze(0).float().cpu().tolist()
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return None
        if not emb:
            return None
        self._last_embedding = list(emb)
        return self._last_embedding

    @property
    def load_error(self) -> str | None:
        return self._load_error


def create_onboard_dinov3_from_env() -> OnboardDinov3 | None:
    if not onboard_dinov3_enabled():
        return None
    return OnboardDinov3()
