# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import builtins
import os

import numpy as np
import torch
from innate_mars_bridge.onboard_dinov3 import (
    OnboardDinov3,
    create_onboard_dinov3_from_env,
    onboard_dinov3_enabled,
)


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_image(self, rgb):
        del rgb
        self.calls += 1
        return torch.tensor([[0.5, 0.5, 0.5, 0.5]])


def test_onboard_dinov3_disabled_by_default():
    os.environ.pop("EMET_MARS_ONBOARD_DINOV3", None)
    assert not onboard_dinov3_enabled()
    assert create_onboard_dinov3_from_env() is None


def test_onboard_dinov3_enabled_from_env():
    os.environ["EMET_MARS_ONBOARD_DINOV3"] = "1"
    try:
        assert onboard_dinov3_enabled()
        assert create_onboard_dinov3_from_env() is not None
    finally:
        os.environ.pop("EMET_MARS_ONBOARD_DINOV3", None)


def test_onboard_dinov3_infers_every_nth_frame_and_caches():
    enc = OnboardDinov3()
    fake = _FakeEncoder()
    enc._encoder = fake
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    assert enc.infer_head_embedding(rgb) == [0.5, 0.5, 0.5, 0.5]
    for _ in range(3):
        enc.infer_head_embedding(rgb)
    assert fake.calls == 1
    enc.infer_head_embedding(rgb)
    assert fake.calls == 2
    assert enc._last_embedding == [0.5, 0.5, 0.5, 0.5]
    assert enc.load_error is None


def test_onboard_dinov3_reports_import_error_when_encoders_missing(monkeypatch):
    enc = OnboardDinov3()
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name.startswith("emet.perception.encoders"):
            raise ImportError("no emet on robot")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    out = enc.infer_head_embedding(np.zeros((8, 8, 3), dtype=np.uint8))
    assert out is None
    assert enc.load_error is not None