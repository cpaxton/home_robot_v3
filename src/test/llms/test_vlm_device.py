# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from emet.llms.vlm_device import assert_cuda_placement


class _FakeParam:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)


class _FakeModel:
    def __init__(self, devices: list[str], *, device_map: dict | None = None) -> None:
        self._params = [_FakeParam(d) for d in devices]
        if device_map is not None:
            self.hf_device_map = device_map

    def parameters(self):
        yield from self._params

    def buffers(self):
        return iter(())


def test_assert_cuda_placement_trusts_clean_device_map(monkeypatch: pytest.MonkeyPatch) -> None:
    # Would hang/fail if we walked every param when map says CUDA-only.
    model = _FakeModel(["cuda:0"], device_map={"": 0})
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("should not sample params when device_map is clean")

    monkeypatch.setattr("emet.llms.vlm_device.parameter_device_counts", boom)
    assert assert_cuda_placement(model, requested_device="cuda", model_label="fake") == "cuda:0"
    assert calls["n"] == 0


def test_assert_cuda_placement_rejects_cpu_in_device_map() -> None:
    model = _FakeModel(["cpu"], device_map={"": "cpu"})
    with pytest.raises(RuntimeError, match="hf_device_map"):
        assert_cuda_placement(model, requested_device="cuda", model_label="fake")


def test_assert_cuda_placement_samples_without_device_map() -> None:
    model = _FakeModel(["cuda:0", "cuda:0"])
    assert assert_cuda_placement(model, requested_device="cuda", model_label="fake") == "cuda:0"
