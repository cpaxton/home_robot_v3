# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for HF local-cache resolution and VLM device placement guards."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from emet.llms.hf_local import merge_pretrained_kwargs, resolve_pretrained_source
from emet.llms.vlm_device import assert_cuda_placement, parameter_device_counts, summarize_model_devices


def test_merge_pretrained_kwargs_extra_wins():
    out = merge_pretrained_kwargs({"a": 1, "b": 2}, {"b": 9, "c": 3})
    assert out == {"a": 1, "b": 9, "c": 3}


def test_resolve_pretrained_source_directory():
    src, kw = resolve_pretrained_source("/tmp")
    assert src == "/tmp"
    assert kw.get("local_files_only") is True


def test_resolve_pretrained_source_local_hit(tmp_path):
    # A "complete" cache snapshot has a processor/tokenizer/weights marker file;
    # resolve_pretrained_source refuses to force local_files_only without one.
    marker = tmp_path / "model.safetensors"
    marker.write_bytes(b"x")
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snap:
        src, kw = resolve_pretrained_source("org/model")
    assert src == str(tmp_path)
    assert kw == {"local_files_only": True}
    snap.assert_called_once_with("org/model", local_files_only=True)


def test_resolve_pretrained_source_prefer_local_false_skips_hub():
    src, kw = resolve_pretrained_source("org/missing-model", prefer_local=False)
    assert src == "org/missing-model"
    assert kw == {}


def test_resolve_pretrained_source_miss_falls_back_when_not_forced():
    with patch("huggingface_hub.snapshot_download", side_effect=OSError("missing")):
        src, kw = resolve_pretrained_source("org/missing-model", prefer_local=None)
    assert src == "org/missing-model"
    assert kw == {}


def test_resolve_pretrained_source_force_local_raises():
    with patch("huggingface_hub.snapshot_download", side_effect=OSError("missing")):
        with pytest.raises(OSError):
            resolve_pretrained_source("org/missing-model", prefer_local=True)


def test_parameter_device_counts_and_summary():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    m = _M()
    counts = parameter_device_counts(m)
    assert sum(counts.values()) >= 1
    assert "cpu" in summarize_model_devices(m)


def test_assert_cuda_placement_rejects_cpu_weights():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    m = _M()
    with pytest.raises(RuntimeError, match="not fully on GPU"):
        assert_cuda_placement(m, requested_device="cuda", model_label="test", allow_cpu=False)


def test_assert_cuda_placement_allows_cpu_when_flagged():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    m = _M()
    primary = assert_cuda_placement(m, requested_device="cuda", model_label="test", allow_cpu=True)
    assert primary.startswith("cpu")


def test_assert_cuda_placement_inspects_all_parameters():
    """Late CPU tensors must be caught (not only the first 64)."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(70):
                self.register_parameter(f"p{i}", torch.nn.Parameter(torch.zeros(1)))
            # Simulate a late offloaded param: first 70 are still CPU; assert uses full scan.
            self.tail = torch.nn.Parameter(torch.zeros(1))

    m = _M()
    m.hf_device_map = {f"p{i}": "cuda:0" for i in range(70)}
    m.hf_device_map["tail"] = "cpu"
    with pytest.raises(RuntimeError, match="hf_device_map|not fully on GPU"):
        assert_cuda_placement(m, requested_device="cuda", model_label="test", allow_cpu=False)


def test_parameter_device_counts_max_params_none_counts_all():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(70):
                self.register_parameter(f"p{i}", torch.nn.Parameter(torch.zeros(1)))

    m = _M()
    sampled = parameter_device_counts(m, max_params=64)
    full = parameter_device_counts(m, max_params=None)
    assert sum(sampled.values()) == 64
    assert sum(full.values()) == 70
