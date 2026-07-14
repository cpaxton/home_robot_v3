# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Optional integration smoke: real HF VLM load (excluded from default pytest via marker)."""

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.vllm_load


@pytest.mark.timeout(600)
def test_qwen3_vl_minimal_generate():
    """Loads Qwen3-VL (tiny default or VLLM_LOAD_TEST_MODEL) and runs one text-only turn."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for VLLM load smoke")
    mid = os.environ.get("VLLM_LOAD_TEST_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    from emet.llms.qwen3_vl_client import Qwen3VLClient

    client = Qwen3VLClient(
        prompt=None,
        hf_model_id=mid,
        max_tokens=16,
        device="cuda",
        quantization="int4",
    )
    out = client.generate_multimodal(
        "Say OK.",
        system_prompt="You are a terse assistant.",
        max_new_tokens=8,
        reset_context=True,
    )
    assert isinstance(out, str)
    assert len(out) > 0


@pytest.mark.timeout(900)
def test_qwen3_vl_prefix_cache_second_turn():
    """Two turns with the same system prompt; second call should hit prefix cache."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for VLLM load smoke")
    mid = os.environ.get("VLLM_LOAD_TEST_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    from emet.llms.qwen3_vl_client import Qwen3VLClient

    client = Qwen3VLClient(
        prompt=None,
        hf_model_id=mid,
        max_tokens=16,
        device="cuda",
        quantization="int4",
        cache_system_prefix=True,
    )
    sys_p = "You are a terse assistant."
    client.generate_multimodal("Say OK.", system_prompt=sys_p, max_new_tokens=8, reset_context=True)
    assert len(client._prefix_cache) == 1
    client.generate_multimodal(
        "Say OK again.",
        system_prompt=sys_p,
        max_new_tokens=8,
        reset_context=True,
    )
    assert len(client._prefix_cache) == 1


@pytest.mark.timeout(600)
def test_gemma_multimodal_minimal_generate():
    """Loads Gemma multimodal pipeline (default small IT checkpoint)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for VLLM load smoke")
    mid = os.environ.get("VLLM_LOAD_TEST_GEMMA_MODEL", "google/gemma-3-4b-it")
    from emet.llms.gemma4_vllm_client import Gemma4VLLMClient

    client = Gemma4VLLMClient(hf_model_id=mid, max_tokens=16, device="cuda", quantization="int4")
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    out = client.generate_multimodal(
        "What color is this image?",
        system_prompt=None,
        max_new_tokens=16,
        reset_context=True,
        image=arr,
    )
    assert isinstance(out, str)
    assert len(out) > 0
