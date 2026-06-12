# Copyright (c) Hello Robot, Inc. All rights reserved.

"""Factory wiring for Gemma VLM quantization (no model load)."""

from __future__ import annotations

from unittest.mock import patch

from emet.llms.vllm_factory import create_dynamem_vllm


def test_create_gemma4_passes_quantization():
    with patch("emet.llms.gemma4_vllm_client.Gemma4VLLMClient") as mock_cls:
        create_dynamem_vllm(
            "gemma4",
            hf_model_id="google/gemma-4-E4B-it",
            vl_model_size="4B",
            max_tokens=64,
            device="cuda",
            quantization="int4",
            prompt=None,
        )
    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["quantization"] == "int4"
    assert kwargs["hf_model_id"] == "google/gemma-4-E4B-it"
