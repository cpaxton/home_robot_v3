# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""qwen35-* must use Qwen35Client (Qwen3_5ForConditionalGeneration), not CausalLM pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from emet.llms import get_llm_client


@patch("emet.llms.qwen3_5_client.Qwen35Client")
def test_qwen35_4b_routes_to_qwen35_client(mock_cls):
    mock_cls.return_value = MagicMock(name="qwen35")
    client = get_llm_client("qwen35-4B", prompt="sys", device="cuda", max_tokens=128)
    assert client is mock_cls.return_value
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["hf_model_id"] == "Qwen/Qwen3.5-4B"
    assert kwargs["quantization"] == "int4"
    assert kwargs["device"] == "cuda"
    assert kwargs["max_tokens"] == 128
    assert kwargs["cache_system_prefix"] is False


@patch("emet.llms.qwen_client.Qwen25Client")
def test_qwen25_still_uses_qwen25_client(mock_cls):
    # get_llm_client imports Qwen25Client lazily from emet.llms.qwen_client; patching
    # emet.llms.Qwen25Client would miss and construct a real client (HF download).
    mock_cls.return_value = MagicMock(name="qwen25")
    client = get_llm_client("qwen25-3B-Instruct-Int4", prompt="sys", device="cuda")
    assert client is mock_cls.return_value
    mock_cls.assert_called_once()
