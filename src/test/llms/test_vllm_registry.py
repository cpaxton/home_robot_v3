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
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.

"""Unit tests for VLLM registry deduplication (no model load)."""

from typing import Any

from emet.llms.base import AbstractVLLMClient
from emet.llms.vllm_registry import (
    SUPPORTED_VLLMS,
    VLLMRunConfig,
    config_from_client,
    default_hf_model_id,
    normalize_vl_family,
    registry_entry,
    should_share_vllm,
)


def test_should_share_same_config():
    a = VLLMRunConfig("qwen3_vl", "Qwen/Qwen3-VL-4B-Instruct", "cuda", "int4")
    b = VLLMRunConfig("qwen3_vl", "Qwen/Qwen3-VL-4B-Instruct", "cuda", "int4")
    assert should_share_vllm(a, b)


def test_should_not_share_different_hf():
    a = VLLMRunConfig("qwen3_vl", "Qwen/Qwen3-VL-4B-Instruct", "cuda", "int4")
    b = VLLMRunConfig("qwen3_vl", "Qwen/Qwen3-VL-8B-Instruct", "cuda", "int4")
    assert not should_share_vllm(a, b)


def test_should_not_share_different_family():
    a = VLLMRunConfig("qwen3_vl", "Qwen/Qwen3-VL-4B-Instruct", "cuda", "int4")
    b = VLLMRunConfig("qwen25_vl", "Qwen/Qwen2.5-VL-3B-Instruct", "cuda", "int4")
    assert not should_share_vllm(a, b)


def test_should_share_qwen25_alias_with_qwen2_5_when_same_weights():
    """``qwen25_vl`` normalizes to the same registry family as ``qwen2_5_vl``."""
    a = VLLMRunConfig("qwen25_vl", "Qwen/Qwen2.5-VL-3B-Instruct", "cuda", "int4")
    b = VLLMRunConfig("qwen2_5_vl", "Qwen/Qwen2.5-VL-3B-Instruct", "cuda", "int4")
    assert should_share_vllm(a, b)


def test_should_not_share_different_device():
    a = VLLMRunConfig("qwen3_vl", "Qwen/Qwen3-VL-4B-Instruct", "cuda", "int4")
    b = VLLMRunConfig("qwen3_vl", "Qwen/Qwen3-VL-4B-Instruct", "cpu", "int4")
    assert not should_share_vllm(a, b)


def test_normalize_vl_family_aliases():
    assert normalize_vl_family("qwen25_vl") == "qwen2_5_vl"
    assert normalize_vl_family("Qwen3_VL") == "qwen3_vl"
    assert normalize_vl_family("qwen3.5") == "qwen3_5"
    assert normalize_vl_family("qwen3-5") == "qwen3_5"
    assert normalize_vl_family("qwen35") == "qwen3_5"


def test_default_hf_model_id_known_families():
    assert "Qwen3" in (default_hf_model_id("qwen3_vl") or "")
    assert default_hf_model_id("gemma4") is not None


def test_supported_vllms_table_has_expected_keys():
    for k in ("qwen3_vl", "qwen3_5", "qwen2_5_vl", "gemma4"):
        assert k in SUPPORTED_VLLMS
        assert registry_entry(k) is not None
        assert registry_entry(k).supports_dedup is True


class _StubVLLM(AbstractVLLMClient):
    def __init__(self, canonical_model_key: str):
        super().__init__(None)
        self._canonical_model_key = canonical_model_key

    @property
    def canonical_model_key(self) -> str:
        return self._canonical_model_key

    def generate_multimodal(
        self,
        user_content: str | list[Any],
        *,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        reset_context: bool = True,
        verbose: bool = False,
        image=None,
    ) -> str:
        return ""


def test_config_from_client_qwen3():
    c = _StubVLLM("qwen3_vl:Qwen/Qwen3-VL-4B-Instruct:cuda:int4")
    cfg = config_from_client(c)
    assert cfg.family == "qwen3_vl"
    assert cfg.hf_model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert cfg.device == "cuda"
    assert cfg.quantization == "int4"


def test_config_from_client_qwen3_5():
    c = _StubVLLM("qwen3_5:Qwen/Qwen3.5-9B:cuda:int4")
    cfg = config_from_client(c)
    assert cfg.family == "qwen3_5"
    assert cfg.hf_model_id == "Qwen/Qwen3.5-9B"
    assert cfg.quantization == "int4"


def test_config_from_client_gemma4():
    c = _StubVLLM("gemma4:google/gemma-3-4b-it:cuda:int4")
    cfg = config_from_client(c)
    assert cfg.family == "gemma4"
    assert cfg.hf_model_id == "google/gemma-3-4b-it"
    assert cfg.device == "cuda"
    assert cfg.quantization == "int4"


def test_vllm_id_alias_matches_canonical():
    c = _StubVLLM("qwen3_vl:Qwen/Qwen3-VL-4B-Instruct:mps:none")
    assert c.vllm_id == c.canonical_model_key
