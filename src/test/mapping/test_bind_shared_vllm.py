# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""bind_shared_vllm_from_agent must only share matching EQA checkpoints (not text routers)."""

from __future__ import annotations

from typing import Any

from emet.llms.base import AbstractVLLMClient
from emet.llms.vllm_registry import DEFAULT_QWEN3_VL_HF_MODEL_ID
from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


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


def _pending_vm(*, family: str = "qwen3_vl", hf: str | None = None, quant: str = "int4") -> SparseVoxelMap:
    vm = object.__new__(SparseVoxelMap)
    vm.run_eqa = True
    vm._eqa_backend = "qwen_vl"
    vm._eqa_device_resolved = "cuda"
    vm.image_description_client = None
    vm.eqa_client = None
    vm._eqa_pending = {
        "vl_family": family,
        "eqa_vl_hf_model_id": hf,
        "eqa_vl_model_size": "8B",
        "eqa_vl_max_tokens": 512,
        "eqa_vl_quantization": quant,
        "gemini_model": "gemini-2.5-flash",
    }
    return vm


def test_bind_rejects_qwen35_router_when_eqa_is_qwen3_vl_8b():
    """Default agent --llm qwen35-4B must not steal deferred Qwen3-VL-8B captions."""
    vm = _pending_vm(hf=DEFAULT_QWEN3_VL_HF_MODEL_ID)
    router = _StubVLLM("qwen3_5:Qwen/Qwen3.5-4B:cuda:int4")
    assert vm.bind_shared_vllm_from_agent(router) is False
    assert vm._eqa_pending is not None
    assert vm.image_description_client is None
    assert vm.eqa_client is None


def test_bind_accepts_matching_qwen3_vl_eqa_client():
    vm = _pending_vm(hf=DEFAULT_QWEN3_VL_HF_MODEL_ID)
    eqa = _StubVLLM(f"qwen3_vl:{DEFAULT_QWEN3_VL_HF_MODEL_ID}:cuda:int4")
    assert vm.bind_shared_vllm_from_agent(eqa) is True
    assert vm._eqa_pending is None
    assert vm.image_description_client is eqa
    assert vm.eqa_client is eqa


def test_bind_rejects_mismatched_quantization():
    vm = _pending_vm(hf=DEFAULT_QWEN3_VL_HF_MODEL_ID, quant="int4")
    eqa = _StubVLLM(f"qwen3_vl:{DEFAULT_QWEN3_VL_HF_MODEL_ID}:cuda:none")
    assert vm.bind_shared_vllm_from_agent(eqa) is False
    assert vm._eqa_pending is not None
