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
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.

"""Gemma 4 any-to-any presets: routing must not land in legacy Gemma3 ``GemmaClient``."""

from unittest import mock

from emet.llms import (
    GEMMA4_PRESETS,
    Gemma4AnyToAnyClient,
    Gemma4VLLMClient,
    GemmaClient,
    get_llm_client,
    is_vl_llm_key,
)
from emet.llms.gemma4_any_client import _extract_text_from_any_to_any_output


def test_gemma4_presets_point_at_hf_ids():
    assert "e2b" in GEMMA4_PRESETS["gemma4-e2b"].lower()
    assert "E4B" in GEMMA4_PRESETS["gemma4-e4b"] or "e4b" in GEMMA4_PRESETS["gemma4-e4b"].lower()


@mock.patch("emet.llms.gemma4_any_client.pipeline")
def test_get_llm_client_gemma4_e4b_uses_any_client(mock_pipe):
    mock_pipe.return_value = object()
    c = get_llm_client("gemma4-e4b", "system prompt", device="cpu", max_tokens=64)
    assert isinstance(c, Gemma4AnyToAnyClient)
    assert c.hf_model_id == GEMMA4_PRESETS["gemma4-e4b"]
    assert not isinstance(c, GemmaClient)


@mock.patch("emet.llms.gemma4_any_client.pipeline")
def test_get_llm_client_gemma4_case_insensitive(mock_pipe):
    mock_pipe.return_value = object()
    c = get_llm_client("Gemma4-E2B", "p", device="cpu")
    assert isinstance(c, Gemma4AnyToAnyClient)
    assert c.hf_model_id == GEMMA4_PRESETS["gemma4-e2b"]


def test_extract_text_from_empty():
    assert _extract_text_from_any_to_any_output([]) == ""


def test_extract_text_string_chunk():
    assert _extract_text_from_any_to_any_output([{"generated_text": "  hi  "}]) == "hi"


def test_extract_text_list_content():
    out = [{"generated_text": [{"content": "x"}]}]
    assert _extract_text_from_any_to_any_output(out) == "x"


def test_gemma4_vlm_presets_in_llm_choices():
    from emet.llms import get_llm_choices

    choices = get_llm_choices()
    assert "gemma4-vlm-e4b" in choices
    assert "gemma4-vl-eqa" in choices


@mock.patch.object(Gemma4VLLMClient, "__init__", lambda self, *args, **kwargs: None)
def test_get_llm_client_gemma4_vlm_e4b():
    c = get_llm_client("gemma4-vlm-e4b", "system prompt", device="cpu", max_tokens=64)
    assert isinstance(c, Gemma4VLLMClient)


def test_is_vl_llm_key():
    assert is_vl_llm_key("gemma4-vlm-e4b")
    assert is_vl_llm_key("gemma4-vl-eqa")
    assert is_vl_llm_key("qwen3-vl-eqa")
    assert not is_vl_llm_key("gemma4-e4b")
    assert not is_vl_llm_key("qwen35-9B")


@mock.patch("emet.llms.vllm_factory.create_dynamem_vllm")
def test_get_llm_client_gemma4_vl_eqa(mock_create):
    mock_create.return_value = object()
    get_llm_client("gemma4-vl-eqa", "p", device="cpu", parameters={"eqa": {"vl_family": "gemma4"}})
    mock_create.assert_called_once()
    assert mock_create.call_args[0][0] == "gemma4"
