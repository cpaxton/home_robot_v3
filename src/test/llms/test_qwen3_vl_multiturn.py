# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from unittest.mock import MagicMock, patch

import pytest
import torch

from emet.llms.prefix_kv_cache import PrefixKVCache
from emet.llms.qwen3_vl_client import Qwen3VLClient, generation_metadata


def _make_client(*, cache_system_prefix: bool = True) -> Qwen3VLClient:
    client = Qwen3VLClient.__new__(Qwen3VLClient)
    client.conversation_history = []
    client._prompt = "You are Virgil."
    client.cache_system_prefix = cache_system_prefix
    client._prefix_cache = PrefixKVCache(max_entries=1)
    client.max_tokens = 64
    client.num_beams = 1
    client._device = "cpu"
    client._quantization = None
    client._TEMPLATE_KWARGS = {}
    client.processor = MagicMock()
    client.model = MagicMock()
    client.model.generation_config.eos_token_id = 99
    _p = torch.nn.Parameter(torch.zeros(1))
    client.model.parameters = MagicMock(return_value=iter([_p]))
    # Re-create iterator on each call so _model_device() can be invoked repeatedly.
    client.model.parameters.side_effect = lambda: iter([_p])
    return client


def test_has_system_in_history():
    client = _make_client()
    assert client._has_system_in_history() is False
    client.add_history({"role": "system", "content": "sys"})
    assert client._has_system_in_history() is True


def test_explicit_dialogue_encodes_prior_images_without_losing_assistant_turns():
    import numpy as np

    client = _make_client()
    client._iterations = 0
    client._downsample_image = lambda image: np.asarray(image)
    seen = []
    client.generate_multimodal = lambda content, **kwargs: seen.append(client.get_history()) or "reply"
    client.generate_messages(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": ["earlier", np.zeros((4, 4, 3), dtype=np.uint8)]},
            {"role": "assistant", "content": "previous decision"},
            {"role": "user", "content": "TOOL_RESULT failed approach"},
            {"role": "user", "content": "current state"},
        ]
    )
    assert seen[0][1]["content"][1]["type"] == "image"
    assert seen[0][2] == {"role": "assistant", "content": "previous decision"}
    assert client.get_history() == []


def test_separate_evidence_images_keep_their_own_preprocessing_budget():
    import numpy as np

    client = _make_client()
    client.image_max_side, client.image_max_pixels = 512, 0
    images = [np.full((480, 640, 3), index * 60, dtype=np.uint8) for index in range(4)]
    encoded = client._process_input(["CURRENT and three historical views", *images])
    processed = [part["image"] for part in encoded if part["type"] == "image"]
    assert len(processed) == 4
    assert all(image.size == (512, 384) for image in processed)
    assert [np.array(image)[0, 0, 0] for image in processed] == [0, 60, 120, 180]


def test_prefix_ids_align():
    client = _make_client()
    full = torch.tensor([[1, 2, 3, 4, 5]])
    from emet.llms.prefix_kv_cache import PrefixKVCacheEntry

    entry = PrefixKVCacheEntry(
        past_key_values=None,
        prefix_token_len=3,
        prefix_token_ids=torch.tensor([[1, 2, 3]]),
    )
    assert client._prefix_ids_align(full, entry) is True
    bad = PrefixKVCacheEntry(
        past_key_values=None,
        prefix_token_len=3,
        prefix_token_ids=torch.tensor([[9, 9, 9]]),
    )
    assert client._prefix_ids_align(full, bad) is False


@patch("emet.llms.qwen3_vl_client.env_agent_model_debug", return_value=False)
def test_generate_multimodal_reset_false_does_not_duplicate_system(_mock_dbg):
    client = _make_client(cache_system_prefix=False)
    client.reset = MagicMock()
    client._processor_inputs = MagicMock(
        return_value=MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            to=MagicMock(return_value=MagicMock(input_ids=torch.tensor([[1, 2, 3]]))),
        )
    )
    client._generate_ids = MagicMock(return_value=torch.tensor([[1, 2, 3, 99]]))
    client.processor.batch_decode.return_value = ["tool json"]
    client.add_history({"role": "system", "content": "sys"})
    client.add_history({"role": "user", "content": "hi"})
    client.add_history({"role": "assistant", "content": "ok"})

    client.generate_multimodal("followup", system_prompt="sys", reset_context=False)

    roles = [m["role"] for m in client.conversation_history if isinstance(m, dict)]
    assert roles.count("system") == 1
    assert roles[-2:] == ["user", "assistant"]
    assert client.last_generation_metadata["usage"] == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    assert client.last_generation_metadata["finish_reason"] == "stop"


@pytest.mark.parametrize("tokens,expected", [([1, 2, 3], "length"), ([1, 2, 99], "stop"), ([1, 99], "stop")])
def test_decode_receipt_distinguishes_budget_exhaustion_from_eos(tokens, expected):
    receipt = generation_metadata(
        prompt_tokens=5, new_token_ids=torch.tensor(tokens), max_new_tokens=3, eos_token_id=[98, 99]
    )
    assert receipt["finish_reason"] == expected
    assert receipt["finish_reason_verified"] is True
    assert receipt["usage"]["total_tokens"] == 5 + len(tokens)


@patch("emet.llms.qwen3_vl_client.env_agent_model_debug", return_value=False)
def test_generate_ids_passes_full_prompt_with_past(_mock_dbg):
    """transformers 5.x crops by past_length; do not pre-slice to an empty suffix."""
    client = _make_client(cache_system_prefix=True)
    past = MagicMock()
    past.get_seq_length.return_value = 3
    full = MagicMock()
    full.input_ids = torch.tensor([[10, 11, 12, 20, 21]])
    full.items = MagicMock(
        return_value=[
            ("input_ids", full.input_ids),
            ("attention_mask", torch.ones(1, 5, dtype=torch.long)),
            ("mm_token_type_ids", torch.zeros(1, 5, dtype=torch.long)),
        ]
    )
    captured: dict = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[10, 11, 12, 20, 21, 99]])

    client.model.generate = _fake_generate
    with patch("emet.llms.qwen3_vl_client.clone_past_key_values", return_value=past) as mock_clone:
        out = client._generate_ids(full, max_new_tokens=8, past_key_values=past, prefix_len=3)

    assert out.shape[-1] == 6
    mock_clone.assert_called_once()
    assert torch.equal(captured["input_ids"], full.input_ids)
    assert int(captured["attention_mask"].shape[1]) == 5
    assert captured["past_key_values"] is past


@patch("emet.llms.qwen3_vl_client.env_agent_model_debug", return_value=False)
def test_generate_ids_with_prefix_guard_falls_back_on_error(_mock_dbg):
    client = _make_client(cache_system_prefix=True)
    past = MagicMock()
    full = MagicMock()
    full.input_ids = torch.tensor([[10, 11, 12, 20, 21]])
    full.items = MagicMock(
        return_value=[
            ("input_ids", full.input_ids),
            ("attention_mask", torch.ones(1, 5, dtype=torch.long)),
        ]
    )
    calls = {"n": 0}

    def _gen(**kwargs):
        calls["n"] += 1
        if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
            raise RuntimeError("bad cache")
        return torch.tensor([[10, 11, 12, 20, 21, 99]])

    client.model.generate = _gen
    with patch("emet.llms.qwen3_vl_client.clone_past_key_values", return_value=past):
        out, used = client._generate_ids_with_prefix_guard(full, max_new_tokens=8, past_key_values=past, prefix_len=3)
    assert used is False
    assert client.cache_system_prefix is False
    assert out.shape[-1] == 6
    assert calls["n"] == 2


@patch("emet.llms.qwen3_vl_client.env_agent_model_debug", return_value=False)
def test_generate_multimodal_uses_prefix_cache_on_second_call(_mock_dbg):
    client = _make_client(cache_system_prefix=True)
    past = ((torch.tensor([1.0]), torch.tensor([2.0])),)
    prefix_ids = torch.tensor([[10, 11, 12]])
    from emet.llms.prefix_kv_cache import PrefixKVCacheEntry

    client._prefix_cache.put(
        "key",
        PrefixKVCacheEntry(past_key_values=past, prefix_token_len=3, prefix_token_ids=prefix_ids),
    )

    with patch.object(client, "_ensure_prefix_cached") as mock_ensure:
        mock_ensure.return_value = client._prefix_cache.get("key")
        full_inputs = MagicMock()
        full_inputs.input_ids = torch.tensor([[10, 11, 12, 20, 21]])
        full_inputs.to = MagicMock(return_value=full_inputs)
        full_inputs.items = MagicMock(
            return_value=[
                ("input_ids", full_inputs.input_ids),
                ("attention_mask", torch.ones(1, 5, dtype=torch.long)),
            ]
        )
        client._processor_inputs = MagicMock(return_value=full_inputs)
        client._prefix_ids_align = MagicMock(return_value=True)
        client._generate_ids_with_prefix_guard = MagicMock(
            return_value=(torch.tensor([[10, 11, 12, 20, 21, 99]]), True)
        )
        client.processor.batch_decode.return_value = ["answer"]

        out = client.generate_multimodal("hello", system_prompt="sys", reset_context=True)

    assert out == "answer"
    client._generate_ids_with_prefix_guard.assert_called_once()
    assert client._generate_ids_with_prefix_guard.call_args.kwargs.get("past_key_values") is past
    assert client._generate_ids_with_prefix_guard.call_args.kwargs.get("prefix_len") == 3
