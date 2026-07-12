# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import torch

from emet.llms.prefix_kv_cache import (
    PrefixKVCache,
    PrefixKVCacheEntry,
    clone_past_key_values,
    system_prompt_cache_key,
)


def test_system_prompt_cache_key_stable():
    k1 = system_prompt_cache_key("You are Virgil.")
    k2 = system_prompt_cache_key("You are Virgil.")
    k3 = system_prompt_cache_key("Other prompt")
    assert k1 == k2
    assert k1 != k3


def test_prefix_kv_cache_lru_eviction():
    cache = PrefixKVCache(max_entries=2)
    e1 = PrefixKVCacheEntry(past_key_values=None, prefix_token_len=3, prefix_token_ids=torch.tensor([[1, 2, 3]]))
    e2 = PrefixKVCacheEntry(past_key_values=None, prefix_token_len=2, prefix_token_ids=torch.tensor([[4, 5]]))
    e3 = PrefixKVCacheEntry(past_key_values=None, prefix_token_len=1, prefix_token_ids=torch.tensor([[6]]))
    cache.put("a", e1)
    cache.put("b", e2)
    cache.get("a")
    cache.put("c", e3)
    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None


def test_clone_past_key_values_tuple():
    k = torch.tensor([1.0])
    v = torch.tensor([2.0])
    cloned = clone_past_key_values(((k, v),))
    assert cloned is not None
    assert cloned[0][0] is not k
    assert cloned[0][1] is not v
    assert torch.equal(cloned[0][0], k)
    assert torch.equal(cloned[0][1], v)


def test_clone_past_key_values_returns_none_on_deepcopy_failure(monkeypatch):
    class UnclonableCache:
        pass

    obj = UnclonableCache()

    def _boom(_x):
        raise RuntimeError("cannot deepcopy")

    monkeypatch.setattr("copy.deepcopy", _boom)
    assert clone_past_key_values(obj) is None
    assert clone_past_key_values(obj) is not obj


def test_clone_past_key_values_dynamic_cache():
    from transformers.cache_utils import DynamicCache

    c = DynamicCache()
    k = torch.zeros(1, 2, 4, 8)
    v = torch.zeros(1, 2, 4, 8)
    c.update(k, v, layer_idx=0)
    cloned = clone_past_key_values(c)
    assert cloned is not c
    assert cloned.get_seq_length() == 4
    c.update(torch.ones(1, 2, 1, 8), torch.ones(1, 2, 1, 8), layer_idx=0)
    assert c.get_seq_length() == 5
    assert cloned.get_seq_length() == 4
