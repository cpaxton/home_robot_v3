# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from emet.llms.vllm_factory import eqa_prefix_cache_kwargs


def test_eqa_prefix_cache_kwargs_default_on(monkeypatch):
    monkeypatch.delenv("EMET_VL_CACHE_SYSTEM_PREFIX", raising=False)
    kw = eqa_prefix_cache_kwargs({})
    assert kw["cache_system_prefix"] is True
    assert kw["max_cached_prefixes"] == 1


def test_eqa_prefix_cache_kwargs_env_off(monkeypatch):
    monkeypatch.setenv("EMET_VL_CACHE_SYSTEM_PREFIX", "0")
    kw = eqa_prefix_cache_kwargs({"vl_cache_system_prefix": True})
    assert kw["cache_system_prefix"] is False
