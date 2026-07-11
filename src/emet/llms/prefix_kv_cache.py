# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""System-prompt prefix KV cache for Hugging Face VL clients."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


@dataclass
class PrefixKVCacheEntry:
    """Cached prefill state for a fixed system-prompt token prefix."""

    past_key_values: Any
    prefix_token_len: int
    prefix_token_ids: torch.Tensor


def env_vl_cache_system_prefix() -> bool | None:
    """When set, overrides config default for system-prefix KV cache (``0``/``1``)."""
    v = os.environ.get("EMET_VL_CACHE_SYSTEM_PREFIX", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return None


def system_prompt_cache_key(system_prompt: str) -> str:
    """Stable cache key for a system prompt string."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def clone_past_key_values(past_key_values: Any) -> Any:
    """Return a detached copy of ``past_key_values`` for safe reuse across forwards.

    ``generate`` mutates the cache in place, so the stored system prefix must be
    cloned on every hit. ``DynamicCache`` (transformers 5.x) has no ``.copy()``.
    """
    if past_key_values is None:
        return None
    # transformers Cache API with explicit copy/clone
    for attr in ("clone", "copy"):
        fn = getattr(past_key_values, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(past_key_values, (tuple, list)):
        cloned: list[Any] = []
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)) and len(layer) == 2:
                k, v = layer
                cloned.append(
                    (
                        k.detach().clone() if torch.is_tensor(k) else k,
                        v.detach().clone() if torch.is_tensor(v) else v,
                    )
                )
            else:
                cloned.append(layer)
        return tuple(cloned)
    # DynamicCache / other Cache subclasses: deepcopy preserves layer tensors.
    try:
        import copy

        return copy.deepcopy(past_key_values)
    except Exception:
        return past_key_values


class PrefixKVCache:
    """Small LRU cache keyed by :func:`system_prompt_cache_key`."""

    def __init__(self, max_entries: int = 1) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, PrefixKVCacheEntry] = OrderedDict()

    def get(self, key: str) -> PrefixKVCacheEntry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, key: str, entry: PrefixKVCacheEntry) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
