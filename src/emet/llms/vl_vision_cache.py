# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""LRU cache for vision image embeds (safe reuse across EQA questions)."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class VisionPrefixCacheEntry:
    """Cached vision embeds or (experimental) prefix KV for a fixed image hash."""

    past_key_values: Any
    prefix_token_len: int
    image_embeds: Any | None = None


class VisionPrefixCache:
    """Keyed by (model_id, resize_side, image content hash)."""

    def __init__(self, max_entries: int = 8):
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict[str, VisionPrefixCacheEntry] = OrderedDict()

    @staticmethod
    def make_key(*, model_id: str, resize_side: int, image_bytes: bytes) -> str:
        h = hashlib.sha256(image_bytes).hexdigest()[:32]
        return f"{model_id}|{int(resize_side)}|{h}"

    def get(self, key: str) -> VisionPrefixCacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return entry

    def put(self, key: str, *, past_key_values: Any = None, prefix_token_len: int = 0, image_embeds: Any = None) -> None:
        self._store[key] = VisionPrefixCacheEntry(
            past_key_values=past_key_values,
            prefix_token_len=int(prefix_token_len),
            image_embeds=image_embeds,
        )
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
