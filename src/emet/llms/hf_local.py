# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Prefer local Hugging Face cache for ``from_pretrained`` (skip hub metadata checks)."""

from __future__ import annotations

import os
from typing import Any

_TRUE = frozenset({"1", "true", "yes", "on"})


def env_hf_local_only() -> bool:
    """True when ``EMET_HF_LOCAL_ONLY`` or ``HF_HUB_OFFLINE`` requests cache-only loads."""
    for key in ("EMET_HF_LOCAL_ONLY", "HF_HUB_OFFLINE"):
        if os.environ.get(key, "").strip().lower() in _TRUE:
            return True
    return False


def resolve_pretrained_source(model_id: str, *, prefer_local: bool | None = None) -> tuple[str, dict[str, Any]]:
    """Resolve a Hub id to a local snapshot path when the cache is complete.

    Returns ``(source, from_pretrained_kwargs)`` where ``source`` is either the local
    snapshot directory or the original ``model_id``, and kwargs may include
    ``local_files_only=True``.

    When the cache is incomplete:
    - if ``prefer_local`` / ``EMET_HF_LOCAL_ONLY`` / ``HF_HUB_OFFLINE`` is set → re-raise
    - otherwise → return the Hub id with empty kwargs (normal download/check)
    """
    mid = (model_id or "").strip()
    if not mid:
        return model_id, {}
    # Already a filesystem path
    if os.path.isdir(mid):
        return mid, {"local_files_only": True}

    force_local = env_hf_local_only() if prefer_local is None else bool(prefer_local)
    # Default agent behavior: try local first even when env unset (warm cache = no hub RTT).
    try_local = True if prefer_local is None else bool(prefer_local) or force_local

    if not try_local:
        return mid, {}

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        if force_local:
            raise
        return mid, {}

    try:
        path = snapshot_download(mid, local_files_only=True)
        # Incomplete snapshots can still resolve (refs present, few files). Require a
        # processor/tokenizer marker or weight index before forcing local_files_only.
        markers = (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model.safetensors",
            "pytorch_model.bin",
        )
        if not any(os.path.exists(os.path.join(path, name)) for name in markers):
            if force_local:
                raise FileNotFoundError(f"HF cache for {mid!r} at {path} is incomplete (no processor/weights markers)")
            return mid, {}
        return path, {"local_files_only": True}
    except Exception:
        if force_local:
            raise
        return mid, {}


def merge_pretrained_kwargs(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge ``from_pretrained`` kwargs; ``extra`` wins on key conflicts."""
    out = dict(base)
    out.update(extra)
    return out
