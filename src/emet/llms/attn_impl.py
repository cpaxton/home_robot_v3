# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Pick a Transformers ``attn_implementation`` for local VL / LLM loads."""

from __future__ import annotations

import os
from typing import Literal

AttnImpl = Literal["flash_attention_2", "sdpa", "eager"]

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def env_allow_sdpa_attn() -> bool:
    """Escape hatch: permit SDPA when Flash-Attn 2 is missing (slower long-vision decode)."""
    return os.environ.get("EMET_ALLOW_SDPA_ATTN", "").strip().lower() in _TRUE


def env_require_flash_attn() -> bool | None:
    """Explicit override for requiring Flash-Attn 2 on CUDA.

    ``None`` means use the default policy (require on CUDA unless ``EMET_ALLOW_SDPA_ATTN=1``).
    """
    v = os.environ.get("EMET_REQUIRE_FLASH_ATTN", "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def flash_attn_2_available() -> bool:
    """True when Transformers can use Flash-Attn 2 (package importable / registered)."""
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            return True
    except Exception:
        pass
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_attn_implementation(
    *,
    prefer_flash: bool = True,
    device: str = "cuda",
    require_flash: bool | None = None,
) -> AttnImpl:
    """Return the attention backend to pass to ``from_pretrained``.

    On CUDA with ``prefer_flash=True`` (default for VL loads), Flash-Attn 2 is **required**
    unless the caller passes ``require_flash=False`` or sets ``EMET_ALLOW_SDPA_ATTN=1`` /
    ``EMET_REQUIRE_FLASH_ATTN=0``. Silent SDPA fallback hid multi-minute vision decode
    regressions; fail loud instead.

    ``EMET_ALLOW_SDPA_ATTN=1`` is an explicit escape hatch: return ``sdpa`` even when
    Flash-Attn 2 is installed (HM-EQA / stuck-FA2 recovery). Without that env, flash
    wins when available.

    Order when flash is not required: ``flash_attention_2`` → ``sdpa`` → ``eager`` (CPU).
    """
    if not str(device).startswith("cuda"):
        return "eager"

    # Explicit escape: force SDPA even if flash is installed (do not early-return FA2).
    if env_allow_sdpa_attn():
        return "sdpa"

    has_flash = flash_attn_2_available() if prefer_flash else False
    if prefer_flash and has_flash:
        return "flash_attention_2"

    if require_flash is None:
        env_req = env_require_flash_attn()
        if env_req is not None:
            require_flash = env_req
        else:
            # Default: require flash on CUDA VL unless explicitly allowed to use SDPA.
            require_flash = prefer_flash and not env_allow_sdpa_attn()

    if prefer_flash and require_flash and not has_flash:
        raise RuntimeError(
            "Flash-Attn 2 is required for CUDA VL / EQA loads but is not available "
            f"(device={device!r}). Install a matching flash-attn wheel for this torch/CUDA, "
            "or set EMET_ALLOW_SDPA_ATTN=1 / EMET_REQUIRE_FLASH_ATTN=0 to permit the slower "
            "PyTorch SDPA path (long multi-image EQA decode can take tens of seconds per token)."
        )

    return "sdpa"
