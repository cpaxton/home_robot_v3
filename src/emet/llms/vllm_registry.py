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

"""Supported local VLLMs for DynaMem: defaults, dedup policy, and runtime config matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# HM-EQA bake-off winner on a single 24 GB GPU (canonical-6, Dynagraph + debias).
DEFAULT_QWEN3_VL_HF_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass(frozen=True)
class VLLMRegistryEntry:
    """One row in the supported-VLLM table (``dynav_config.yaml`` ``eqa:`` ``vl_family``)."""

    family_key: str
    """Canonical ``vl_family`` string (e.g. ``qwen3_vl``)."""

    default_hf_model_id: str
    """Default Hugging Face repo id when ``vl_hf_model_id`` is unset."""

    supports_dedup: bool
    """If True, one loaded instance may serve caption, keyword, and EQA roles (per-call prompts)."""


# Logical keys match normalized ``vl_family`` from config (see ``normalize_vl_family``).
SUPPORTED_VLLMS: dict[str, VLLMRegistryEntry] = {
    "qwen3_vl": VLLMRegistryEntry(
        family_key="qwen3_vl",
        default_hf_model_id=DEFAULT_QWEN3_VL_HF_MODEL_ID,
        supports_dedup=True,
    ),
    "qwen3_5": VLLMRegistryEntry(
        family_key="qwen3_5",
        default_hf_model_id="Qwen/Qwen3.5-9B",
        supports_dedup=True,
    ),
    "qwen2_5_vl": VLLMRegistryEntry(
        family_key="qwen2_5_vl",
        default_hf_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        supports_dedup=True,
    ),
    "gemma4": VLLMRegistryEntry(
        family_key="gemma4",
        default_hf_model_id="google/gemma-4-E4B-it",
        supports_dedup=True,
    ),
}


def normalize_vl_family(family: str) -> str:
    """Map config aliases to registry keys."""
    f = (family or "").strip().lower().replace("-", "_").replace(".", "_")
    if f in ("qwen25_vl", "qwen2_5vl"):
        return "qwen2_5_vl"
    if f in ("qwen35", "qwen3_5_vl"):
        return "qwen3_5"
    return f


def registry_entry(family: str) -> VLLMRegistryEntry | None:
    """Return the registry row for *family* after normalization, or None if unknown."""
    return SUPPORTED_VLLMS.get(normalize_vl_family(family))


def default_hf_model_id(family: str) -> str | None:
    """Default HF checkpoint id for *family*, or None if unknown."""
    ent = registry_entry(family)
    return ent.default_hf_model_id if ent else None


def supports_dedup_for_family(family: str) -> bool:
    """Whether one physical load may back both caption and EQA paths for this family.

    Unknown families return True so existing custom ``vl_family`` strings keep the single-factory behavior.
    """
    ent = registry_entry(family)
    if ent is None:
        return True
    return ent.supports_dedup


@dataclass(frozen=True)
class VLLMRunConfig:
    """Config slice used to decide if two consumers can share one loaded model."""

    family: str
    hf_model_id: str | None
    device: str
    quantization: str | None


def should_share_vllm(a: VLLMRunConfig, b: VLLMRunConfig) -> bool:
    """Return True when both requests map to the same weights/runtime and dedup is allowed for both families."""
    if not supports_dedup_for_family(a.family) or not supports_dedup_for_family(b.family):
        return False
    if normalize_vl_family(a.family) != normalize_vl_family(b.family):
        return False
    if (a.hf_model_id or "") != (b.hf_model_id or ""):
        return False
    if a.device != b.device:
        return False
    if (a.quantization or "") != (b.quantization or ""):
        return False
    return True


def config_from_client(client: Any) -> VLLMRunConfig:
    """Derive a :class:`VLLMRunConfig` from an :class:`emet.llms.base.AbstractVLLMClient` (see ``canonical_model_key``)."""
    from emet.llms.base import AbstractVLLMClient

    if not isinstance(client, AbstractVLLMClient):
        raise TypeError("expected AbstractVLLMClient")
    key = client.canonical_model_key
    parts = key.split(":", 3)
    prefix = parts[0] if parts else ""
    if prefix in ("qwen25_vl", "qwen3_vl", "qwen3_5") and len(parts) >= 4:
        return VLLMRunConfig(
            family=prefix,
            hf_model_id=parts[1],
            device=parts[2],
            quantization=parts[3] if parts[3] else None,
        )
    if prefix == "gemma4" and len(parts) >= 4:
        return VLLMRunConfig(
            family="gemma4",
            hf_model_id=parts[1],
            device=parts[2],
            quantization=parts[3] if parts[3] else None,
        )
    if prefix == "gemma4" and len(parts) >= 3:
        return VLLMRunConfig(
            family="gemma4",
            hf_model_id=parts[1],
            device=parts[2],
            quantization=None,
        )
    return VLLMRunConfig(family=prefix or "unknown", hf_model_id=None, device="unknown", quantization=None)
