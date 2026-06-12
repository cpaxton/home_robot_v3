# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# EQA multimodal (Qwen3.5) options from ``dynav_config.yaml`` under ``eqa_vl:`` and env overrides.

from __future__ import annotations

import os
import subprocess
from typing import Any

from emet.core.parameters import Parameters
from emet.utils.logger import Logger, suppress_hf_hub_http_logging

logger = Logger(__name__)

# Set on first successful resolution; keeps one size for the process even if free VRAM drops after loading SigLIP / first VL.
_resolved_eqa_vl_model_size: str | None = None

_VALID_SIZES = frozenset({"0.8B", "2B", "4B", "9B", "27B"})
_LEGACY_SIZE = {"8B": "9B", "32B": "27B", "7B": "9B"}


def get_nvidia_gpu_free_mib() -> float | None:
    """Return free memory (MiB) for the first NVIDIA GPU, or None if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if out.returncode != 0 or not (out.stdout or "").strip():
            return None
        first = (out.stdout or "").strip().splitlines()[0].strip()
        return float(first)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return None


def _pget(parameters: Parameters | dict | None, key: str, default: Any = None) -> Any:
    if parameters is None:
        return default
    if isinstance(parameters, Parameters):
        return parameters.get(key, default=default)
    keys = key.split("/")
    data: Any = parameters
    for k in keys[:-1]:
        if not isinstance(data, dict) or k not in data:
            return default
        data = data[k]
    if not isinstance(data, dict):
        return default
    return data.get(keys[-1], default)


def _normalize_size(raw: str) -> str:
    u = raw.strip().upper()
    return _LEGACY_SIZE.get(u, u)


def apply_eqa_vl_runtime_settings(parameters: Parameters | dict | None) -> None:
    """Apply ``eqa_vl/verbose_hf`` from config when ``EMET_VERBOSE_HF`` is unset."""
    if os.environ.get("EMET_VERBOSE_HF", "").strip():
        suppress_hf_hub_http_logging()
        return
    if _pget(parameters, "eqa_vl/verbose_hf", False):
        os.environ["EMET_VERBOSE_HF"] = "1"
    suppress_hf_hub_http_logging()


def resolve_eqa_vl_model_size(
    parameters: Parameters | dict | None,
    *,
    device: str,
) -> str:
    """
    Resolve Qwen3.5 size once per process: config ``eqa_vl/model_size``, then env, then VRAM tiers from config.
    """
    global _resolved_eqa_vl_model_size
    if _resolved_eqa_vl_model_size is not None:
        return _resolved_eqa_vl_model_size

    env_override = os.environ.get("EMET_EQA_VL_MODEL_SIZE", "").strip()
    if env_override:
        norm = _normalize_size(env_override)
        if norm in _VALID_SIZES:
            _resolved_eqa_vl_model_size = norm
            logger.info(f"EQA VL model size from EMET_EQA_VL_MODEL_SIZE: {norm}")
            return _resolved_eqa_vl_model_size

    fixed = _pget(parameters, "eqa_vl/model_size", None)
    if fixed is not None and str(fixed).strip() and str(fixed).lower() != "null":
        norm = _normalize_size(str(fixed).strip())
        if norm in _VALID_SIZES:
            _resolved_eqa_vl_model_size = norm
            logger.info(f"EQA VL model size from config eqa_vl/model_size: {norm}")
            return _resolved_eqa_vl_model_size
        logger.warning(f"Invalid eqa_vl/model_size={fixed!r}; using VRAM tiers")

    tier_9 = float(_pget(parameters, "eqa_vl/vram_mib_tier_9b", 20000))
    tier_4 = float(_pget(parameters, "eqa_vl/vram_mib_tier_4b", 11000))

    def pick(free_mib: float | None) -> str:
        if free_mib is None:
            logger.warning("nvidia-smi unavailable; defaulting EQA VL to Qwen3.5-2B")
            return "2B"
        if free_mib >= tier_9:
            logger.info(
                f"GPU free memory ~{free_mib:.0f} MiB (tiers >={tier_9:.0f} / >={tier_4:.0f} MiB): Qwen3.5-9B",
            )
            return "9B"
        if free_mib >= tier_4:
            logger.info(f"GPU free memory ~{free_mib:.0f} MiB (tier >={tier_4:.0f} MiB): Qwen3.5-4B")
            return "4B"
        logger.info(f"GPU free memory ~{free_mib:.0f} MiB: Qwen3.5-2B")
        return "2B"

    if device == "cuda":
        free = get_nvidia_gpu_free_mib()
        size = pick(free)
    else:
        size = pick(None)

    _resolved_eqa_vl_model_size = size
    return _resolved_eqa_vl_model_size


def sync_resolved_eqa_vl_model_size_from_explicit(raw: str) -> str:
    """
    When ``get_shared_qwen35_vl_client(model_size=...)`` is used with an explicit size, record it
    so ``resolve_eqa_vl_model_size`` never re-tiers from a later VRAM sample in the same process.
    """
    global _resolved_eqa_vl_model_size
    norm = _normalize_size(str(raw).strip())
    if norm in _VALID_SIZES:
        _resolved_eqa_vl_model_size = norm
    return norm


def reset_eqa_vl_resolution_for_tests() -> None:
    global _resolved_eqa_vl_model_size
    _resolved_eqa_vl_model_size = None


def get_eqa_vl_int(parameters: Parameters | dict | None, key: str, default: int) -> int:
    """Read ``eqa_vl/<key>`` from parameters (dynav_config) with fallback."""
    v = _pget(parameters, f"eqa_vl/{key}", default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def get_eqa_vl_str(parameters: Parameters | dict | None, key: str, default: str) -> str:
    """Read ``eqa_vl/<key>`` from parameters (dynav_config) with fallback."""
    v = _pget(parameters, f"eqa_vl/{key}", default)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _eqa_cfg(parameters: Parameters | dict | None) -> dict[str, Any]:
    raw = _pget(parameters, "eqa", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _hf_id_matches_family(hf_model_id: str, family: str) -> bool:
    from emet.llms.vllm_registry import normalize_vl_family

    mid = (hf_model_id or "").lower().replace("_", ".")
    fam = normalize_vl_family(family or "")
    if fam == "gemma4":
        return "gemma" in mid
    if fam == "qwen3_vl":
        return "qwen3" in mid
    if fam == "qwen2_5_vl":
        return "qwen2.5" in mid or "qwen2_5" in (hf_model_id or "").lower()
    return True


def resolve_vl_hf_model_id(
    vl_family: str,
    parameters: Parameters | dict | None,
    *,
    device: str = "cuda",
    explicit_hf_id: str | None = None,
) -> str:
    """Pick the largest Gemma checkpoint that fits free VRAM (int4); else registry default."""
    if explicit_hf_id and str(explicit_hf_id).strip():
        return str(explicit_hf_id).strip()

    from emet.llms.vllm_registry import default_hf_model_id, normalize_vl_family

    fam = normalize_vl_family(vl_family or "")
    eqa = _eqa_cfg(parameters)
    cfg_fam = normalize_vl_family(str(eqa.get("vl_family", "") or ""))
    cfg_id = eqa.get("vl_hf_model_id")
    if (
        cfg_id
        and str(cfg_id).strip()
        and (not cfg_fam or cfg_fam == fam)
        and _hf_id_matches_family(str(cfg_id), fam)
    ):
        return str(cfg_id).strip()
    if fam != "gemma4":
        return default_hf_model_id(fam) or ""

    tier_e2b = float(_pget(parameters, "eqa/vram_mib_tier_gemma_e2b", 7000))
    tier_e4b = float(_pget(parameters, "eqa/vram_mib_tier_gemma_e4b", 20000))
    allow_e4b = os.environ.get("EMET_EQA_GEMMA_E4B", "").strip().lower() in ("1", "true", "yes", "on")

    def pick(free_mib: float | None) -> str:
        if free_mib is None:
            logger.warning("nvidia-smi unavailable; defaulting Gemma VLM to google/gemma-3-4b-it")
            return "google/gemma-3-4b-it"
        if allow_e4b and free_mib >= tier_e4b:
            logger.info(
                f"GPU free ~{free_mib:.0f} MiB (Gemma E4B opt-in tier >={tier_e4b:.0f} MiB): google/gemma-4-E4B-it",
            )
            return "google/gemma-4-E4B-it"
        if free_mib >= tier_e2b:
            logger.info(
                f"GPU free ~{free_mib:.0f} MiB (Gemma E2B tier >={tier_e2b:.0f} MiB): google/gemma-4-e2b-it",
            )
            return "google/gemma-4-e2b-it"
        logger.info(f"GPU free ~{free_mib:.0f} MiB: google/gemma-3-4b-it")
        return "google/gemma-3-4b-it"

    if device == "cuda":
        return pick(get_nvidia_gpu_free_mib())
    return pick(None)
