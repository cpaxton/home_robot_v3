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
