# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Device-placement checks for local VL / LLM loads (catch silent CPU fallback)."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

_TRUE = frozenset({"1", "true", "yes", "on"})


def env_allow_cpu_vlm() -> bool:
    """When set, permit loading / running the agent VLM on CPU (very slow)."""
    return os.environ.get("EMET_ALLOW_CPU_VLM", "").strip().lower() in _TRUE


def parameter_device_counts(model: Any, *, max_params: int | None = 64) -> Counter[str]:
    """Count parameter/buffer devices.

    When ``max_params`` is set, only the first ``max_params`` tensors are sampled
    (for short diagnostic summaries). Pass ``max_params=None`` to inspect all.
    """
    counts: Counter[str] = Counter()
    n = 0
    for p in model.parameters():
        counts[str(p.device)] += 1
        n += 1
        if max_params is not None and n >= max_params:
            break
    # Also peek at buffers (bnb scales sometimes live as buffers)
    if (max_params is None or n < max_params) and hasattr(model, "buffers"):
        for b in model.buffers():
            counts[str(b.device)] += 1
            n += 1
            if max_params is not None and n >= max_params:
                break
    return counts


def summarize_model_devices(model: Any) -> str:
    """Human-readable device summary, e.g. ``cuda:0=64`` or ``cpu=12,cuda:0=4``."""
    counts = parameter_device_counts(model)
    if not counts:
        return "(no parameters)"
    return ",".join(f"{dev}={n}" for dev, n in sorted(counts.items()))


def primary_param_device(model: Any) -> str:
    """Device string of the first parameter, or ``unknown``."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "unknown"


def assert_cuda_placement(
    model: Any,
    *,
    requested_device: str,
    model_label: str = "VLM",
    allow_cpu: bool | None = None,
) -> str:
    """Ensure weights live on CUDA when ``requested_device`` is cuda.

    Returns the primary device string. Raises ``RuntimeError`` if any
    weights are on CPU/meta/disk while CUDA was requested, unless
    ``EMET_ALLOW_CPU_VLM=1`` (or ``allow_cpu=True``).

    When ``hf_device_map`` is present and all entries are CUDA, skip walking
    every parameter/buffer (bitsandbytes int4 walks can stall for a long time
    and look like a hang after the ``weights+int4`` print). Otherwise sample a
    large bounded set of tensors.
    """
    req = (requested_device or "").strip().lower()
    primary = primary_param_device(model)
    if not req.startswith("cuda"):
        return primary

    allow = env_allow_cpu_vlm() if allow_cpu is None else bool(allow_cpu)
    # Prefer hf_device_map when present (covers offloaded / multi-device layouts).
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        bad_map = {
            k: v
            for k, v in device_map.items()
            if str(v).startswith("cpu") or str(v) in ("meta", "disk") or "disk" in str(v)
        }
        if bad_map and not allow:
            raise RuntimeError(
                f"{model_label} was requested on {requested_device!r} but hf_device_map has non-GPU "
                f"placements ({bad_map}). Free VRAM or set EMET_ALLOW_CPU_VLM=1 for the slow CPU path."
            )
        if not bad_map:
            # Explicit CUDA-only map (e.g. {"": 0} from bitsandbytes) — trust it.
            if not primary.startswith("cuda") and not allow:
                raise RuntimeError(
                    f"{model_label} primary device is {primary!r} but device={requested_device!r} "
                    "was requested. Set EMET_ALLOW_CPU_VLM=1 only if you intentionally want CPU inference."
                )
            return primary

    # No clean device_map: sample tensors (full walk on int4 can stall for hours).
    sample_n = int(os.environ.get("EMET_VLM_DEVICE_CHECK_MAX_PARAMS", "512") or "512")
    counts = parameter_device_counts(model, max_params=max(1, sample_n))
    bad = {d: n for d, n in counts.items() if d.startswith("cpu") or d in ("meta",) or "disk" in d}
    if bad and not allow:
        detail = ",".join(f"{dev}={n}" for dev, n in sorted(counts.items()))
        raise RuntimeError(
            f"{model_label} was requested on {requested_device!r} but weights are not fully on GPU "
            f"(devices: {detail}). This causes multi-minute CPU inference and looks like a hang. "
            "Free VRAM (close other GPU jobs), use a smaller --llm, or set EMET_ALLOW_CPU_VLM=1 to "
            "explicitly allow the slow CPU path."
        )
    if not primary.startswith("cuda") and not allow:
        raise RuntimeError(
            f"{model_label} primary device is {primary!r} but device={requested_device!r} was requested. "
            "Set EMET_ALLOW_CPU_VLM=1 only if you intentionally want CPU inference."
        )
    return primary
