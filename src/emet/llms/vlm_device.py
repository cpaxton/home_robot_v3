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


def parameter_device_counts(model: Any, *, max_params: int = 64) -> Counter[str]:
    """Sample parameter devices (up to ``max_params``) → counts by device string."""
    counts: Counter[str] = Counter()
    n = 0
    for p in model.parameters():
        counts[str(p.device)] += 1
        n += 1
        if n >= max_params:
            break
    # Also peek at buffers (bnb scales sometimes live as buffers)
    if n < max_params and hasattr(model, "buffers"):
        for b in model.buffers():
            counts[str(b.device)] += 1
            n += 1
            if n >= max_params:
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

    Returns the primary device string. Raises ``RuntimeError`` if any sampled
    weights are on CPU/meta/disk while CUDA was requested, unless
    ``EMET_ALLOW_CPU_VLM=1`` (or ``allow_cpu=True``).
    """
    req = (requested_device or "").strip().lower()
    primary = primary_param_device(model)
    if not req.startswith("cuda"):
        return primary

    allow = env_allow_cpu_vlm() if allow_cpu is None else bool(allow_cpu)
    counts = parameter_device_counts(model)
    bad = {d: n for d, n in counts.items() if d.startswith("cpu") or d in ("meta",) or "disk" in d}
    if bad and not allow:
        detail = summarize_model_devices(model)
        raise RuntimeError(
            f"{model_label} was requested on {requested_device!r} but weights are not fully on GPU "
            f"(sampled devices: {detail}). This causes multi-minute CPU inference and looks like a hang. "
            "Free VRAM (close other GPU jobs), use a smaller --llm, or set EMET_ALLOW_CPU_VLM=1 to "
            "explicitly allow the slow CPU path."
        )
    if not primary.startswith("cuda") and not allow:
        raise RuntimeError(
            f"{model_label} primary device is {primary!r} but device={requested_device!r} was requested. "
            "Set EMET_ALLOW_CPU_VLM=1 only if you intentionally want CPU inference."
        )
    return primary
