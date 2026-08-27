# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Write RGB frames for agentic VLM queries so they can be reviewed on disk.

OVMM find and HM-EQA both feed Qwen a graph observation. Those pixels used to
stay in memory only. When a dump directory is known (trace path, episode dir,
or ``EMET_AGENTIC_QUERY_IMAGES_DIR``), each assess/capture writes:

- ``images/rgb_{obs_id:04d}.png`` — canonical frame for that obs_id
- ``images/{kind}_r{round:02d}_obs{obs_id:04d}.png`` — one file per query
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

_FALSE = frozenset({"0", "false", "no", "off"})
_KIND_RE = re.compile(r"[^a-z0-9_]+")


def query_images_enabled() -> bool:
    raw = os.environ.get("EMET_AGENTIC_QUERY_IMAGES", "1").strip().lower()
    return raw not in _FALSE


def as_uint8_rgb(rgb: Any) -> np.ndarray | None:
    """Return HxWx3 uint8, or None if ``rgb`` is not a real image array."""
    if rgb is None:
        return None
    try:
        arr = np.asarray(rgb)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2 or arr.shape[-1] < 3:
        return None
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr[:, :, :3])
    if not np.issubdtype(arr.dtype, np.number):
        return None
    finite = arr[:, :, :3]
    if np.issubdtype(finite.dtype, np.floating):
        peak = float(np.nanmax(finite)) if finite.size else 0.0
        if peak <= 1.0 + 1e-6:
            finite = np.clip(finite, 0.0, 1.0) * 255.0
    return np.clip(finite, 0, 255).astype(np.uint8)


def save_rgb_png(path: Path | str, rgb: Any) -> Path | None:
    """Write ``rgb`` as PNG. Returns the path, or None if skipped."""
    arr = as_uint8_rgb(rgb)
    if arr is None:
        return None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(arr, mode="RGB").save(dest)
    return dest


def query_images_dir(executor: Any) -> Path | None:
    """Directory for query PNGs, or None when dumping is off / unconfigured."""
    if not query_images_enabled():
        return None
    override = os.environ.get("EMET_AGENTIC_QUERY_IMAGES_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    trace_path = getattr(executor, "_trace_path", None)
    if trace_path:
        return Path(trace_path).expanduser().parent / "images"
    agent = getattr(executor, "agent", None)
    ep = getattr(agent, "_episode_debug_dir", None) if agent is not None else None
    if ep:
        return Path(str(ep)).expanduser() / "images"
    env_ep = os.environ.get("EMET_EQA_EPISODE_DIR", "").strip()
    if env_ep:
        return Path(env_ep).expanduser() / "images"
    return None


def rgb_from_graph_obs(executor: Any, obs_id: int) -> Any | None:
    gm = getattr(executor, "graph_memory", None)
    if gm is None:
        return None
    getter = getattr(gm, "_observation_by_id", None)
    if not callable(getter):
        return None
    try:
        obs = getter(int(obs_id))
    except (TypeError, ValueError):
        return None
    return getattr(obs, "rgb", None) if obs is not None else None


def dump_query_rgb(
    executor: Any,
    obs_id: int,
    rgb: Any | None = None,
    *,
    kind: str = "vlm_assess",
) -> dict[str, str]:
    """Save canonical + per-query PNGs. Returns path fields for the trace row."""
    out_dir = query_images_dir(executor)
    if out_dir is None:
        return {}
    pixels = rgb if rgb is not None else rgb_from_graph_obs(executor, obs_id)
    arr = as_uint8_rgb(pixels)
    if arr is None:
        return {}
    oid = int(obs_id)
    try:
        round_i = int(getattr(executor, "_round", 0) or 0)
    except (TypeError, ValueError):
        round_i = 0
    meta = dict(getattr(executor, "_trace_meta", None) or {})
    phase = str(meta.get("ovmm_phase") or "").strip()
    kind_token = _KIND_RE.sub("_", str(kind or "query").lower()).strip("_") or "query"
    if phase:
        kind_token = f"{kind_token}_{_KIND_RE.sub('_', phase.lower()).strip('_')}"
    canonical = out_dir / f"rgb_{oid:04d}.png"
    query = out_dir / f"{kind_token}_r{round_i:02d}_obs{oid:04d}.png"
    try:
        saved_canonical = save_rgb_png(canonical, arr)
        saved_query = save_rgb_png(query, arr)
    except OSError:
        return {}
    if saved_canonical is None or saved_query is None:
        return {}
    return {
        "rgb_png": str(saved_canonical),
        "query_png": str(saved_query),
    }
