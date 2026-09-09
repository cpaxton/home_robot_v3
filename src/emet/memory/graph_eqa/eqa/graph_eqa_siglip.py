# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SigLIP phrase alignment for GraphEQA CONFIRMED_MEMORY (lightweight vs voxel map).

Thresholds (two spaces — do not mix them):

- Voxel / point features (DynaMem ``verify_point``): ``SIGLIP_PRESENT_THRESHOLD`` (0.21)
  and ``SIGLIP_CONFIRM_THRESHOLD`` (0.28).
- Habitat RGB / dense-patch verify: ``SIGLIP_IMAGE_PRESENT_THRESHOLD`` (0.12) and
  ``SIGLIP_IMAGE_ABSENT_THRESHOLD`` (0.10) in ``agentic_config``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Min SigLIP cosine for open-vocab text vs voxel point features.
SIGLIP_PRESENT_THRESHOLD = 0.21
# Stronger bar before SigLIP-only evidence may override the VLM.
SIGLIP_CONFIRM_THRESHOLD = 0.28


def resolve_siglip_encoder(agent: Any) -> Any | None:
    """Return a SigLIP encoder from agent, voxel map, graph memory, or the shared cache."""
    enc = getattr(agent, "encoder", None)
    if enc is not None and hasattr(enc, "encode_image"):
        return enc
    vm = getattr(agent, "voxel_map", None)
    if vm is not None:
        enc = getattr(vm, "encoder", None)
        if enc is not None and hasattr(enc, "encode_image"):
            return enc
    gm = getattr(agent, "graph_memory", None)
    if gm is not None:
        enc = getattr(gm, "_confirmed_memory_siglip_encoder", None)
        if enc is not None and hasattr(enc, "encode_image"):
            return enc
    try:
        from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

        return get_shared_mask_siglip_encoder()
    except Exception:
        return None


def should_keep_siglip_for_confirmed_memory(agent: Any) -> bool:
    """Keep the shared SigLIP encoder loaded when Dynagraph CONFIRMED_MEMORY is enabled."""
    gm = getattr(agent, "graph_memory", None)
    return bool(gm is not None and getattr(gm, "memory_summary_enabled", False))


def _feature_vector(arr: Any) -> np.ndarray:
    out = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def encode_observation_rgb(encoder: Any, rgb: np.ndarray) -> np.ndarray | None:
    try:
        return _feature_vector(encoder.encode_image(np.asarray(rgb, dtype=np.uint8)))
    except Exception:
        return None


def _as_1d_array(arr: Any) -> np.ndarray:
    if arr is None:
        return np.zeros(0, dtype=np.float64)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    out = np.asarray(arr)
    if out.size == 0:
        return np.zeros(0, dtype=np.float64)
    return out.reshape(-1).astype(np.float64)


def flatten_find_all_images(
    image_ids: Any,
    points: Any = None,
    alignments: Any = None,
) -> list[tuple[float, int, np.ndarray | None]]:
    """Rank ``find_all_images`` hits by score (high to low).

    Voxel DynaMem sorts the returned ids by ``obs_count`` ascending after top-k,
    which is useful for an LLM locator but hides the best SigLIP view. Restore
    score order here. ``image_ids`` are voxel frame counts, not graph obs ids.
    """
    ids = _as_1d_array(image_ids)
    als = _as_1d_array(alignments)
    n = int(ids.size)
    if n == 0:
        return []
    if als.size == 0:
        als = np.ones(n, dtype=np.float64)
    n = min(n, int(als.size))
    xyz = None
    if points is not None:
        raw = points.detach().cpu().numpy() if hasattr(points, "detach") else np.asarray(points)
        xyz = np.asarray(raw, dtype=float)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
    rows: list[tuple[float, int, np.ndarray | None]] = []
    for i in range(n):
        pt = None
        if xyz is not None and i < xyz.shape[0]:
            pt = np.asarray(xyz[i], dtype=float).reshape(-1)[:3].copy()
        rows.append((float(als[i]), int(ids[i]), pt))
    rows.sort(key=lambda t: -t[0])
    return rows


def rank_observations_for_phrase(
    phrase: str,
    encoder: Any,
    obs_features: dict[int, np.ndarray],
) -> list[tuple[float, int]]:
    """All observation ids scored by SigLIP text-image cosine, high to low."""
    text = (phrase or "").strip()
    if not text or encoder is None or not obs_features:
        return []
    try:
        text_feat = _feature_vector(encoder.encode_text(text))
    except Exception:
        return []
    ranked: list[tuple[float, int]] = []
    for oid, img_feat in obs_features.items():
        sim = float(np.dot(text_feat, np.asarray(img_feat, dtype=np.float32).reshape(-1)))
        ranked.append((sim, int(oid)))
    ranked.sort(key=lambda t: -t[0])
    return ranked


def align_phrase_to_observation_features(
    phrase: str,
    encoder: Any,
    observations: list[Any],
    obs_features: dict[int, np.ndarray],
) -> tuple[float, np.ndarray, int] | None:
    """Best SigLIP text-image match for *phrase* over cached observation embeddings."""
    ranked = rank_observations_for_phrase(phrase, encoder, obs_features)
    if not ranked:
        return None
    sim, oid = ranked[0]
    id_to_obs = {int(o.obs_id): o for o in observations}
    obs = id_to_obs.get(int(oid))
    xyz = np.asarray(obs.xyz, dtype=float) if obs is not None else np.zeros(3)
    return sim, xyz, int(oid)


def warm_graph_eqa_siglip_confirmed_memory(agent: Any) -> None:
    """Snapshot graph-obs SigLIP features, phrase alignments, and visual FIND ranks.

    Habitat / answer-only EQA then drops GPU SigLIP before the VLM load. FIND at
    ``query_answer`` must use the ranks cached here (voxel ``find_all_images``
    needs ``encode_text``).
    """
    gm = getattr(agent, "graph_memory", None)
    if gm is None:
        return
    enc = resolve_siglip_encoder(agent)
    if enc is not None and getattr(gm, "memory_summary_enabled", False):
        gm.set_confirmed_memory_siglip_encoder(enc)
        gm.refresh_siglip_confirmed_memory()
    question = str(getattr(agent, "_eqa_question", "") or getattr(gm, "_question", "") or "")
    snap = getattr(gm, "snapshot_visual_find_ranks", None)
    if callable(snap):
        snap(question=question)
