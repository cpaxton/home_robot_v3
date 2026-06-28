# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SigLIP phrase alignment for GraphEQA CONFIRMED_MEMORY (lightweight vs voxel map)."""

from __future__ import annotations

from typing import Any

import numpy as np


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


def align_phrase_to_observation_features(
    phrase: str,
    encoder: Any,
    observations: list[Any],
    obs_features: dict[int, np.ndarray],
) -> tuple[float, np.ndarray, int] | None:
    """Best SigLIP text-image match for *phrase* over cached observation embeddings."""
    text = (phrase or "").strip()
    if not text or not observations or encoder is None:
        return None
    try:
        text_feat = _feature_vector(encoder.encode_text(text))
    except Exception:
        return None
    id_to_obs = {int(o.obs_id): o for o in observations}
    best: tuple[float, np.ndarray, int] | None = None
    for oid, img_feat in obs_features.items():
        sim = float(np.dot(text_feat, img_feat))
        if best is None or sim > best[0]:
            obs = id_to_obs.get(int(oid))
            xyz = np.asarray(obs.xyz, dtype=float) if obs is not None else np.zeros(3)
            best = (sim, xyz, int(oid))
    return best


def warm_graph_eqa_siglip_confirmed_memory(agent: Any) -> None:
    """Snapshot graph-obs SigLIP features + phrase alignments before voxel encoder release."""
    gm = getattr(agent, "graph_memory", None)
    if gm is None or not getattr(gm, "memory_summary_enabled", False):
        return
    enc = resolve_siglip_encoder(agent)
    if enc is None:
        return
    gm.set_confirmed_memory_siglip_encoder(enc)
    gm.refresh_siglip_confirmed_memory()
