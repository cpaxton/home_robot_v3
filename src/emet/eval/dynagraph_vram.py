# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Release non-EQA GPU caches before GraphEQA VLM forward (eval harnesses)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def release_gpu_memory() -> None:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _vram_free_mib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(0)
        return float(free) / (1024.0 * 1024.0)
    except Exception:
        return None


def re_attach_siglip_encoder(agent: Any) -> Any | None:
    """Re-attach the shared SigLIP encoder after :func:`release_siglip_for_vlm`.

    ``release_siglip_for_vlm`` drops ``agent.encoder`` / ``voxel_map.encoder`` so
    Qwen can load; the agentic find loop runs *per phase* (FindObj then FindRec),
    and each phase needs ``localize_text`` on the finished map. Without re-attaching,
    the second phase's localize silently returns nothing (SigLIP cosine over an
    ``encoder=None`` voxel) and the loop can only explore. ``warm_*`` snapshots
    cached ranks but does not restore the live encoder, so call this from the warm
    path.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

    enc = get_shared_mask_siglip_encoder(version="so400m", device="cuda", feature_matching_threshold=0.14)
    if enc is None:
        return None
    agent.encoder = enc
    vm = getattr(agent, "voxel_map", None)
    if vm is not None:
        vm.encoder = enc
    gm = getattr(agent, "graph_memory", None)
    if gm is not None and hasattr(gm, "set_confirmed_memory_siglip_encoder"):
        try:
            gm.set_confirmed_memory_siglip_encoder(enc)
        except Exception:
            pass
    return enc


def warm_siglip_confirmed_memory(agent: Any) -> None:
    """Snapshot SigLIP CONFIRMED_MEMORY features and visual FIND ranks.

    Keep the encoder attached for agentic verify; answer-only / Habitat then
    calls :func:`release_siglip_for_vlm` so Qwen can load. FIND after that
    release uses the ranks cached here. Do **not** re-attach the encoder here —
    HM-EQA relies on the released state (SigLIP dropped before Qwen); the OVMM
    find harness calls :func:`re_attach_siglip_encoder` explicitly so its
    second find phase can still ``localize_text`` on the finished map.
    """
    from emet.memory.graph_eqa.graph_eqa_siglip import warm_graph_eqa_siglip_confirmed_memory

    warm_graph_eqa_siglip_confirmed_memory(agent)
    logger.info("warm_siglip_confirmed_memory: CONFIRMED_MEMORY features + FIND ranks warmed")


def release_siglip_for_vlm(agent: Any) -> None:
    """Drop SigLIP / voxel encoders immediately before the EQA VLM forward."""
    from emet.perception.encoders.siglip_encoder import release_shared_mask_siglip_encoder

    free0 = _vram_free_mib()
    if hasattr(agent, "encoder"):
        agent.encoder = None
    vm = getattr(agent, "voxel_map", None)
    if vm is not None:
        vm.encoder = None
    gm = getattr(agent, "graph_memory", None)
    if gm is not None:
        gm.set_confirmed_memory_siglip_encoder(None)
    release_shared_mask_siglip_encoder()
    release_gpu_memory()
    try:
        from emet.llms.graph_eqa_vlm import trim_shared_graph_eqa_vlm_cache

        trim_shared_graph_eqa_vlm_cache()
    except Exception:
        pass
    free1 = _vram_free_mib()
    if free0 is not None and free1 is not None:
        logger.info(
            "release_siglip_for_vlm: free VRAM %.0f → %.0f MiB",
            free0,
            free1,
        )
    else:
        logger.info("release_siglip_for_vlm: SigLIP released before VLM")


def prepare_dynagraph_vram_for_eqa(agent: Any) -> None:
    """Free GPU headroom before the EQA VLM forward.

    Snapshot SigLIP CONFIRMED_MEMORY features and visual FIND top-k ranks into
    graph-memory caches, then **always** drop SigLIP + voxel encoders. Keeping
    SigLIP loaded next to Qwen3-VL-8B int4 was starving activations: overnight
    smokes loaded the VLM in ~125s then hung after ``ready for inference``
    until STALE_KILL. Voxel ``find_all_images`` cannot run after this release;
    ``query_answer`` uses the cached ranks.

    For agentic verify loops, call :func:`warm_siglip_confirmed_memory` before
    navigate/verify and :func:`release_siglip_for_vlm` only before submit_answer.
    """
    warm_siglip_confirmed_memory(agent)
    release_siglip_for_vlm(agent)
