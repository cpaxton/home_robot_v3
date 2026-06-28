# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Release non-EQA GPU caches before GraphEQA VLM forward (eval harnesses)."""

from __future__ import annotations

from typing import Any


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


def prepare_dynagraph_vram_for_eqa(agent: Any) -> None:
    """Drop voxel SigLIP caches for VLM headroom; keep phrase SigLIP for CONFIRMED_MEMORY."""
    from emet.memory.graph_eqa.graph_eqa_siglip import (
        should_keep_siglip_for_confirmed_memory,
        warm_graph_eqa_siglip_confirmed_memory,
    )

    warm_graph_eqa_siglip_confirmed_memory(agent)
    if hasattr(agent, "encoder"):
        agent.encoder = None
    vm = getattr(agent, "voxel_map", None)
    if vm is not None:
        vm.encoder = None
    if not should_keep_siglip_for_confirmed_memory(agent):
        from emet.perception.encoders.siglip_encoder import release_shared_mask_siglip_encoder

        release_shared_mask_siglip_encoder()
        gm = getattr(agent, "graph_memory", None)
        if gm is not None:
            gm.set_confirmed_memory_siglip_encoder(None)
    release_gpu_memory()
    try:
        from emet.llms.graph_eqa_vlm import trim_shared_graph_eqa_vlm_cache

        trim_shared_graph_eqa_vlm_cache()
    except Exception:
        pass
