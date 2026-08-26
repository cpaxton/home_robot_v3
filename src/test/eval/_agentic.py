# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import importlib

import pytest


def _has_attr_path(mod_name: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return False
    return hasattr(mod, attr)


def _require_agentic():
    if not _has_attr_path("emet.memory.graph_eqa.agentic_eqa", "run_agentic_eqa"):
        pytest.skip("agentic EQA module not present")
    from emet.memory.graph_eqa import GraphEQAMemory

    if not hasattr(GraphEQAMemory, "hypothesize_nav_targets"):
        pytest.skip("GraphEQAMemory.hypothesize_nav_targets not implemented")


def _require_vram_split():
    from emet.eval import dynagraph_vram as dv

    if not hasattr(dv, "warm_siglip_confirmed_memory") or not hasattr(dv, "release_siglip_for_vlm"):
        pytest.skip("VRAM warm/release split not implemented")


def _require_vision_cache():
    if not _has_attr_path("emet.llms.vl_vision_cache", "VisionPrefixCache"):
        pytest.skip("vl_vision_cache not implemented")
