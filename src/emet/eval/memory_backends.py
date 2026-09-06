# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared memory-backend names across benchmark harnesses."""

from __future__ import annotations

from typing import Literal

from emet.utils.logger import Logger

logger = Logger(__name__)

# Canonical backend / HM-EQA method ids.
STATIC_GRAPH = "static_graph"  # object graph, merge/staleness off (was graph_eqa)
DYNAGRAPH = "dynagraph"
LAZY_GRAPH = "lazy_graph"
DYNAMEM = "dynamem"
GROUND_TRUTH = "ground_truth"

# Legacy method/backend id → canonical (warn once per process).
_LEGACY_BACKEND_ALIASES: dict[str, str] = {
    "graph_eqa": STATIC_GRAPH,
}
# Legacy Dynagraph profile name → canonical.
_LEGACY_PROFILE_ALIASES: dict[str, str] = {
    "graph_eqa_baseline": STATIC_GRAPH,
}

_warned_aliases: set[str] = set()

# OVMM find-phase (Emet sim + Habitat proxy): includes oracle + static-graph row.
OVMM_MEMORY_BACKEND = Literal["dynamem", "static_graph", "dynagraph", "ground_truth"]
OVMM_MEMORY_BACKENDS: tuple[str, ...] = (DYNAMEM, STATIC_GRAPH, DYNAGRAPH, GROUND_TRUTH)
# Click Choice: canonical first, then legacy alias for scripts.
OVMM_MEMORY_BACKEND_CHOICES: tuple[str, ...] = (
    DYNAMEM,
    STATIC_GRAPH,
    "graph_eqa",
    DYNAGRAPH,
    GROUND_TRUTH,
)

# SQA3D embodied QA: voxel-only vs Dynagraph (voxel + graph).
SQA3D_MEMORY_BACKEND = Literal["dynamem", "dynagraph"]
SQA3D_MEMORY_BACKENDS: tuple[str, ...] = (DYNAMEM, DYNAGRAPH)

# HM-EQA Habitat methods.
HMEQA_METHODS: tuple[str, ...] = (STATIC_GRAPH, DYNAGRAPH, LAZY_GRAPH)
HMEQA_METHOD = Literal["static_graph", "dynagraph", "lazy_graph"]
HMEQA_METHOD_CHOICES: tuple[str, ...] = (STATIC_GRAPH, "graph_eqa", DYNAGRAPH, LAZY_GRAPH)

# Agent / interactive object-graph plug-ins (plus open_vocab / dynamem voxels).
AGENT_MEMORY_BACKENDS: tuple[str, ...] = (DYNAGRAPH, LAZY_GRAPH, STATIC_GRAPH, DYNAMEM, "open_vocab")
AGENT_MEMORY_BACKEND_CHOICES: tuple[str, ...] = (
    DYNAGRAPH,
    LAZY_GRAPH,
    STATIC_GRAPH,
    "graph_eqa",
    DYNAMEM,
    "open_vocab",
)


def _warn_alias(legacy: str, canonical: str, *, kind: str) -> None:
    key = f"{kind}:{legacy}"
    if key in _warned_aliases:
        return
    _warned_aliases.add(key)
    logger.warning(f"Deprecated {kind} {legacy!r} → use {canonical!r} (legacy alias accepted for compatibility)")


def normalize_benchmark_backend(name: str | None, *, warn: bool = True) -> str:
    """Canonicalize OVMM / dynamic-explore / agent-style memory backend ids."""
    raw = str(name or "").strip().lower().replace("-", "_")
    if not raw:
        raise ValueError("memory backend name is empty")
    if raw in _LEGACY_BACKEND_ALIASES:
        canonical = _LEGACY_BACKEND_ALIASES[raw]
        if warn:
            _warn_alias(raw, canonical, kind="backend")
        return canonical
    return raw


def normalize_hmeqa_method(method: str | None, *, warn: bool = True) -> str:
    """Canonicalize HM-EQA ``--method`` (``static_graph`` | ``dynagraph``)."""
    canonical = normalize_benchmark_backend(method, warn=warn)
    if canonical not in HMEQA_METHODS:
        raise ValueError(
            f"Unknown HM-EQA method {method!r}; use one of {HMEQA_METHODS} (legacy alias: graph_eqa → static_graph)"
        )
    return canonical


def normalize_dynagraph_profile(profile: str | None, *, warn: bool = True) -> str:
    """Canonicalize Dynagraph profile names (``graph_eqa_baseline`` → ``static_graph``)."""
    raw = str(profile or "").strip().lower().replace("-", "_")
    if not raw:
        raise ValueError("Dynagraph profile name is empty")
    if raw in _LEGACY_PROFILE_ALIASES:
        canonical = _LEGACY_PROFILE_ALIASES[raw]
        if warn:
            _warn_alias(raw, canonical, kind="profile")
        return canonical
    return raw


def is_static_graph_backend(name: str | None) -> bool:
    """True for ``static_graph`` or legacy ``graph_eqa``."""
    try:
        return normalize_benchmark_backend(name, warn=False) == STATIC_GRAPH
    except ValueError:
        return False
