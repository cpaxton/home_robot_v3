# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared Dynagraph benchmark profiles (merge, staleness, short-episode caps)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from emet.core.parameters import Parameters
from emet.eval.memory_backends import (
    DYNAGRAPH,
    OVMM_MEMORY_BACKEND,
    SQA3D_MEMORY_BACKEND,
    STATIC_GRAPH,
    normalize_benchmark_backend,
    normalize_dynagraph_profile,
    normalize_hmeqa_method,
)
from emet.utils.config import resolve_config_yaml_path

ExploreWhenUncoveredMode = Literal["off", "on", "conservative"]
DynagraphProfileName = Literal[
    "interactive",
    "smoke",
    "eqa",
    "unified_eqa",
    "find_phase",
    "static_graph",
]
DYNAMIC_EXPLORE_BACKEND = Literal["dynagraph", "static_graph"]
DYNAMIC_EXPLORE_BACKENDS: tuple[str, ...] = (DYNAGRAPH, STATIC_GRAPH)
SQA3DRunProfile = Literal["smoke", "tuned"]
BenchmarkHarnessName = Literal[
    "habitat_eqa",
    "habitat_ovmm_find",
    "ovmm_find_phase",
    "sqa3d",
    "dynamic_explore",
]

DEFAULT_DYNAGRAPH_BENCHMARK_YAML = "configs/benchmarks/dynagraph.yaml"


@lru_cache(maxsize=1)
def load_dynagraph_benchmark_yaml(path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML) -> dict[str, Any]:
    full = Path(resolve_config_yaml_path(path))
    with full.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dynagraph benchmark config must be a mapping: {full}")
    return data


def list_dynagraph_profiles(path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML) -> tuple[str, ...]:
    profiles = load_dynagraph_benchmark_yaml(path).get("profiles", {})
    if not isinstance(profiles, dict):
        return ()
    return tuple(profiles.keys())


def profile_settings(
    profile: DynagraphProfileName | str,
    *,
    path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML,
) -> dict[str, Any]:
    profiles = load_dynagraph_benchmark_yaml(path).get("profiles", {})
    if not isinstance(profiles, dict):
        raise KeyError(f"No profiles in {path}")
    name = normalize_dynagraph_profile(profile)
    block = profiles.get(name)
    if not isinstance(block, dict):
        raise KeyError(f"Unknown Dynagraph profile {profile!r} in {path}")
    return dict(block)


def _as_parameters(parameters: Parameters | dict[str, Any]) -> Parameters:
    if isinstance(parameters, Parameters):
        return parameters
    return Parameters(**parameters)


def apply_eval_graph_fusion_parameters(
    parameters: Parameters | dict[str, Any],
    *,
    merge_xy_m: float | None = None,
) -> Parameters:
    """Enable GraphObjectFusion for eval harnesses (SQA3D / Habitat) with sane fallback merge."""
    from dataclasses import asdict

    from emet.memory.graph_eqa.graph_object_fusion.config import load_graph_object_fusion_config

    params = _as_parameters(parameters)
    merge_xy = merge_xy_m
    if merge_xy is None:
        raw = params.get("dynagraph_merge_xy_m")
        merge_xy = float(raw) if raw is not None else 0.0
    fc = load_graph_object_fusion_config()
    fusion = asdict(fc)
    if float(merge_xy) > 0.0:
        # Dynagraph production/EQA profiles keep merge + fallback aligned with
        # ``dynagraph_merge_xy_m`` (typically 0.45) for long-horizon instance memory.
        fusion["enabled"] = True
        fusion["fallback_spatial_merge_xy_m"] = float(merge_xy)
    else:
        # Zero-merge profiles (``smoke``, ``static_graph``): disable fusion entirely so
        # IoU/embedding gates cannot silently merge instances (not only XY fallback).
        fusion["enabled"] = False
        fusion["fallback_spatial_merge_xy_m"] = 0.0
    params.set("graph_object_fusion", fusion)
    return params


def apply_dynagraph_profile(
    parameters: Parameters | dict[str, Any],
    profile: DynagraphProfileName | str,
    *,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
    path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML,
) -> Parameters:
    """Apply a named Dynagraph profile to a parameter dict (in-place on ``Parameters.data``)."""
    params = _as_parameters(parameters)
    profile = normalize_dynagraph_profile(profile)
    settings = profile_settings(profile, path=path)
    if settings.get("dynagraph_merge_xy_m") is not None:
        params["dynagraph_merge_xy_m"] = float(settings["dynagraph_merge_xy_m"])
    if settings.get("dynagraph_staleness_horizon") is not None:
        params["dynagraph_staleness_horizon"] = int(settings["dynagraph_staleness_horizon"])
    graph_extract = settings.get("graph_eqa_extract")
    if isinstance(graph_extract, dict):
        merged = dict(params.get("graph_eqa_extract", {}) or {})
        merged.update(graph_extract)
        params.set("graph_eqa_extract", merged)
    if merge_xy_m is not None:
        params["dynagraph_merge_xy_m"] = float(merge_xy_m)
    if staleness_horizon is not None:
        params["dynagraph_staleness_horizon"] = int(staleness_horizon)
    effective_merge = merge_xy_m
    if effective_merge is None:
        effective_merge = params.get("dynagraph_merge_xy_m")
    apply_eval_graph_fusion_parameters(
        params, merge_xy_m=float(effective_merge) if effective_merge is not None else None
    )
    return params


def resolve_ovmm_dynagraph_profile(backend: OVMM_MEMORY_BACKEND | str) -> DynagraphProfileName | None:
    name = normalize_benchmark_backend(backend)
    if name == STATIC_GRAPH:
        return STATIC_GRAPH
    if name in (DYNAGRAPH, "ground_truth"):
        return "find_phase"
    return None


def resolve_sqa3d_dynagraph_profile(
    method: SQA3D_MEMORY_BACKEND | str,
    *,
    profile: SQA3DRunProfile,
) -> DynagraphProfileName | None:
    if method != DYNAGRAPH:
        return None
    return "smoke" if profile == "smoke" else "eqa"


def resolve_dynamic_explore_profile(backend: DYNAMIC_EXPLORE_BACKEND | str) -> DynagraphProfileName:
    name = normalize_benchmark_backend(backend)
    if name == STATIC_GRAPH:
        return STATIC_GRAPH
    if name == DYNAGRAPH:
        return "interactive"
    raise KeyError(f"Unknown dynamic exploration backend {backend!r}")


def apply_dynamic_explore_backend(
    parameters: Parameters | dict[str, Any],
    backend: DYNAMIC_EXPLORE_BACKEND | str,
    *,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
    path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML,
) -> Parameters:
    """Configure merge/staleness and harness flags for dynamic exploration backend rows."""
    params = _as_parameters(parameters)
    backend = normalize_benchmark_backend(backend)
    profile = resolve_dynamic_explore_profile(backend)
    apply_dynagraph_harness(
        params,
        "dynamic_explore",
        str(backend),
        profile_override=profile,
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
        path=path,
    )
    return params


def apply_ovmm_backend_dynagraph(
    parameters: Parameters | dict[str, Any],
    backend: OVMM_MEMORY_BACKEND | str,
    *,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
) -> Parameters:
    """Configure dynagraph merge/staleness and harness flags for OVMM find-phase rows."""
    params = _as_parameters(parameters)
    backend = normalize_benchmark_backend(backend)
    if backend == "dynamem":
        return params
    profile = resolve_ovmm_dynagraph_profile(backend)
    apply_dynagraph_harness(
        params,
        "ovmm_find_phase",
        str(backend),
        profile_override=profile,
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
    )
    return params


def apply_sqa3d_dynagraph(
    parameters: Parameters | dict[str, Any],
    *,
    method: SQA3D_MEMORY_BACKEND | str,
    profile: SQA3DRunProfile,
) -> Parameters:
    """Configure dynagraph merge/staleness and harness flags for SQA3D smoke vs tuned runs."""
    params = _as_parameters(parameters)
    dynagraph_profile = resolve_sqa3d_dynagraph_profile(method, profile=profile)
    if dynagraph_profile is not None:
        apply_dynagraph_profile(params, dynagraph_profile)
    if method == "dynagraph":
        apply_dynagraph_harness(params, "sqa3d", "dynagraph", apply_profile=False)
    return params


def _normalize_explore_mode(value: object) -> ExploreWhenUncoveredMode:
    raw = str(value or "off").strip().lower()
    if raw in ("off", "false", "0", "no"):
        return "off"
    if raw in ("conservative", "safe"):
        return "conservative"
    return "on"


def resolve_harness_profile(
    harness: BenchmarkHarnessName | str,
    *,
    path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML,
) -> str | None:
    data = load_dynagraph_benchmark_yaml(path)
    block = data.get("harness", {})
    if not isinstance(block, dict):
        return None
    harness_block = block.get(harness, {})
    if not isinstance(harness_block, dict):
        return None
    profile = harness_block.get("profile")
    return str(profile) if profile is not None else None


def _resolve_harness_profile_name(
    harness_block: dict[str, Any],
    method: str,
    *,
    profile_override: str | None = None,
) -> str | None:
    if profile_override is not None:
        return str(profile_override)
    method_block = harness_block.get(method, {})
    if isinstance(method_block, dict) and method_block.get("profile") is not None:
        return str(method_block["profile"])
    if harness_block.get("profile") is not None:
        return str(harness_block["profile"])
    return None


def harness_controller_kwargs(
    parameters: Parameters | dict[str, Any],
    *,
    harness: BenchmarkHarnessName | str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """Controller constructor kwargs stored under ``dynagraph_harness``."""
    params = _as_parameters(parameters)
    block = dict(params.get("dynagraph_harness") or {})
    if harness is not None and method is not None:
        block = {**harness_controller_options(harness, method), **block}
    keys = (
        "use_instance_graph",
        "manipulation_only",
        "eqa",
        "use_sensor_perception",
    )
    return {k: block[k] for k in keys if k in block}


def apply_dynagraph_harness(
    parameters: Parameters | dict[str, Any],
    harness: BenchmarkHarnessName | str,
    method: str,
    *,
    profile_override: str | None = None,
    apply_profile: bool = True,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
    path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML,
) -> Parameters:
    """Apply harness profile (merge/staleness) and per-method controller flags."""
    params = _as_parameters(parameters)
    method = normalize_benchmark_backend(method)
    if profile_override is not None:
        profile_override = normalize_dynagraph_profile(profile_override)
    data = load_dynagraph_benchmark_yaml(path)
    harness_root = data.get("harness", {})
    if not isinstance(harness_root, dict):
        harness_root = {}
    harness_block = harness_root.get(harness, {})
    if not isinstance(harness_block, dict):
        harness_block = {}

    profile_name = _resolve_harness_profile_name(harness_block, method, profile_override=profile_override)
    if profile_name is not None:
        profile_name = normalize_dynagraph_profile(profile_name)
    if apply_profile and profile_name is not None:
        apply_dynagraph_profile(
            params,
            str(profile_name),
            merge_xy_m=merge_xy_m,
            staleness_horizon=staleness_horizon,
            path=path,
        )
    elif method == DYNAGRAPH:
        apply_eval_graph_fusion_parameters(params, merge_xy_m=merge_xy_m)

    method_opts = {k: v for k, v in harness_controller_options(harness, method, path=path).items() if k != "profile"}
    # YAML 1.1 parses bare on/off as bools; store the canonical string mode.
    if "explore_when_uncovered" in method_opts:
        method_opts["explore_when_uncovered"] = _normalize_explore_mode(method_opts["explore_when_uncovered"])
    merged = dict(params.get("dynagraph_harness") or {})
    merged.update(method_opts)
    merged["harness"] = str(harness)
    merged["method"] = str(method)
    if profile_name is not None:
        merged["profile"] = str(profile_name)
    params.set("dynagraph_harness", merged)

    if method_opts.get("prompt_variant"):
        eqa = dict(params.get("eqa", {}) or {})
        eqa["prompt_variant"] = str(method_opts["prompt_variant"])
        params.set("eqa", eqa)
    if "sqa3d_allow_partial_graph" in method_opts:
        eqa = dict(params.get("eqa", {}) or {})
        eqa["sqa3d_allow_partial_graph"] = bool(method_opts["sqa3d_allow_partial_graph"])
        params.set("eqa", eqa)
    if "merged_memory" in method_opts:
        # Paper rows pin the standalone CONFIRMED_MEMORY block (default is now folded).
        eqa = dict(params.get("eqa", {}) or {})
        eqa["merged_memory"] = bool(method_opts["merged_memory"])
        params.set("eqa", eqa)
    return params


def apply_habitat_eqa_method_parameters(
    parameters: Parameters | dict[str, Any],
    method: str,
) -> Parameters:
    """HM-EQA harness: ``static_graph`` → zero merge/staleness (GraphEQA-inspired);
    ``dynagraph`` → ``unified_eqa`` (0.45 m merge) + tuned EQA extras.

    Legacy ``graph_eqa`` / profile ``graph_eqa_baseline`` alias to ``static_graph``.
    """
    method = normalize_hmeqa_method(method)
    params = _as_parameters(parameters)
    apply_dynagraph_harness(params, "habitat_eqa", method)
    # Habitat HM-EQA is always MCQ A–D. Without this, default runs load the open
    # GraphEQA system prompt (Caption: demos) and skip Reasoning: prefill.
    eqa = dict(params.get("eqa", {}) or {})
    eqa.setdefault("prompt_variant", "hmeqa")
    params.set("eqa", eqa)
    return params


def apply_habitat_ovmm_find_parameters(
    parameters: Parameters | dict[str, Any],
    backend: str,
    *,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
) -> Parameters:
    params = _as_parameters(parameters)
    backend = normalize_benchmark_backend(backend)
    apply_dynagraph_harness(
        params,
        "habitat_ovmm_find",
        str(backend),
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
    )
    return params


def dynagraph_harness_flags(parameters: Parameters | dict[str, Any] | None) -> dict[str, Any]:
    """Resolved Dynagraph EQA/controller flags for ``DynagraphController`` init."""
    params = _as_parameters(parameters) if parameters is not None else Parameters()
    block = dict(params.get("dynagraph_harness") or {})
    eqa_cfg = dict(params.get("eqa", {}) or {})
    variant = str(eqa_cfg.get("prompt_variant", "") or "").strip().lower()
    sqa3d_open_qa = variant in ("sqa3d", "situated")

    defaults: dict[str, Any] = {
        "memory_summary": not sqa3d_open_qa,
        "mcq_debias": not sqa3d_open_qa,
        "explore_when_uncovered": "on" if not sqa3d_open_qa else "off",
        "siglip_grounding": not sqa3d_open_qa,
    }
    if sqa3d_open_qa:
        defaults.update(
            memory_summary=False,
            mcq_debias=False,
            explore_when_uncovered="off",
            siglip_grounding=False,
        )

    for key in ("memory_summary", "mcq_debias", "siglip_grounding"):
        if key in block:
            defaults[key] = bool(block[key])
    if "explore_when_uncovered" in block:
        defaults["explore_when_uncovered"] = _normalize_explore_mode(block["explore_when_uncovered"])
    defaults["explore_when_uncovered"] = _normalize_explore_mode(defaults["explore_when_uncovered"])
    return defaults


def apply_dynagraph_harness_overrides(
    parameters: Parameters | dict[str, Any],
    *,
    memory_summary: bool | None = None,
    mcq_debias: bool | None = None,
    explore_when_uncovered: ExploreWhenUncoveredMode | str | None = None,
) -> Parameters:
    """CLI/env overrides layered on top of harness defaults."""
    params = _as_parameters(parameters)
    block = dict(params.get("dynagraph_harness") or {})
    if memory_summary is not None:
        block["memory_summary"] = bool(memory_summary)
    if mcq_debias is not None:
        block["mcq_debias"] = bool(mcq_debias)
    if explore_when_uncovered is not None:
        block["explore_when_uncovered"] = _normalize_explore_mode(explore_when_uncovered)
    params.set("dynagraph_harness", block)
    return params


def harness_controller_options(
    harness: BenchmarkHarnessName,
    method: str,
    *,
    path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML,
) -> dict[str, Any]:
    """Return documented controller kwargs for a benchmark harness (task-specific)."""
    method = normalize_benchmark_backend(method)
    data = load_dynagraph_benchmark_yaml(path)
    block = data.get("harness", {})
    if not isinstance(block, dict):
        return {}
    harness_block = block.get(harness, {})
    if not isinstance(harness_block, dict):
        return {}
    method_block = harness_block.get(method, {})
    if not isinstance(method_block, dict):
        return {}
    return dict(method_block)
