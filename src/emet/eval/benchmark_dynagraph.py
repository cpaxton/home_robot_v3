# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared Dynagraph benchmark profiles (merge, staleness, short-episode caps)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from emet.core.parameters import Parameters
from emet.eval.memory_backends import OVMM_MEMORY_BACKEND, SQA3D_MEMORY_BACKEND
from emet.utils.config import resolve_config_yaml_path

DynagraphProfileName = Literal[
    "interactive",
    "smoke",
    "eqa",
    "find_phase",
    "graph_eqa_baseline",
]
SQA3DRunProfile = Literal["smoke", "tuned"]
BenchmarkHarnessName = Literal["ovmm_find_phase", "sqa3d"]

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
    block = profiles.get(profile)
    if not isinstance(block, dict):
        raise KeyError(f"Unknown Dynagraph profile {profile!r} in {path}")
    return dict(block)


def _as_parameters(parameters: Parameters | dict[str, Any]) -> Parameters:
    if isinstance(parameters, Parameters):
        return parameters
    return Parameters(**parameters)


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
    return params


def resolve_ovmm_dynagraph_profile(backend: OVMM_MEMORY_BACKEND | str) -> DynagraphProfileName | None:
    if backend == "graph_eqa":
        return "graph_eqa_baseline"
    if backend in ("dynagraph", "ground_truth"):
        return "find_phase"
    return None


def resolve_sqa3d_dynagraph_profile(
    method: SQA3D_MEMORY_BACKEND | str,
    *,
    profile: SQA3DRunProfile,
) -> DynagraphProfileName | None:
    if method != "dynagraph":
        return None
    return "smoke" if profile == "smoke" else "eqa"


def apply_ovmm_backend_dynagraph(
    parameters: Parameters | dict[str, Any],
    backend: OVMM_MEMORY_BACKEND | str,
    *,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
) -> Parameters:
    """Configure dynagraph merge/staleness for OVMM find-phase backend rows."""
    profile = resolve_ovmm_dynagraph_profile(backend)
    if profile is None:
        return _as_parameters(parameters)
    return apply_dynagraph_profile(
        parameters,
        profile,
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
    )


def apply_sqa3d_dynagraph(
    parameters: Parameters | dict[str, Any],
    *,
    method: SQA3D_MEMORY_BACKEND | str,
    profile: SQA3DRunProfile,
) -> Parameters:
    """Configure dynagraph merge/staleness for SQA3D smoke vs tuned runs."""
    dynagraph_profile = resolve_sqa3d_dynagraph_profile(method, profile=profile)
    if dynagraph_profile is None:
        return _as_parameters(parameters)
    return apply_dynagraph_profile(parameters, dynagraph_profile)


def harness_controller_options(
    harness: BenchmarkHarnessName,
    method: str,
    *,
    path: str = DEFAULT_DYNAGRAPH_BENCHMARK_YAML,
) -> dict[str, Any]:
    """Return documented controller kwargs for a benchmark harness (task-specific)."""
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
