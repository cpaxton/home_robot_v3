# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Load ``configs/sqa3d/benchmark.yaml`` (home-dir paths for outputs and caches)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from emet.utils.config import resolve_config_yaml_path

DEFAULT_BENCHMARK_YAML = "configs/sqa3d/benchmark.yaml"


@dataclass(frozen=True)
class Sqa3dBenchmarkPaths:
    """Resolved SQA3D benchmark directories (always expanded absolute paths)."""

    output_dir: Path
    sqa3d_data: Path
    scannet_root: Path


@dataclass(frozen=True)
class Sqa3dSmokeConfig:
    split: str
    question_id: int
    method: str
    replay_mode: str
    mock_llm: bool


@dataclass(frozen=True)
class Sqa3dSweepConfig:
    split: str
    question_start: int
    question_end: int
    method: str
    replay_mode: str
    isolate_episodes: bool


@dataclass(frozen=True)
class Sqa3dBenchmarkConfig:
    paths: Sqa3dBenchmarkPaths
    smoke: Sqa3dSmokeConfig
    sweep: Sqa3dSweepConfig


def _path_from_env_or_yaml(env_key: str, yaml_value: str) -> Path:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(yaml_value).expanduser().resolve()


def load_sqa3d_benchmark_config(path: str | Path | None = None) -> Sqa3dBenchmarkConfig:
    """Load benchmark YAML; ``~`` and env overrides are expanded to absolute paths."""
    full = Path(resolve_config_yaml_path(str(path or DEFAULT_BENCHMARK_YAML)))
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    paths_raw = raw.get("paths") if isinstance(raw, dict) else {}
    smoke_raw = raw.get("smoke") if isinstance(raw, dict) else {}
    sweep_raw = raw.get("sweep") if isinstance(raw, dict) else {}
    if not isinstance(paths_raw, dict):
        paths_raw = {}
    if not isinstance(smoke_raw, dict):
        smoke_raw = {}
    if not isinstance(sweep_raw, dict):
        sweep_raw = {}

    paths = Sqa3dBenchmarkPaths(
        output_dir=_path_from_env_or_yaml(
            "EMET_SQA3D_OUTPUT",
            str(paths_raw.get("output_dir", "~/runs/emet/sqa3d")),
        ),
        sqa3d_data=_path_from_env_or_yaml(
            "SQA3D_DATA_DIR",
            str(paths_raw.get("sqa3d_data", "~/.cache/sqa3d/data")),
        ),
        scannet_root=_path_from_env_or_yaml(
            "SCANNET_ROOT",
            str(paths_raw.get("scannet_root", "~/.cache/scannet")),
        ),
    )
    return Sqa3dBenchmarkConfig(
        paths=paths,
        smoke=Sqa3dSmokeConfig(
            split=str(smoke_raw.get("split", "val")),
            question_id=int(smoke_raw.get("question_id", 220602000000)),
            method=str(smoke_raw.get("method", "dynagraph")),
            replay_mode=str(smoke_raw.get("replay_mode", "auto")),
            mock_llm=bool(smoke_raw.get("mock_llm", True)),
        ),
        sweep=Sqa3dSweepConfig(
            split=str(sweep_raw.get("split", "val")),
            question_start=int(sweep_raw.get("question_start", 0)),
            question_end=int(sweep_raw.get("question_end", 30)),
            method=str(sweep_raw.get("method", "dynagraph")),
            replay_mode=str(sweep_raw.get("replay_mode", "sens")),
            isolate_episodes=bool(sweep_raw.get("isolate_episodes", True)),
        ),
    )
