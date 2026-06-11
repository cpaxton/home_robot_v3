# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Load ``configs/ovmm/benchmark.yaml`` (home-dir paths for outputs and caches)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from emet.utils.config import resolve_config_yaml_path

DEFAULT_BENCHMARK_YAML = "configs/ovmm/benchmark.yaml"


@dataclass(frozen=True)
class OvmmBenchmarkPaths:
    """Resolved OVMM benchmark directories (always expanded absolute paths)."""

    output_dir_sim: Path
    output_dir_full: Path
    output_dir_habitat: Path
    habitat_eqa_data: Path
    hm3d_data: Path


@dataclass(frozen=True)
class OvmmBenchmarkConfig:
    paths: OvmmBenchmarkPaths
    sim_episodes_yaml: Path
    full_episodes_yaml: Path
    habitat_episodes_yaml: Path
    smoke_sim_episode_id: str
    smoke_sim_backend: str
    smoke_full_episode_id: str
    smoke_full_backend: str
    smoke_full_manip_mode: str
    smoke_habitat_episode_id: str
    smoke_habitat_backend: str


def _path_from_env_or_yaml(env_key: str, yaml_value: str) -> Path:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(yaml_value).expanduser().resolve()


def load_ovmm_benchmark_config(path: str | Path | None = None) -> OvmmBenchmarkConfig:
    """Load benchmark YAML; ``~`` and env overrides are expanded to absolute paths."""
    full = Path(resolve_config_yaml_path(str(path or DEFAULT_BENCHMARK_YAML)))
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    paths_raw = raw.get("paths") if isinstance(raw, dict) else {}
    episodes_raw = raw.get("episodes") if isinstance(raw, dict) else {}
    smoke_raw = raw.get("smoke") if isinstance(raw, dict) else {}
    if not isinstance(paths_raw, dict):
        paths_raw = {}
    if not isinstance(episodes_raw, dict):
        episodes_raw = {}
    if not isinstance(smoke_raw, dict):
        smoke_raw = {}

    paths = OvmmBenchmarkPaths(
        output_dir_sim=_path_from_env_or_yaml(
            "EMET_OVMM_OUTPUT_SIM",
            str(paths_raw.get("output_dir_sim", "~/runs/emet/ovmm_find_phase")),
        ),
        output_dir_full=_path_from_env_or_yaml(
            "EMET_OVMM_OUTPUT_FULL",
            str(paths_raw.get("output_dir_full", "~/runs/emet/ovmm_full")),
        ),
        output_dir_habitat=_path_from_env_or_yaml(
            "EMET_OVMM_OUTPUT_HABITAT",
            str(paths_raw.get("output_dir_habitat", "~/runs/emet/ovmm_habitat")),
        ),
        habitat_eqa_data=_path_from_env_or_yaml(
            "HABITAT_EQA_DATA_DIR",
            str(paths_raw.get("habitat_eqa_data", "~/.cache/habitat_eqa/data")),
        ),
        hm3d_data=_path_from_env_or_yaml(
            "HM3D_DATA_PATH",
            str(paths_raw.get("hm3d_data", "~/.cache/habitat_eqa/hm3d")),
        ),
    )
    return OvmmBenchmarkConfig(
        paths=paths,
        sim_episodes_yaml=Path(
            resolve_config_yaml_path(str(episodes_raw.get("sim", "configs/ovmm/find_phase_episodes.yaml")))
        ),
        full_episodes_yaml=Path(
            resolve_config_yaml_path(str(episodes_raw.get("full", "configs/ovmm/full_episodes.yaml")))
        ),
        habitat_episodes_yaml=Path(
            resolve_config_yaml_path(str(episodes_raw.get("habitat", "configs/ovmm/habitat_find_phase_episodes.yaml")))
        ),
        smoke_sim_episode_id=str(smoke_raw.get("sim_episode_id", "default_table_s0")),
        smoke_sim_backend=str(smoke_raw.get("sim_backend", "ground_truth")),
        smoke_full_episode_id=str(smoke_raw.get("full_episode_id", "default_table_s0_distinct_recep")),
        smoke_full_backend=str(smoke_raw.get("smoke_full_backend", smoke_raw.get("full_backend", "ground_truth"))),
        smoke_full_manip_mode=str(smoke_raw.get("full_manip_mode", "oracle")),
        smoke_habitat_episode_id=str(smoke_raw.get("habitat_episode_id", "hm3d_lamp_bed_00006")),
        smoke_habitat_backend=str(smoke_raw.get("habitat_backend", "ground_truth")),
    )
