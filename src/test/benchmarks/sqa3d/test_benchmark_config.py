# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from pathlib import Path

from emet.benchmarks.sqa3d.benchmark_config import load_sqa3d_benchmark_config


def test_load_sqa3d_benchmark_config_expands_home():
    cfg = load_sqa3d_benchmark_config()
    assert str(cfg.paths.output_dir).startswith(str(Path.home()))
    assert cfg.paths.sqa3d_data.name == "data"
    assert cfg.smoke.method == "dynagraph"
    assert cfg.sweep.replay_mode == "sens"
