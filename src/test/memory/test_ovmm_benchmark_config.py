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

from __future__ import annotations

from pathlib import Path

from emet.eval.ovmm_benchmark_config import load_ovmm_benchmark_config


def test_load_ovmm_benchmark_config_expands_home():
    cfg = load_ovmm_benchmark_config()
    assert str(cfg.paths.output_dir_sim).startswith(str(Path.home()))
    assert str(cfg.paths.output_dir_full).startswith(str(Path.home()))
    assert str(cfg.paths.output_dir_habitat).startswith(str(Path.home()))
    assert cfg.paths.habitat_eqa_data.name == "data"
    assert cfg.smoke_sim_episode_id == "default_table_s0"
    assert cfg.smoke_full_manip_mode == "oracle"
