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

"""Tests for YAML config path resolution."""

from pathlib import Path

import pytest

from emet.utils.config import get_full_config_path, resolve_config_yaml_path


def test_resolve_packaged_basename():
    p = resolve_config_yaml_path("dynav_config.yaml")
    assert Path(p).name == "dynav_config.yaml"
    assert Path(p).is_file()


def test_resolve_absolute_path(tmp_path):
    src = Path(get_full_config_path("dynav_config.yaml"))
    dst = tmp_path / "my_dynav.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    p = resolve_config_yaml_path(str(dst))
    assert Path(p).resolve() == dst.resolve()


def test_resolve_missing_raises():
    with pytest.raises(FileNotFoundError):
        resolve_config_yaml_path("nonexistent_config_xyz_12345.yaml")
