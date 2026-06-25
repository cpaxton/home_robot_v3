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

"""Dynav / config preset resolution for emet stream/capture."""

from emet.app.stream_agent_factory import (
    resolve_stream_config_path,
    resolve_stream_dynav_config,
)
from emet.config.loader import default_config_path
from emet.robots import DEFAULT_DYNAV_CONFIG_YAML


def test_default_dynav_resolves_to_unified_config():
    resolved = resolve_stream_config_path(DEFAULT_DYNAV_CONFIG_YAML, dynav_from_default=True)
    assert resolved == default_config_path()


def test_explicit_dynav_not_overridden():
    resolved = resolve_stream_dynav_config(
        "innate_mars",
        "herman",
        "dynav_config.yaml",
        dynav_from_default=False,
    )
    assert resolved == "dynav_config.yaml"


def test_legacy_alias_resolve_stream_dynav_compat():
    resolved = resolve_stream_dynav_config(
        "innate_mars",
        "127.0.0.1",
        DEFAULT_DYNAV_CONFIG_YAML,
        dynav_from_default=True,
    )
    assert resolved == default_config_path()
