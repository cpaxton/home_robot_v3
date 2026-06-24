# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Dynav preset resolution for emet stream/capture on innate_mars hardware."""

from emet.app.stream_agent_factory import (
    INNATE_MARS_HW_DYNAV,
    resolve_stream_dynav_config,
)
from emet.robots import DEFAULT_DYNAV_CONFIG_YAML


def test_innate_mars_remote_host_uses_da3_dynav_by_default():
    resolved = resolve_stream_dynav_config(
        "innate_mars",
        "herman",
        DEFAULT_DYNAV_CONFIG_YAML,
        dynav_from_default=True,
    )
    assert resolved == INNATE_MARS_HW_DYNAV


def test_innate_mars_localhost_keeps_sensor_dynav_by_default():
    resolved = resolve_stream_dynav_config(
        "innate_mars",
        "127.0.0.1",
        DEFAULT_DYNAV_CONFIG_YAML,
        dynav_from_default=True,
    )
    assert resolved == DEFAULT_DYNAV_CONFIG_YAML


def test_explicit_dynav_not_overridden():
    resolved = resolve_stream_dynav_config(
        "innate_mars",
        "herman",
        "dynav_config.yaml",
        dynav_from_default=False,
    )
    assert resolved == "dynav_config.yaml"
