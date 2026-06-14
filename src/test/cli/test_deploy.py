# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Unit tests for robot deploy helpers."""

from emet.deploy import build_remote_bridge_import_verify_cmd


def test_remote_bridge_import_verify_cmd_includes_emet_core_and_bridge():
    cmd = build_remote_bridge_import_verify_cmd(
        remote_emet="~/emet",
        remote_ws="~/innate-os/ros2_ws",
    )
    assert "~/emet/emet_core" in cmd
    assert "innate_mars_bridge.ros.camera" in cmd
    assert "emet.utils.image" in cmd
    assert "emet.core.server" in cmd
    assert "colcon" not in cmd
