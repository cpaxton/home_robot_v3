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

from emet.core.zmq_protocol import (
    EMET_ZMQ_ROBOT_ID_KEY,
    is_stretch_family,
    normalize_robot_id,
    read_emet_robot_id,
    robot_ids_match,
)


def test_normalize_robot_id():
    assert normalize_robot_id("RB-Y1") == "rb_y1"
    assert normalize_robot_id("stretch") == "stretch"


def test_robot_ids_match():
    assert robot_ids_match("rby1", "RBY1")
    assert robot_ids_match("hello-stretch", "hello_stretch")


def test_is_stretch_family():
    assert is_stretch_family("stretch")
    assert is_stretch_family("Hello-Stretch")
    assert not is_stretch_family("rby1")


def test_read_emet_robot_id():
    assert read_emet_robot_id(None) is None
    assert read_emet_robot_id({}) is None
    assert read_emet_robot_id({EMET_ZMQ_ROBOT_ID_KEY: "rby1"}) == "rby1"
