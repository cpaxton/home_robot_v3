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
    EMET_ZMQ_SIM_TIME_RATIO_KEY,
    EMET_ZMQ_SIM_WAIT_SCALE_MAX,
    is_stretch_family,
    motion_wait_timeout_scale,
    normalize_robot_id,
    read_emet_robot_id,
    read_sim_to_real_ratio,
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


def test_motion_wait_timeout_scale_absent_and_invalid():
    assert motion_wait_timeout_scale(None) == 1.0
    assert motion_wait_timeout_scale(0.0) == 1.0
    assert motion_wait_timeout_scale(-1.0) == 1.0
    assert motion_wait_timeout_scale(float("nan")) == 1.0
    assert motion_wait_timeout_scale("bad") == 1.0  # type: ignore[arg-type]


def test_motion_wait_timeout_scale_fast_and_slow():
    assert motion_wait_timeout_scale(1.0) == 1.0
    assert motion_wait_timeout_scale(2.0) == 1.0
    assert motion_wait_timeout_scale(0.5) == 2.0
    assert abs(motion_wait_timeout_scale(0.27) - (1.0 / 0.27)) < 1e-9
    assert motion_wait_timeout_scale(0.01) == EMET_ZMQ_SIM_WAIT_SCALE_MAX


def test_read_sim_to_real_ratio():
    assert read_sim_to_real_ratio(None) is None
    assert read_sim_to_real_ratio({}) is None
    assert read_sim_to_real_ratio({EMET_ZMQ_SIM_TIME_RATIO_KEY: None}) is None
    assert read_sim_to_real_ratio({EMET_ZMQ_SIM_TIME_RATIO_KEY: "x"}) is None
    assert read_sim_to_real_ratio({EMET_ZMQ_SIM_TIME_RATIO_KEY: 0.0}) is None
    assert read_sim_to_real_ratio({EMET_ZMQ_SIM_TIME_RATIO_KEY: 0.27}) == 0.27
