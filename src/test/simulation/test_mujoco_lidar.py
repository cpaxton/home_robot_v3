# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from emet.robots.innate_mars import get_robot_mjcf_path
from emet.simulation.mujoco_lidar import (
    attach_lidar_to_zmq_message,
    base_lidar_sensor_names,
    lidar_ranges_to_points,
    model_has_base_lidar,
    read_base_lidar_ranges,
)


def test_innate_mars_mjcf_defines_base_lidar_sensors():
    path = get_robot_mjcf_path("innate_mars")
    if not path.is_file():
        pytest.skip("innate_mars MJCF not present")
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model_has_base_lidar(model)
    names = base_lidar_sensor_names(model)
    assert len(names) == 360
    assert names[0] == "base_lidar000"
    assert names[-1] == "base_lidar359"


def test_lidar_points_from_merged_innate_mars_scene():
    from emet.simulation.mujoco_server import _load_default_scene_with_robot

    model = _load_default_scene_with_robot("innate_mars")
    if model is None:
        pytest.skip("Merged innate_mars scene not available")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ranges = read_base_lidar_ranges(data, model)
    assert ranges is not None and ranges.shape == (360,)
    points = lidar_ranges_to_points(ranges)
    assert points.ndim == 2 and points.shape[1] == 2
    msg: dict = {"lidar_points": None, "lidar_timestamp": None}
    attach_lidar_to_zmq_message(msg, model, data)
    assert msg["lidar_points"] is not None
    assert msg["lidar_points"].dtype == np.float32
    assert isinstance(msg["lidar_timestamp"], int)
