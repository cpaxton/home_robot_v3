# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.robots.galaxea_r1 import R1_JOINT_NAMES
from emet.visualization.mjcf_rerun_robot import MjcfBodySkeletonLogger

_MJCF = Path(__file__).resolve().parents[2] / "emet" / "assets" / "robot" / "galaxea_r1" / "galaxea_r1.xml"


@pytest.mark.skipif(not _MJCF.is_file(), reason="galaxea_r1 MJCF not present")
def test_mjcf_body_skeleton_logger_smoke():
    pytest.importorskip("mujoco")
    log = MjcfBodySkeletonLogger(_MJCF, R1_JOINT_NAMES, 26, "base_link")
    obs = {
        "gps": [0.1, -0.2],
        "compass": [0.3],
        "joint": [0.01] * 26,
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [1.0, 2.0, 0.5]},
    }
    log.apply_and_log(obs)
