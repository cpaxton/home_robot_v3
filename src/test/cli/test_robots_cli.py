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

import subprocess
import sys

import pytest


def test_stereo_right_camera_name_xlerobot():
    from emet.simulation.stereo_camera_utils import stereo_right_camera_name_from_spec

    assert stereo_right_camera_name_from_spec(["head_camera_left", "head_camera_right"]) == "head_camera_right"
    assert stereo_right_camera_name_from_spec(["head_camera", "head_camera_right"]) == "head_camera_right"


def test_robots_list_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "robots", "list", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "MJCF" in result.stdout or "camera" in result.stdout.lower()


def test_robots_info_xlerobot():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "robots", "info", "xlerobot"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "head_camera_left" in result.stdout
    assert "head_camera_right" in result.stdout
    assert "Stereo right" in result.stdout


def test_robots_list_includes_xlerobot_and_franka():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "robots", "list"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "xlerobot" in result.stdout
    assert "franka_fr3" in result.stdout


@pytest.mark.parametrize("robot", ["xlerobot", "franka_fr3"])
def test_robot_spec_cameras_in_mjcf(robot: str):
    import mujoco

    from emet.robots import get_robot_spec

    spec = get_robot_spec(robot)
    assert spec is not None and spec.mjcf_path
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    mjcf_cams = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(int(model.ncam))}
    for cam in spec.camera_names:
        assert cam in mjcf_cams
