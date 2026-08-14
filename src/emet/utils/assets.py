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
# Central location for robot model assets. Assets are stored in emet/assets/
# and installed with the package.

import importlib.resources
from pathlib import Path


def get_assets_root() -> Path:
    """Return the path to the stretch package's assets directory."""
    return Path(str(importlib.resources.files("emet") / "assets"))


def get_robot_assets_path() -> Path:
    """Return the path to robot model assets (MuJoCo XML, meshes, etc.)."""
    return get_assets_root() / "robot"


def get_mujoco_models_path() -> Path:
    """Return the path to MuJoCo robot models (stretch.xml, scene.xml, etc.)."""
    return get_robot_assets_path()


def get_robot_mjcf_path(robot_key: str) -> Path | None:
    """Return path to the robot MJCF for merges (MolmoSpaces, bundled table scenes), or None."""
    robot_key = robot_key.lower().replace("-", "_")
    if robot_key in ("rby1", "galaxea_r1", "rb_y1"):
        path = get_mujoco_models_path() / "galaxea_r1" / "galaxea_r1.xml"
        return path if path.exists() else None
    if robot_key in ("innate_mars", "maurice"):
        path = get_mujoco_models_path() / "innate_mars" / "innate_mars.xml"
        return path if path.exists() else None
    if robot_key in ("xlerobot", "xlerobot_dual"):
        path = get_mujoco_models_path() / "xlerobot" / "xlerobot.xml"
        return path if path.exists() else None
    if robot_key in ("sourccey",):
        path = get_mujoco_models_path() / "sourccey" / "sourccey.xml"
        return path if path.exists() else None
    if robot_key in ("franka_fr3", "franka"):
        for name in ("franka_fr3.xml", "fr3.xml"):
            path = get_mujoco_models_path() / "franka_fr3" / name
            if path.exists():
                return path
        return None
    if robot_key in ("stretch", "hello_stretch", "hellostretch"):
        path = get_mujoco_models_path() / "stretch.xml"
        return path if path.exists() else None
    return None
