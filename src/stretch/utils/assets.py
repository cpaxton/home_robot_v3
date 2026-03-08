# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Central location for robot model assets. Assets are stored in stretch/assets/
# and installed with the package.

import importlib.resources
from pathlib import Path


def get_assets_root() -> Path:
    """Return the path to the stretch package's assets directory."""
    return Path(str(importlib.resources.files("stretch") / "assets"))


def get_robot_assets_path() -> Path:
    """Return the path to robot model assets (MuJoCo XML, meshes, etc.)."""
    return get_assets_root() / "robot"


def get_mujoco_models_path() -> Path:
    """Return the path to MuJoCo robot models (stretch.xml, scene.xml, etc.)."""
    return get_robot_assets_path()
