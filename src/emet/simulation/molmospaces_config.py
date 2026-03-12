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
# MolmoSpaces integration: scene/robot name mapping and runner env discovery.
# The actual sim runs in a separate venv (molmo-spaces requires mujoco 3.4, numpy>=2.2).
# See docs/molmospaces.md.

from __future__ import annotations

import os
from pathlib import Path

# Robots supported by MolmoSpaces (from molmo_spaces_constants / their assets).
# rby1 / rby1m are Rainbow Robotics "Galaxea R1" family.
MOLMOSPACES_ROBOT_IDS = [
    "rby1",
    "rby1m",
    "franka_droid",
    "franka_cap",
    "floating_rum",
    "floating_robotiq",
    "franka_fr3",
]

DEFAULT_MOLMOSPACES_ROBOT = "rby1"

# Scene names used by MolmoSpaces get_scenes(scene_name, split).
# ithor = MSCrafted, procthor-10k = MSProc, procthor-objaverse = MSProcObja, holodeck-objaverse = MSMultiType.
MOLMOSPACES_SCENE_NAMES = [
    "ithor",
    "procthor-10k",
    "procthor-objaverse",
    "holodeck-objaverse",
]

MOLMOSPACES_SPLITS = ("train", "val", "test")


def get_molmospaces_runner_python() -> Path | None:
    """Return Python executable for the MolmoSpaces runner (separate venv).
    Prefer MOLMOSPACES_PYTHON env, then .venv-molmospaces in project root.
    """
    env_py = os.environ.get("MOLMOSPACES_PYTHON")
    if env_py:
        p = Path(env_py).resolve()
        if p.exists():
            return p
    root = Path(__file__).resolve().parent.parent.parent.parent
    for name in ("python", "python3"):
        candidate = root / ".venv-molmospaces" / "bin" / name
        if candidate.exists():
            return candidate
    return None
