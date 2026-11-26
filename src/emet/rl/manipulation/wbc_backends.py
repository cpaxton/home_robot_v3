# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Backend factory for direct MuJoCo and RoboCasa WBC plants."""

from __future__ import annotations

from typing import Any

from emet.controller.manipulation.wbc_robots import get_wbc_robot
from emet.rl.manipulation.wbc_robocasa import make_robocasa_backend


def make_direct_mujoco_backend(robot: str, **kwargs: Any):
    """Create the direct-MuJoCo plant for a robot, if implemented."""
    key = get_wbc_robot(robot).name
    if key == "innate_mars":
        from emet.rl.manipulation.wbc_mujoco import MuJoCoWbcBackend

        return MuJoCoWbcBackend(**kwargs)
    if key == "stretch":
        from emet.rl.manipulation.wbc_stretch import StretchWbcBackend

        return StretchWbcBackend(**kwargs)
    if key == "sourccey":
        from emet.rl.manipulation.wbc_sourccey import SourcceyWbcBackend

        return SourcceyWbcBackend(**kwargs)
    if key == "rby1":
        from emet.rl.manipulation.wbc_rby1 import Rby1WbcBackend

        return Rby1WbcBackend(**kwargs)
    raise NotImplementedError(
        f"{key} has no packaged direct-MuJoCo WBC plant; use the RoboCasa hook or add an MJCF adapter"
    )


def make_wbc_backend(robot: str, *, sim: str = "mujoco", **kwargs: Any):
    """Create a WBC backend for ``sim='mujoco'`` or ``sim='robocasa'``."""
    key = get_wbc_robot(robot).name
    if sim == "mujoco":
        return make_direct_mujoco_backend(key, **kwargs)
    if sim == "robocasa":
        return make_robocasa_backend(key, **kwargs)
    raise ValueError(f"unknown WBC simulator {sim!r}; expected mujoco or robocasa")


__all__ = ["make_direct_mujoco_backend", "make_wbc_backend"]
