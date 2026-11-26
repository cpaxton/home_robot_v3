# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""RoboCasa/Robosuite registration hooks for the WBC contract.

The RoboCasa benchmark uses composite controllers and its own task/reset
interface; it is not a drop-in ``MjData`` plant. This module is the explicit
boundary: a backend factory registers itself for a robot once the adapter is
available. Until then, calls fail closed instead of silently training against
the wrong controller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from emet.controller.manipulation.wbc_robots import get_wbc_robot


@dataclass(frozen=True)
class RoboCasaHook:
    """Registration record for one RoboCasa robot/adaptor."""

    robot: str
    robocasa_robot: str
    factory: Callable[..., Any] | None = None
    notes: str = ""

    @property
    def available(self) -> bool:
        return self.factory is not None


_HOOKS: dict[str, RoboCasaHook] = {
    "innate_mars": RoboCasaHook(
        "innate_mars",
        "innate_mars",
        notes="Mars is trained in direct MuJoCo; RoboCasa integration is separate.",
    ),
    "stretch": RoboCasaHook(
        "stretch",
        "Stretch",
        notes="RoboCasa composite-controller adapter not yet registered.",
    ),
    "sourccey": RoboCasaHook(
        "sourccey",
        "Sourccey",
        notes="RoboCasa composite-controller adapter not yet registered.",
    ),
    "rby1": RoboCasaHook(
        "rby1",
        "galaxea_r1",
        notes="RoboCasa uses the strip-and-replace Galaxea R1 MJCF path.",
    ),
    "pandaomron": RoboCasaHook(
        "pandaomron",
        "PandaOmron",
        notes="Omron + Panda composite controller; adapter not yet registered.",
    ),
}


def register_robocasa_backend(robot: str, factory: Callable[..., Any], *, notes: str = "") -> None:
    """Register a backend factory for a robot's RoboCasa adapter."""
    key = get_wbc_robot(robot).name
    if key not in _HOOKS:
        raise KeyError(f"no RoboCasa hook for {robot!r}")
    current = _HOOKS[key]
    _HOOKS[key] = RoboCasaHook(current.robot, current.robocasa_robot, factory, notes or current.notes)


def robocasa_hook(robot: str) -> RoboCasaHook:
    return _HOOKS[get_wbc_robot(robot).name]


def make_robocasa_backend(robot: str, **kwargs: Any):
    hook = robocasa_hook(robot)
    if hook.factory is None:
        raise NotImplementedError(
            f"RoboCasa adapter for {hook.robot!r} ({hook.robocasa_robot}) is not registered. "
            "Register it with register_robocasa_backend(); direct MuJoCo training must not be "
            "presented as RoboCasa benchmark parity."
        )
    return hook.factory(**kwargs)


__all__ = [
    "RoboCasaHook",
    "make_robocasa_backend",
    "register_robocasa_backend",
    "robocasa_hook",
]
