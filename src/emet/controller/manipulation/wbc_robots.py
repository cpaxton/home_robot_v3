# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Robot and simulator hooks for the whole-body contract.

The profile is the shared contract; the plant/simulator adapter is separate.
``direct_mujoco`` marks whether a backend exists in this checkout, while
``robocasa_robot`` is the RoboCasa/Robosuite name or composition hook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from emet.controller.manipulation.wbc_contract import EmbodimentProfile
from emet.controller.manipulation.wbc_profiles import (
    mars_sim_profile,
    pandaomron_sim_profile,
    rby1_sim_profile,
    sourccey_sim_profile,
    stretch_sim_profile,
)


@dataclass(frozen=True)
class WbcRobot:
    """One robot in the WBC matrix."""

    name: str
    profile_factory: Callable[[], EmbodimentProfile]
    direct_mujoco: bool
    robocasa_robot: str | None
    notes: str = ""


WBC_ROBOTS: dict[str, WbcRobot] = {
    "innate_mars": WbcRobot(
        "innate_mars",
        mars_sim_profile,
        True,
        None,
        "MuJoCo adapter-matched Mars; DAgger result available.",
    ),
    "mars": WbcRobot(
        "innate_mars",
        mars_sim_profile,
        True,
        None,
        "Alias for innate_mars.",
    ),
    "stretch": WbcRobot(
        "stretch",
        stretch_sim_profile,
        True,
        "Stretch",
        "Direct-MuJoCo wheel plant; teacher smoke passes.",
    ),
    "sourccey": WbcRobot(
        "sourccey",
        sourccey_sim_profile,
        True,
        "Sourccey",
        "Direct-MuJoCo planar base + dual-arm plant; teacher smoke passes.",
    ),
    "rby1": WbcRobot(
        "rby1",
        rby1_sim_profile,
        True,
        "galaxea_r1",
        "Galaxea R1 proxy direct-MuJoCo plant; RoboCasa uses the strip-and-replace rby1 MJCF path.",
    ),
    "rb_y1": WbcRobot(
        "rby1",
        rby1_sim_profile,
        True,
        "galaxea_r1",
        "Alias for rby1.",
    ),
    "pandaomron": WbcRobot(
        "pandaomron",
        pandaomron_sim_profile,
        True,
        "PandaOmron",
        "RoboCasa Omron + Panda composition with a direct composed MuJoCo plant.",
    ),
    "panda_omron": WbcRobot(
        "pandaomron",
        pandaomron_sim_profile,
        True,
        "PandaOmron",
        "Alias for pandaomron.",
    ),
}


def get_wbc_robot(name: str) -> WbcRobot:
    key = name.lower().replace("-", "_")
    try:
        return WBC_ROBOTS[key]
    except KeyError as exc:
        raise KeyError(f"unknown WBC robot {name!r}; known: {sorted(set(WBC_ROBOTS))}") from exc


def wbc_profile_for(name: str) -> EmbodimentProfile:
    return get_wbc_robot(name).profile_factory()


def robocasa_robot_name(name: str) -> str | None:
    return get_wbc_robot(name).robocasa_robot


__all__ = [
    "WBC_ROBOTS",
    "WbcRobot",
    "get_wbc_robot",
    "robocasa_robot_name",
    "wbc_profile_for",
]
