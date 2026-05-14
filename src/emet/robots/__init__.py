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

"""Robot backends for EMET — Stretch, Mobile ALOHA, Galaxea R1 / RB-Y1, Innate Mars, YOR."""

import importlib

from emet.robots.base import (
    RobotBackend,
    RobotSpawnSpec,
    RobotSpec,
    format_robot_runtime_notes,
    format_uv_sync_extras_hint,
)

ROBOT_REGISTRY = {
    "stretch": "emet.robots.stretch",
    "mobile_aloha": "emet.robots.mobile_aloha",
    "galaxea_r1": "emet.robots.galaxea_r1",
    "rby1": "emet.robots.rby1",  # Rainbow RB-Y1 (Galaxea R1 family); MolmoSpaces id
    "rb_y1": "emet.robots.rby1",  # same, for --robot rb-y1
    "innate_mars": "emet.robots.innate_mars",
    "yor": "emet.robots.yor",
}


def get_robot_spec(robot: str) -> RobotSpec | None:
    """Return :class:`RobotSpec` for a CLI robot name (``stretch``, ``innate_mars``, …), or None if unknown."""
    key = robot.lower().replace("-", "_")
    if key == "stretch":
        from emet.robots.stretch import StretchBackend

        return StretchBackend().get_spec()
    mod_name = ROBOT_REGISTRY.get(key)
    if mod_name is None:
        return None
    mod = importlib.import_module(mod_name)
    backend_cls = None
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and hasattr(attr, "get_spec") and attr_name != "RobotBackend":
            backend_cls = attr
            break
    if backend_cls is None:
        return None
    return backend_cls().get_spec()


__all__ = [
    "ROBOT_REGISTRY",
    "RobotBackend",
    "RobotSpec",
    "RobotSpawnSpec",
    "format_robot_runtime_notes",
    "format_uv_sync_extras_hint",
    "get_robot_spec",
]
