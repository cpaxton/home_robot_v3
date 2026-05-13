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

from emet.robots.base import RobotBackend, RobotSpec, format_robot_runtime_notes, format_uv_sync_extras_hint

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


DEFAULT_DYNAV_CONFIG_YAML = "dynav_config.yaml"
"""Global default for ``emet run dynamem --dynav-config`` (sensor depth unless overridden per robot)."""


def resolve_dynav_config_yaml(robot: str, dynav_config: str) -> str:
    """If *dynav_config* is :data:`DEFAULT_DYNAV_CONFIG_YAML` and the robot spec declares
    :attr:`RobotSpec.default_dynav_config`, return that basename; else return *dynav_config* unchanged.
    """
    if dynav_config != DEFAULT_DYNAV_CONFIG_YAML:
        return dynav_config
    spec = get_robot_spec(robot.lower().replace("-", "_"))
    if spec is None or not spec.default_dynav_config:
        return dynav_config
    return spec.default_dynav_config


__all__ = [
    "DEFAULT_DYNAV_CONFIG_YAML",
    "ROBOT_REGISTRY",
    "RobotBackend",
    "RobotSpec",
    "format_robot_runtime_notes",
    "format_uv_sync_extras_hint",
    "get_robot_spec",
    "resolve_dynav_config_yaml",
]
