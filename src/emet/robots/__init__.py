# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Robot backends for EMET — Stretch, Mobile ALOHA, Galaxea R1 / RB-Y1, Innate Mars, Nori A3, YOR."""

import importlib
from typing import Any

from emet.robots.base import RobotBackend, RobotSpec, format_robot_runtime_notes, format_uv_sync_extras_hint

ROBOT_REGISTRY = {
    "stretch": "emet.robots.stretch",
    "mobile_aloha": "emet.robots.mobile_aloha",
    "galaxea_r1": "emet.robots.galaxea_r1",
    "rby1": "emet.robots.rby1",  # Rainbow RB-Y1 (Galaxea R1 family); MolmoSpaces id
    "rb_y1": "emet.robots.rby1",  # same, for --robot rb-y1
    "innate_mars": "emet.robots.innate_mars",
    "xlerobot": "emet.robots.xlerobot",
    "xlerobot_dual": "emet.robots.xlerobot",
    "franka_fr3": "emet.robots.franka_fr3",
    "franka": "emet.robots.franka_fr3",
    "sourccey": "emet.robots.sourccey",
    "yor": "emet.robots.yor",
    "nori": "emet.robots.nori",  # Nori A3 bimanual mobile manipulator
    "nori_a3": "emet.robots.nori",
}


def _backend_class_from_module(mod) -> type[RobotBackend] | None:
    """First :class:`RobotBackend` subclass defined on ``mod`` (not the ABC itself)."""
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, RobotBackend) and attr is not RobotBackend:
            return attr
    return None


def get_robot_spec(robot: str) -> RobotSpec | None:
    """Return :class:`RobotSpec` for a CLI robot name (``stretch``, ``innate_mars``, …), or None if unknown."""
    backend = get_robot_backend(robot)
    return None if backend is None else backend.get_spec()


def get_robot_backend(robot: str) -> RobotBackend | None:
    """Return the :class:`RobotBackend` instance for a CLI robot name, or None if unknown."""
    key = robot.lower().replace("-", "_")
    if key == "stretch":
        from emet.robots.stretch import StretchBackend

        return StretchBackend()
    mod_name = ROBOT_REGISTRY.get(key)
    if mod_name is None:
        return None
    backend_cls = _backend_class_from_module(importlib.import_module(mod_name))
    if backend_cls is None:
        return None
    return backend_cls()


# Global default for ``emet run dynamem --dynav-config`` (shared ``dynav_config.yaml`` unless overridden).
DEFAULT_DYNAV_CONFIG_YAML = "dynav_config.yaml"


def resolve_dynav_config_yaml(robot: str, dynav_config: str) -> str:
    """If *dynav_config* is :data:`DEFAULT_DYNAV_CONFIG_YAML` and the robot spec declares
    :attr:`RobotSpec.default_dynav_config`, return that basename; else return *dynav_config* unchanged.

    Most robots use the shared default YAML. Optional ``RobotSpec.default_dynav_config`` lets a backend
    ship a packaged preset; Innate Mars does **not** set one—use ``--dynav-config dynav_innate_mars.yaml``
    when the ZMQ stack has no hardware depth and you need the DA3 preset.
    """
    if dynav_config != DEFAULT_DYNAV_CONFIG_YAML:
        return dynav_config
    spec = get_robot_spec(robot.lower().replace("-", "_"))
    if spec is None or not spec.default_dynav_config:
        return dynav_config
    return spec.default_dynav_config


def apply_robot_dynav_parameter_overrides(robot: str, parameters: dict[str, Any]) -> None:
    """Merge :attr:`RobotSpec.dynav_parameter_overrides` into a loaded dynav parameter dict."""
    spec = get_robot_spec(robot.lower().replace("-", "_"))
    if spec is None or not spec.dynav_parameter_overrides:
        return
    for key, value in spec.dynav_parameter_overrides.items():
        parameters[key] = value


__all__ = [
    "apply_robot_dynav_parameter_overrides",
    "DEFAULT_DYNAV_CONFIG_YAML",
    "ROBOT_REGISTRY",
    "RobotBackend",
    "RobotSpec",
    "format_robot_runtime_notes",
    "format_uv_sync_extras_hint",
    "get_robot_backend",
    "get_robot_spec",
    "resolve_dynav_config_yaml",
]
