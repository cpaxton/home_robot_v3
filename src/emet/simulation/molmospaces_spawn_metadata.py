# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Checked-in MolmoSpaces spawn hints colocated with robot MJCF (``molmospaces_spawn.json``).

Runtime code may load optional fields and fall back to heuristics in :mod:`emet.simulation.molmospaces_spawn`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emet.robots.base import RobotSpawnSpec
from emet.utils.assets import get_robot_mjcf_path

METADATA_FILENAME = "molmospaces_spawn.json"


@dataclass
class MolmospacesSpawnMetadata:
    """Subset of JSON fields used by sim spawn (extend as tooling matures)."""

    schema_version: int = 1
    molmospaces_target_foot_clearance_above_floor_m: float | None = None
    molmospaces_nominal_base_height_above_floor_m: float | None = None
    requires_floating_base_spawn_settle: bool = False


def molmospaces_spawn_metadata_path(robot_key: str) -> Path | None:
    """Return ``<robot_mjcf_dir>/molmospaces_spawn.json`` if the robot has a vendored MJCF."""
    p = get_robot_mjcf_path(robot_key.strip().lower().replace("-", "_"))
    if p is None or not p.is_file():
        return None
    return p.parent / METADATA_FILENAME


def load_molmospaces_spawn_metadata(robot_key: str) -> MolmospacesSpawnMetadata | None:
    """Load ``molmospaces_spawn.json`` next to the robot MJCF, or None if missing / invalid."""
    path = molmospaces_spawn_metadata_path(robot_key)
    if path is None or not path.is_file():
        return None
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        ver = int(raw.get("schema_version", 1))
    except (TypeError, ValueError):
        ver = 1

    def _opt_float(key: str) -> float | None:
        v = raw.get(key)
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    settle = raw.get("requires_floating_base_spawn_settle", False)
    return MolmospacesSpawnMetadata(
        schema_version=ver,
        molmospaces_target_foot_clearance_above_floor_m=_opt_float("molmospaces_target_foot_clearance_above_floor_m"),
        molmospaces_nominal_base_height_above_floor_m=_opt_float("molmospaces_nominal_base_height_above_floor_m"),
        requires_floating_base_spawn_settle=bool(settle),
    )


def robot_spawn_spec_from_metadata(robot_key: str) -> RobotSpawnSpec | None:
    """Build :class:`~emet.robots.base.RobotSpawnSpec` from checked-in JSON, if present."""
    meta = load_molmospaces_spawn_metadata(robot_key)
    if meta is None:
        return None
    return RobotSpawnSpec(
        molmospaces_target_foot_clearance_above_floor_m=meta.molmospaces_target_foot_clearance_above_floor_m,
        molmospaces_nominal_base_height_above_floor_m=meta.molmospaces_nominal_base_height_above_floor_m,
        requires_floating_base_spawn_settle=meta.requires_floating_base_spawn_settle,
    )
