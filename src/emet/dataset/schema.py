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

"""Stable JSON-friendly schema for simulator object ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ObjectRecord:
    """One manipulable / scene object instance in world frame (MuJoCo)."""

    name: str
    body_name: str
    pos_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    aabb_min_xyz: tuple[float, float, float] | None = None
    aabb_max_xyz: tuple[float, float, float] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = GT_SCHEMA_VERSION
        return d


def object_record_from_dict(d: dict[str, Any]) -> ObjectRecord:
    """Parse a dict produced by :meth:`ObjectRecord.to_json_dict` or the ZMQ payload."""
    return ObjectRecord(
        name=str(d["name"]),
        body_name=str(d["body_name"]),
        pos_xyz=_triple(d["pos_xyz"]),
        quat_wxyz=_quad(d["quat_wxyz"]),
        aabb_min_xyz=_optional_triple(d.get("aabb_min_xyz")),
        aabb_max_xyz=_optional_triple(d.get("aabb_max_xyz")),
    )


def _triple(v: Any) -> tuple[float, float, float]:
    t = tuple(float(x) for x in v)  # type: ignore[arg-type]
    if len(t) != 3:
        raise ValueError(f"expected length-3 xyz, got {v!r}")
    return t[0], t[1], t[2]


def _quad(v: Any) -> tuple[float, float, float, float]:
    t = tuple(float(x) for x in v)  # type: ignore[arg-type]
    if len(t) != 4:
        raise ValueError(f"expected length-4 quat, got {v!r}")
    return t[0], t[1], t[2], t[3]


def _optional_triple(v: Any) -> tuple[float, float, float] | None:
    if v is None:
        return None
    return _triple(v)
