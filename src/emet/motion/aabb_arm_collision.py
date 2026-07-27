# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Axis-aligned box obstacles for kinematic arm planning (table solids, etc.)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Aabb3:
    """Inclusive world-frame AABB (mins / maxs)."""

    mins: np.ndarray
    maxs: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "mins", np.asarray(self.mins, dtype=np.float64).reshape(3))
        object.__setattr__(self, "maxs", np.asarray(self.maxs, dtype=np.float64).reshape(3))

    @classmethod
    def from_center_half(cls, center: Sequence[float], half: Sequence[float]) -> Aabb3:
        c = np.asarray(center, dtype=np.float64).reshape(3)
        h = np.asarray(half, dtype=np.float64).reshape(3)
        return cls(mins=c - h, maxs=c + h)

    def inflate(self, margin: float) -> Aabb3:
        m = float(margin)
        return Aabb3(mins=self.mins - m, maxs=self.maxs + m)

    def contains(self, xyz: Sequence[float]) -> bool:
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        return bool(np.all(p >= self.mins) and np.all(p <= self.maxs))


def default_table_scene_aabb(*, margin: float = 0.01) -> Aabb3:
    """AABB for the wood table in ``scene_environment.xml`` (body ``table``).

    ``pos="0 -1 .24"`` with box half-sizes ``.6 .5 .24``.
    """
    return Aabb3.from_center_half((0.0, -1.0, 0.24), (0.6, 0.5, 0.24)).inflate(margin)


def fk_link_xyz_samples(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: Sequence[str],
) -> list[np.ndarray]:
    """World XYZ of named bodies (caller sets arm/base qpos; this runs ``mj_forward``)."""
    mujoco.mj_forward(model, data)
    out: list[np.ndarray] = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        if bid < 0:
            continue
        out.append(np.asarray(data.body(bid).xpos, dtype=np.float64).reshape(3).copy())
    return out


class AabbArmCollisionChecker:
    """Reject arm configs whose link origins fall inside any world AABB (e.g. table solid)."""

    def __init__(
        self,
        boxes: Sequence[Aabb3],
        *,
        link_bodies: Sequence[str],
    ) -> None:
        self.boxes = list(boxes)
        self.link_bodies = list(link_bodies)

    @classmethod
    def for_default_table(cls, *, link_bodies: Sequence[str], margin: float = 0.01) -> AabbArmCollisionChecker:
        return cls([default_table_scene_aabb(margin=margin)], link_bodies=link_bodies)

    def configuration_collides(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        if not self.boxes:
            return False
        for xyz in fk_link_xyz_samples(model, data, self.link_bodies):
            for box in self.boxes:
                if box.contains(xyz):
                    return True
        return False
