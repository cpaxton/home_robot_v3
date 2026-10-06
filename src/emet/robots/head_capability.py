# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Effective head-command limits shared by adapters and inspection preflight."""

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HeadCapability:
    pan: tuple[float, float]
    tilt: tuple[float, float]
    source: str

    def contains(self, angles) -> bool:
        angles = np.asarray(angles, dtype=float)
        return bool(
            angles.shape == (2,)
            and np.isfinite(angles).all()
            and all(lo <= value <= hi for value, (lo, hi) in zip(angles, (self.pan, self.tilt), strict=True))
        )

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_session(cls, session):
        if not isinstance(session, dict):
            return None
        capabilities = session.get("capabilities")
        raw = capabilities.get("head_motion") if isinstance(capabilities, dict) else None
        if not isinstance(raw, dict):
            return None
        try:
            bounds = np.asarray([raw["pan"], raw["tilt"]], dtype=float)
            if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
                return None
            return cls(tuple(bounds[0]), tuple(bounds[1]), str(raw["source"]))
        except (KeyError, TypeError, ValueError):
            return None


STRETCH_LEGACY_HEAD = HeadCapability((-math.pi, math.pi / 4), (-math.pi / 2, 0.0), "legacy_safety")


def mujoco_head_capability(model, joint_names):
    """Intersect direct position-actuator and joint bounds; unsupported mappings abstain."""
    if model is None or joint_names is None:
        return None
    bounds = []
    for name in joint_names:
        if name is None:
            bounds.append((0.0, 0.0))
            continue
        try:
            joint = model.joint(name)
        except (KeyError, ValueError):
            return None
        if not model.jnt_limited[joint.id]:
            return None
        lo, hi = map(float, model.jnt_range[joint.id])
        actuators = np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)
        if not len(actuators):
            return None
        for aid in actuators:
            if (
                model.actuator_trntype[aid] != 0
                or model.actuator_gear[aid, 0] != 1
                or model.actuator_biasprm[aid, 1] >= 0
            ):
                return None
            if model.actuator_ctrllimited[aid]:
                lo = max(lo, float(model.actuator_ctrlrange[aid, 0]))
                hi = min(hi, float(model.actuator_ctrlrange[aid, 1]))
        if lo > hi:
            return None
        bounds.append((lo, hi))
    return HeadCapability(*bounds, source="active_mujoco_model") if len(bounds) == 2 else None
