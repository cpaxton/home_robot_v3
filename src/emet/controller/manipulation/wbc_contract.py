# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Frozen contract for the whole-body RL reaching primitive (sim-to-real).

This is the interface the training loop and the executor agree on, and it must stay
stable so the case library + generators + policy can interoperate.

- **Observation** is real-world-derivable only: joint encoders, odometry, EE pose from
  forward kinematics, and a local 3D occupancy crop from depth. **No GT object pose.**
- **Action** is base velocity + arm/torso joint **torques** (a true whole-body control).
- **Reward** is computed from the same signals.

See ``docs/plans/whole-body-rl-control.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# EE poses are (position xyz, quaternion xyzw).
EE_POSE_DIM = 7


@dataclass(frozen=True)
class WbcObservation:
    """One observation for the whole-body reaching primitive."""

    q: tuple[float, ...]  # joint positions (n_joints)
    qd: tuple[float, ...]  # joint velocities (n_joints)
    base_xyt: tuple[float, float, float]  # base pose (x, y, theta) from odometry
    ee_pose: tuple[float, ...]  # EE pose (7: pos xyz + quat xyzw) from FK
    ee_target: tuple[float, ...]  # target EE pose (7) from the grasp/place planner
    occupancy: np.ndarray  # local 3D binary occupancy crop (H, W, D), base frame
    failure_mode: str = ""  # optional prior: '' | tracking_lag | unreachable | collision | joint_bounds

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": list(self.q),
            "qd": list(self.qd),
            "base_xyt": list(self.base_xyt),
            "ee_pose": list(self.ee_pose),
            "ee_target": list(self.ee_target),
            "occupancy": np.asarray(self.occupancy, dtype=np.uint8).tolist(),
            "failure_mode": self.failure_mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WbcObservation:
        return cls(
            q=tuple(float(x) for x in d["q"]),
            qd=tuple(float(x) for x in d["qd"]),
            base_xyt=tuple(float(x) for x in d["base_xyt"]),
            ee_pose=tuple(float(x) for x in d["ee_pose"]),
            ee_target=tuple(float(x) for x in d["ee_target"]),
            occupancy=np.asarray(d["occupancy"], dtype=np.uint8),
            failure_mode=str(d.get("failure_mode", "")),
        )


@dataclass(frozen=True)
class WbcAction:
    """One whole-body action: base velocity + arm/torso joint torques."""

    base_velocity: tuple[float, float, float]  # (vx, vy, omega)
    joint_torques: tuple[float, ...]  # (n_joints) torques

    def to_dict(self) -> dict[str, Any]:
        return {"base_velocity": list(self.base_velocity), "joint_torques": list(self.joint_torques)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WbcAction:
        return cls(
            base_velocity=tuple(float(x) for x in d["base_velocity"]),
            joint_torques=tuple(float(x) for x in d["joint_torques"]),
        )


@dataclass(frozen=True)
class ReachSpec:
    """Reward weights and tolerances for the reaching primitive (freeze these)."""

    pos_weight: float = 1.0  # -pos_weight * ||ee_pos - target_pos||
    quat_weight: float = 0.25  # -quat_weight * (1 - |<ee_quat, target_quat>|)
    collision_penalty: float = 10.0  # added when the robot contacts an obstacle
    action_penalty: float = 0.01  # -action_penalty * ||action||^2 (energy/smoothness)
    jerk_penalty: float = 0.0  # -jerk_penalty * ||torques_{t+1} - torques_t||
    success_bonus: float = 20.0  # added when the EE is within tolerance
    pos_tolerance_m: float = 0.02  # success when EE pos error <= this
    quat_tolerance: float = 0.05  # success when quat alignment error <= this


def _ee_pos_err(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.linalg.norm(np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64)))


def _ee_quat_err(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    qa = np.asarray(a[3:7], dtype=np.float64)
    qb = np.asarray(b[3:7], dtype=np.float64)
    qa = qa / (np.linalg.norm(qa) + 1e-12)
    qb = qb / (np.linalg.norm(qb) + 1e-12)
    return float(1.0 - abs(float(np.dot(qa, qb))))


def reach_reward(
    action: WbcAction,
    next_obs: WbcObservation,
    *,
    collision: bool = False,
    prev_torques: tuple[float, ...] | None = None,
    spec: ReachSpec = ReachSpec(),
) -> tuple[float, dict[str, float]]:
    """Dense reward for one reaching step, with the components split for logging.

    Returns ``(reward, components)``. The components are the raw terms (before weights)
    so the training loop can log and reweight without changing the contract.
    """
    pos_err = _ee_pos_err(next_obs.ee_pose, next_obs.ee_target)
    quat_err = _ee_quat_err(next_obs.ee_pose, next_obs.ee_target)
    action_norm_sq = float(np.sum(np.square(np.asarray(action.joint_torques, dtype=np.float64)))) + float(
        np.sum(np.square(np.asarray(action.base_velocity, dtype=np.float64)))
    )
    jerk = 0.0
    if prev_torques is not None:
        jerk = float(np.sum(np.square(np.asarray(action.joint_torques, dtype=np.float64) - np.asarray(prev_torques, dtype=np.float64))))

    reward = -spec.pos_weight * pos_err - spec.quat_weight * quat_err - spec.action_penalty * action_norm_sq
    reward -= spec.jerk_penalty * jerk
    if collision:
        reward -= spec.collision_penalty
    success = pos_err <= spec.pos_tolerance_m and quat_err <= spec.quat_tolerance
    if success:
        reward += spec.success_bonus

    components = {
        "pos_err": pos_err,
        "quat_err": quat_err,
        "action_norm_sq": action_norm_sq,
        "jerk": jerk,
        "collision": float(collision),
        "success": float(success),
    }
    return float(reward), components
