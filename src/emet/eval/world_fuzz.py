# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Environment "fuzzing" for dynamic-world benchmarks: scripted or seeded-random object
relocations (``sim_set_body_pose``) and door/drawer joint changes (``sim_set_joint_qpos``).

Used by the lifelong dynamic exploration runner between cycles. Scripted mode is the
deterministic paper protocol; random mode draws from configured candidate lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class FuzzAction:
    """One world change: relocate a freejoint body or set a named hinge/slide joint."""

    kind: Literal["move", "joint"]
    target: str  # body name for "move", joint name for "joint"
    pos: tuple[float, float, float] | None = None  # absolute world position (move)
    quat: tuple[float, float, float, float] | None = None  # optional wxyz (move)
    value: float | None = None  # joint qpos (joint)

    def as_record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "target": self.target}
        if self.pos is not None:
            out["pos"] = [float(x) for x in self.pos]
        if self.quat is not None:
            out["quat"] = [float(x) for x in self.quat]
        if self.value is not None:
            out["value"] = float(self.value)
        return out


def _placement_pos(placements: dict[str, Any], body: str) -> np.ndarray:
    if body not in placements:
        raise RuntimeError(f"fuzz: body {body!r} not in sim_object_placements ({len(placements)} entries)")
    return np.asarray(placements[body]["pos"], dtype=np.float64).reshape(-1)[:3]


def _move_from_spec(spec: dict[str, Any], placements: dict[str, Any]) -> FuzzAction:
    body = str(spec.get("body", "")).strip()
    if not body:
        raise RuntimeError(f"fuzz: move spec missing 'body': {spec!r}")
    if spec.get("pos") is not None:
        pos = np.asarray(spec["pos"], dtype=np.float64).reshape(3)
    elif spec.get("delta") is not None:
        pos = _placement_pos(placements, body) + np.asarray(spec["delta"], dtype=np.float64).reshape(3)
    else:
        raise RuntimeError(f"fuzz: move spec needs 'pos' or 'delta': {spec!r}")
    quat = None
    if spec.get("quat") is not None:
        quat = tuple(float(x) for x in np.asarray(spec["quat"], dtype=np.float64).reshape(4))
    return FuzzAction(kind="move", target=body, pos=tuple(float(x) for x in pos), quat=quat)


def _joint_from_spec(spec: dict[str, Any]) -> FuzzAction:
    joint = str(spec.get("joint", "")).strip()
    if not joint or spec.get("value") is None:
        raise RuntimeError(f"fuzz: door spec needs 'joint' and 'value': {spec!r}")
    return FuzzAction(kind="joint", target=joint, value=float(spec["value"]))


def scripted_fuzz_actions(
    cycle_spec: dict[str, Any],
    placements: dict[str, Any],
) -> list[FuzzAction]:
    """Build fuzz actions from one YAML cycle spec (``moves:`` + ``doors:`` lists).

    Move specs use ``pos`` (absolute world XYZ) or ``delta`` (offset from the body's
    current GT position). Door specs use ``joint`` + ``value`` (radians or meters).
    """
    actions: list[FuzzAction] = []
    for spec in cycle_spec.get("moves") or []:
        if isinstance(spec, dict):
            actions.append(_move_from_spec(spec, placements))
    for spec in cycle_spec.get("doors") or []:
        if isinstance(spec, dict):
            actions.append(_joint_from_spec(spec))
    return actions


def random_fuzz_actions(
    random_spec: dict[str, Any],
    placements: dict[str, Any],
    rng: np.random.Generator,
) -> list[FuzzAction]:
    """Draw seeded-random fuzz actions from configured candidates.

    ``random_spec`` keys:
      bodies: candidate freejoint body names (required for moves)
      joints: candidate door/drawer joint names (required for joint changes)
      n_moves / n_doors: how many of each per cycle (default 1 each when candidates exist)
      move_radius_m: max XY displacement from the current GT position (default 1.0)
      joint_values: values to sample for joints (default [0.0, 1.2] = closed/open)
    """
    actions: list[FuzzAction] = []
    bodies = [str(b) for b in (random_spec.get("bodies") or [])]
    joints = [str(j) for j in (random_spec.get("joints") or [])]
    n_moves = int(random_spec.get("n_moves", 1 if bodies else 0))
    n_doors = int(random_spec.get("n_doors", 1 if joints else 0))
    radius = float(random_spec.get("move_radius_m", 1.0))
    joint_values = [float(v) for v in (random_spec.get("joint_values") or [0.0, 1.2])]

    if n_moves > 0 and not bodies:
        raise RuntimeError("fuzz: random n_moves > 0 but no candidate 'bodies' configured")
    if n_doors > 0 and not joints:
        raise RuntimeError("fuzz: random n_doors > 0 but no candidate 'joints' configured")

    for body in rng.choice(bodies, size=min(n_moves, len(bodies)), replace=False) if n_moves else []:
        base = _placement_pos(placements, str(body))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        r = float(rng.uniform(0.3, radius))
        pos = base + np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        actions.append(FuzzAction(kind="move", target=str(body), pos=tuple(float(x) for x in pos)))
    for joint in rng.choice(joints, size=min(n_doors, len(joints)), replace=False) if n_doors else []:
        value = float(rng.choice(joint_values))
        actions.append(FuzzAction(kind="joint", target=str(joint), value=value))
    return actions


def fuzz_actions_for_cycle(
    cycle_spec: dict[str, Any] | None,
    placements: dict[str, Any],
    *,
    rng: np.random.Generator | None = None,
) -> list[FuzzAction]:
    """Resolve one cycle's fuzz actions: scripted ``moves``/``doors`` plus optional ``random`` block."""
    if not cycle_spec:
        return []
    actions = scripted_fuzz_actions(cycle_spec, placements)
    random_spec = cycle_spec.get("random")
    if isinstance(random_spec, dict):
        if rng is None:
            rng = np.random.default_rng(int(random_spec.get("seed", 0)))
        actions.extend(random_fuzz_actions(random_spec, placements, rng))
    return actions


def apply_fuzz_actions(robot: Any, actions: list[FuzzAction]) -> list[dict[str, Any]]:
    """Send fuzz actions over the robot ZMQ client; returns applied-action records for logging."""
    from emet.simulation.sim_manipulation import robot_zmq_set_body_pose, robot_zmq_set_joint_qpos

    applied: list[dict[str, Any]] = []
    for action in actions:
        if action.kind == "move":
            assert action.pos is not None
            robot_zmq_set_body_pose(robot, action.target, list(action.pos), quat=action.quat)
        else:
            assert action.value is not None
            robot_zmq_set_joint_qpos(robot, action.target, action.value)
        applied.append(action.as_record())
    return applied
