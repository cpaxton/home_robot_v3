# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Simulator-only articulation descriptors; no category or object-name rules.

The assisted convention calls a default joint pose at a limit 'closed' and the
opposite limit 'open'. It does not establish physical access or collision safety.
"""

from __future__ import annotations

import mujoco
import numpy as np


def scene_articulations(model, data, robot_root_name):
    """Group scalar scene joints by their top-level fixture, excluding the robot.

    Multi-joint fixtures, free objects, unbounded joints and interior reference
    poses are exposed as unsupported, never guessed into actionable endpoints.
    """
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name)
    roots = {}
    for bid in range(1, model.nbody):
        path, parent = [], bid
        while parent:
            path.append(parent)
            parent = int(model.body_parentid[parent])
        if robot in path:
            continue
        roots.setdefault(path[-1], []).append(bid)
    groups = []
    for bodies in roots.values():
        joints = [
            j
            for b in bodies
            for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b] + model.body_jntnum[b]))
        ]
        scalar = [j for j in joints if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        if not scalar:
            continue
        group = {
            "bodies": [model.body(b).name for b in bodies],
            "supported": False,
            "state": "unknown",
            "convention": "default_limit_is_closed",
            "joint_positions": {model.joint(j).name: float(data.qpos[model.jnt_qposadr[j]]) for j in scalar},
        }
        if len(joints) == 1 and len(scalar) == 1 and model.joint(scalar[0]).name:
            j = scalar[0]
            lo, hi = model.jnt_range[j]
            ref = float(model.qpos0[model.jnt_qposadr[j]])
            if model.jnt_limited[j] and np.isfinite([lo, hi, ref]).all() and hi - lo > 1e-6:
                if abs(ref - lo) < 1e-6 or abs(ref - hi) < 1e-6:
                    closed = float(lo if abs(ref - lo) < 1e-6 else hi)
                    opened = float(hi if closed == lo else lo)
                    position = float(data.qpos[model.jnt_qposadr[j]])
                    tolerance = max(1e-5, float(hi - lo) * 0.02)
                    state = (
                        "open"
                        if abs(position - opened) <= tolerance
                        else "closed"
                        if abs(position - closed) <= tolerance
                        else "partial"
                    )
                    group.update(
                        supported=True,
                        joint=model.joint(j).name,
                        closed=closed,
                        open=opened,
                        position=position,
                        tolerance=tolerance,
                        state=state,
                    )
        groups.append(group)
    return groups


def articulation_for_body(session, body):
    """Return a private descriptor or None for an unarticulated body."""
    matches = [g for g in session.get("sim_articulations", []) if body in g.get("bodies", [])]
    return matches[0] if len(matches) == 1 else None


def access_precondition(session, body):
    """Conservative whole-fixture gate; even exterior support tops require open."""
    if not (session.get("capabilities") or {}).get("sim_articulation_state"):
        return None  # Legacy providers cannot establish articulation state.
    if not isinstance(session.get("sim_articulations"), list):
        return "articulation_state_unavailable"
    group = articulation_for_body(session, body)
    if group is None:
        return None
    if not group.get("supported"):
        return "articulation_unsupported"
    return None if group.get("state") == "open" else "receptacle_requires_open"


def articulation_geometry_changed(current, previous, *, tolerance=1e-6):
    """Compare against the last geometry rebuild, not the last telemetry sample.

    Joint jitter below one micrometre/microradian does not rescan meshes. State
    transitions and structural changes always refresh; accumulated drift does too.
    """
    if previous is None:
        return True
    ignored = {"position", "joint_positions"}
    if {k: v for k, v in current.items() if k not in ignored} != {
        k: v for k, v in previous.items() if k not in ignored
    }:
        return True
    a, b = current.get("joint_positions", {}), previous.get("joint_positions", {})
    return a.keys() != b.keys() or any(
        not np.isfinite(a[k]) or not np.isfinite(b[k]) or abs(a[k] - b[k]) > tolerance for k in a
    )


def articulation_access(session, body):
    """Public access evidence and available assisted actions for task admission."""
    caps = session.get("capabilities") or {}
    if not caps.get("sim_articulation_state"):
        return {"state": "unknown", "required_action": None, "available_actions": []}
    group = articulation_for_body(session, body)
    from emet.simulation.physical_execution import physical_execution_enabled
    actionable = (group and group.get("supported") and caps.get("sim_set_joint_qpos")
                  and session.get("is_simulation") and not physical_execution_enabled())
    return {"state": group.get("state", "unknown") if group else "not_articulated",
            "required_action": access_precondition(session, body),
            "available_actions": ["open", "closed"] if actionable else []}
