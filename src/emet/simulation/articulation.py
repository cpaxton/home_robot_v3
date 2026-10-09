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
        if len(joints) == 1 and len(scalar) == 1:
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
