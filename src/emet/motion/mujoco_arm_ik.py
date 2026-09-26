# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""MuJoCo Jacobian position IK for dual-arm / registry robots (rby1, …).

Lightweight damped least-squares on ``mj_jacBody`` — not collision-aware motion
planning. Use as a smoke / seed for later CuRobo or sampling-based planners.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Galaxea R1 / RB-Y1 defaults (see galaxea_r1.xml).
RBY1_LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"left_arm_joint{i}" for i in range(1, 7))
RBY1_RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"right_arm_joint{i}" for i in range(1, 7))
RBY1_LEFT_EE_BODY = "left_arm_link6"
RBY1_RIGHT_EE_BODY = "right_arm_link6"


@dataclass(frozen=True)
class MujocoArmIkResult:
    success: bool
    qpos: np.ndarray
    pos_error_m: float
    iterations: int
    orientation_error_rad: float | None = None


def solve_pose_ik(
    model,
    data,
    *,
    ee_body: str,
    joint_names,
    target_pos,
    target_rotation,
    max_iters: int = 160,
    tol_m: float = 0.01,
    tol_rad: float = 0.1,
    damping: float = 0.02,
    step: float = 0.5,
    coupled_groups=(),
    joint_limit_margins=None,
) -> MujocoArmIkResult:
    """Pose IK on an offline model, with optional equal-motion joint groups.

    Coupled groups describe one actuator driving several joints (e.g. a
    telescoping arm). They are not independent IK degrees of freedom.
    Does not certify collision or change simulator state.
    """
    from scipy.optimize import lsq_linear
    from scipy.spatial.transform import Rotation

    target = np.asarray(target_pos, dtype=float).reshape(3)
    rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
    if (
        not np.isfinite(target).all()
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
        or min(tol_m, tol_rad, damping, step) <= 0
        or max_iters < 1
    ):
        raise ValueError("Pose IK requires a finite rigid target and positive solver parameters")
    body_id = model.body(ee_body).id
    names = list(joint_names)
    margins = dict(joint_limit_margins or {})
    if not set(margins).issubset(names) or any(not np.isfinite(v) or v < 0 for v in margins.values()):
        raise ValueError("Joint limit margins must be finite nonnegative values for controlled joints")
    qadr, dadr = joint_qpos_addrs(model, names), joint_dof_addrs(model, names)
    bounds = {}
    for name in names:
        joint = model.joint(name)
        if model.jnt_limited[joint.id]:
            lo, hi = model.jnt_range[joint.id]
            margin = margins.get(name, 0.0)
            if lo + margin >= hi - margin:
                raise ValueError("Joint limit margin leaves no usable range")
            bounds[name] = (lo + margin, hi - margin)
    groups = [list(group) for group in coupled_groups]
    grouped = [name for group in groups for name in group]
    if len(grouped) != len(set(grouped)) or not set(grouped).issubset(names) or any(not g for g in groups):
        raise ValueError("Coupled groups must contain distinct controlled joints")
    groups.extend([[name] for name in names if name not in grouped])
    coupling = np.zeros((len(names), len(groups)))
    for column, group in enumerate(groups):
        for name in group:
            coupling[names.index(name), column] = 1.0 / len(group)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    # Meters and radians have different acceptance scales. Otherwise a small
    # orientation improvement can outweigh a position error several times its
    # tolerance, despite a feasible solution inside both declared tolerances.
    weights = np.r_[np.full(3, 1.0 / tol_m), np.full(3, 1.0 / tol_rad)]
    for iteration in range(max_iters + 1):
        # IK needs transforms and motion axes, not full-scene contact solving.
        # Collision certification remains a separate mandatory path check.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        delta = target - data.body(body_id).xpos
        angular = Rotation.from_matrix(rotation @ data.body(body_id).xmat.reshape(3, 3).T).as_rotvec()
        pos_error, rot_error = float(np.linalg.norm(delta)), float(np.linalg.norm(angular))
        inside = all(lo - 1e-9 <= data.qpos[model.joint(name).qposadr[0]] <= hi + 1e-9 for name, (lo, hi) in bounds.items())
        success = pos_error <= tol_m and rot_error <= tol_rad and inside
        if success or iteration == max_iters:
            mujoco.mj_kinematics(model, data)
            return MujocoArmIkResult(success, data.qpos.copy(), pos_error, iteration, rot_error)
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        jacobian = (np.vstack((jacp[:, dadr], jacr[:, dadr])) @ coupling) * weights[:, None]
        error = np.r_[delta, angular] * weights
        # Solve bounded increments in actuator coordinates. Scaling every joint
        # by the first saturated joint freezes otherwise usable degrees of freedom.
        lower, upper = np.empty(len(groups)), np.empty(len(groups))
        for column, group in enumerate(groups):
            lower[column], upper[column] = -0.15 * len(group), 0.15 * len(group)
            for name in group:
                index = names.index(name)
                joint = model.joint(name)
                if model.jnt_limited[joint.id]:
                    lo, hi = bounds[name]
                    lower[column] = max(lower[column], (lo - data.qpos[qadr[index]]) * len(group))
                    upper[column] = min(upper[column], (hi - data.qpos[qadr[index]]) * len(group))
        if np.any(lower > upper):
            return MujocoArmIkResult(False, data.qpos.copy(), pos_error, iteration, rot_error)
        free = upper - lower > 1e-12
        delta_q = lower.copy()
        if free.any():
            fixed_error = jacobian[:, ~free] @ delta_q[~free]
            matrix = np.vstack((jacobian[:, free], damping * np.eye(int(free.sum()))))
            rhs = np.r_[step * error - fixed_error, np.zeros(int(free.sum()))]
            delta_q[free] = lsq_linear(matrix, rhs, bounds=(lower[free], upper[free])).x
        data.qpos[qadr] += coupling @ delta_q
    raise AssertionError("unreachable")


def joint_qpos_addrs(model: mujoco.MjModel, joint_names: list[str] | tuple[str, ...]) -> list[int]:
    """Return qpos addresses for named hinge/slide joints (1 DoF each)."""
    addrs: list[int] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            raise ValueError(f"joint not found: {name!r}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return addrs


def joint_dof_addrs(model: mujoco.MjModel, joint_names: list[str] | tuple[str, ...]) -> list[int]:
    """Return dof (qvel) addresses for named 1-DoF joints."""
    addrs: list[int] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            raise ValueError(f"joint not found: {name!r}")
        addrs.append(int(model.jnt_dofadr[jid]))
    return addrs


def _apply_joint_seed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str] | tuple[str, ...],
    seed_q: np.ndarray | list[float] | None,
    *,
    mode: str = "values",
) -> None:
    """Write a seed into ``data.qpos`` for ``joint_names``.

    ``mode``:
      - ``values``: use ``seed_q`` (len == joints)
      - ``midrange``: midpoint of each limited joint range
      - ``keep``: leave current qpos for these joints unchanged
    """
    qadr = joint_qpos_addrs(model, joint_names)
    if mode == "keep":
        return
    if mode == "midrange":
        for i, name in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
            if jid >= 0 and model.jnt_limited[jid]:
                lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                data.qpos[qadr[i]] = 0.5 * (lo + hi)
        return
    if seed_q is None:
        return
    qq = np.asarray(seed_q, dtype=np.float64).reshape(-1)
    if len(qq) != len(qadr):
        raise ValueError(f"seed_q length {len(qq)} != joints {len(qadr)}")
    for i, a in enumerate(qadr):
        data.qpos[a] = float(qq[i])


def solve_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_body: str,
    joint_names: list[str] | tuple[str, ...],
    target_pos: np.ndarray | list[float],
    max_iters: int = 80,
    tol_m: float = 0.01,
    damping: float = 1e-2,
    step: float = 0.5,
) -> MujocoArmIkResult:
    """Damped least-squares position IK for a subset of joints.

    Updates ``data.qpos`` in place for the controlled joints. Orientation is ignored.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(ee_body))
    if body_id < 0:
        raise ValueError(f"ee body not found: {ee_body!r}")
    qadr = joint_qpos_addrs(model, joint_names)
    dadr = joint_dof_addrs(model, joint_names)
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)
    n = len(qadr)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    err = float("inf")
    it = 0
    for it in range(int(max_iters)):
        mujoco.mj_forward(model, data)
        cur = np.asarray(data.body(body_id).xpos, dtype=np.float64).reshape(3)
        err_vec = target - cur
        err = float(np.linalg.norm(err_vec))
        if err <= float(tol_m):
            return MujocoArmIkResult(True, data.qpos.copy(), err, it + 1)

        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = jacp[:, dadr]  # 3 x n
        # Damped least squares: dq = J^T (J J^T + λ² I)^{-1} e
        lam2 = float(damping) ** 2
        A = J @ J.T + lam2 * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err_vec)
        for i in range(n):
            data.qpos[qadr[i]] = float(data.qpos[qadr[i]] + float(step) * float(dq[i]))
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_names[i]))
            if model.jnt_limited[jid]:
                lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                data.qpos[qadr[i]] = min(max(float(data.qpos[qadr[i]]), lo), hi)

    mujoco.mj_forward(model, data)
    cur = np.asarray(data.body(body_id).xpos, dtype=np.float64).reshape(3)
    err = float(np.linalg.norm(target - cur))
    return MujocoArmIkResult(err <= float(tol_m), data.qpos.copy(), err, it + 1)


def solve_position_ik_multiseed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_body: str,
    joint_names: list[str] | tuple[str, ...],
    target_pos: np.ndarray | list[float],
    seeds: list[np.ndarray | list[float] | None] | None = None,
    try_midrange: bool = True,
    max_iters: int = 120,
    tol_m: float = 0.025,
    damping: float = 1e-2,
    step: float = 0.5,
) -> MujocoArmIkResult:
    """Try several joint seeds; keep the best successful (or lowest-error) solution.

    Always restores non-controlled qpos from the caller's pre-call state. Controlled
    joints are left at the best IK solution when returning.
    """
    qadr = joint_qpos_addrs(model, joint_names)
    qpos0 = data.qpos.copy()
    seed_list: list[tuple[str, np.ndarray | list[float] | None]] = []
    # Current configuration first (caller usually synced / last-commanded).
    seed_list.append(("current", np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)))
    if seeds:
        for i, s in enumerate(seeds):
            if s is None:
                continue
            seed_list.append((f"seed{i}", s))
    if try_midrange:
        seed_list.append(("midrange", None))

    best: MujocoArmIkResult | None = None
    best_qpos = qpos0
    for label, seed in seed_list:
        data.qpos[:] = qpos0
        if label == "midrange":
            _apply_joint_seed(model, data, joint_names, None, mode="midrange")
        elif label != "current":
            _apply_joint_seed(model, data, joint_names, seed, mode="values")
        # label == current: already restored qpos0
        mujoco.mj_forward(model, data)
        result = solve_position_ik(
            model,
            data,
            ee_body=ee_body,
            joint_names=joint_names,
            target_pos=target_pos,
            max_iters=max_iters,
            tol_m=tol_m,
            damping=damping,
            step=step,
        )
        if best is None or result.pos_error_m < best.pos_error_m:
            best = result
            best_qpos = data.qpos.copy()
        if result.success:
            data.qpos[:] = best_qpos
            mujoco.mj_forward(model, data)
            return MujocoArmIkResult(True, best_qpos, result.pos_error_m, result.iterations)

    assert best is not None
    data.qpos[:] = best_qpos
    mujoco.mj_forward(model, data)
    return MujocoArmIkResult(False, best_qpos, best.pos_error_m, best.iterations)


def plan_cartesian_ik_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_body: str,
    joint_names: list[str] | tuple[str, ...],
    waypoints_xyz: list[np.ndarray] | np.ndarray,
    tol_m: float = 0.015,
    max_iters: int = 80,
) -> tuple[bool, list[np.ndarray]]:
    """IK each Cartesian waypoint; return success and per-waypoint arm joint vectors.

    Each entry in the returned list is ``len(joint_names)`` joint angles (not full qpos).
    """
    qadr = joint_qpos_addrs(model, joint_names)
    wps = [np.asarray(w, dtype=np.float64).reshape(3) for w in waypoints_xyz]
    arm_qs: list[np.ndarray] = []
    for target in wps:
        result = solve_position_ik(
            model,
            data,
            ee_body=ee_body,
            joint_names=joint_names,
            target_pos=target,
            tol_m=tol_m,
            max_iters=max_iters,
        )
        if not result.success:
            return False, arm_qs
        arm_qs.append(np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64))
    return True, arm_qs


def interpolate_arm_waypoints(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    n_steps: int = 20,
) -> list[np.ndarray]:
    """Linear joint-space interpolation from ``q_start`` to ``q_goal`` (inclusive endpoints)."""
    a = np.asarray(q_start, dtype=np.float64).reshape(-1)
    b = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"q shape mismatch {a.shape} vs {b.shape}")
    n = max(int(n_steps), 1)
    return [((1.0 - t) * a + t * b) for t in np.linspace(0.0, 1.0, n + 1)]


def pack_arm_into_actuator_dict(
    actuator_names: list[str] | tuple[str, ...],
    arm_joint_names: list[str] | tuple[str, ...],
    arm_q: np.ndarray | list[float],
    *,
    hold: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map arm joint angles into an actuator-name dict.

    rby1: ``left_arm_joint1`` → ``left_arm1``. sourccey: ``left_shoulder_pan`` → ``left_shoulder_pan_act``.
    """
    q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
    if len(q) != len(arm_joint_names):
        raise ValueError(f"arm_q length {len(q)} != joints {len(arm_joint_names)}")
    out: dict[str, float] = dict(hold or {})
    for jname, val in zip(arm_joint_names, q, strict=True):
        aname = str(jname).replace("_joint", "")
        if aname in actuator_names:
            out[aname] = float(val)
            continue
        if jname in actuator_names:
            out[str(jname)] = float(val)
            continue
        act_alias = f"{jname}_act"
        if act_alias in actuator_names:
            out[act_alias] = float(val)
    return out


def actuator_vector_from_dict(
    actuator_names: list[str] | tuple[str, ...],
    positions: dict[str, float],
    *,
    fill: float | None = None,
) -> list[float]:
    """Dense actuator vector in ``actuator_names`` order."""
    vec: list[float] = []
    for name in actuator_names:
        if name in positions:
            vec.append(float(positions[name]))
        elif fill is not None:
            vec.append(float(fill))
        else:
            raise KeyError(f"missing actuator {name!r} in positions")
    return vec
