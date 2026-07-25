# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Sim-only pick/place helpers (teleport freejoint bodies for OVMM full-task benchmark)."""

from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np


def freejoint_ancestor_body_id(model: mujoco.MjModel, body_id: int) -> int | None:
    """Walk parents until a body with a free joint is found (Molmo ``*_1_0_0`` roots)."""
    bid = int(body_id)
    while bid >= 0:
        jnt_adr = int(model.body_jntadr[bid])
        if jnt_adr >= 0 and int(model.jnt_type[jnt_adr]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return bid
        parent = int(model.body_parentid[bid])
        if parent == bid:
            break
        bid = parent
    return None


def set_free_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pos: np.ndarray | list[float],
    quat: np.ndarray | list[float] | None = None,
) -> bool:
    """Set world pose of a freejoint body, or a welded child via its freejoint ancestor.

    MolmoSpaces iTHOR objects use a freejoint root (``…_1_0_0``) with mesh children
    (``…_1_1_0``). GT placements often name the child; teleport the ancestor so the
    requested body ends near ``pos``.
    """
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
    except Exception:
        return False
    if body_id < 0:
        return False
    mujoco.mj_forward(model, data)
    free_id = freejoint_ancestor_body_id(model, body_id)
    if free_id is None:
        return False
    jnt_adr = int(model.body_jntadr[free_id])
    qpos_adr = int(model.jnt_qposadr[jnt_adr])
    target = np.asarray(pos, dtype=np.float64).reshape(3)
    if free_id == body_id:
        p = target
    else:
        # Preserve child offset from freejoint root in world frame (no rotation change).
        offset = np.asarray(data.body(body_id).xpos, dtype=np.float64).reshape(3) - np.asarray(
            data.body(free_id).xpos, dtype=np.float64
        ).reshape(3)
        p = target - offset
    data.qpos[qpos_adr : qpos_adr + 3] = p
    if quat is not None and free_id == body_id:
        q = np.asarray(quat, dtype=np.float64).reshape(4)
    else:
        q = np.array(data.qpos[qpos_adr + 3 : qpos_adr + 7], dtype=np.float64)
        if not np.isfinite(q).all() or abs(float(np.linalg.norm(q)) - 1.0) > 1e-3:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = q
    data.qvel[qpos_adr : qpos_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return True


def set_named_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    value: float,
) -> bool:
    """Set qpos of a named hinge/slide joint (cabinet doors, drawers); zeros its qvel."""
    try:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
    except Exception:
        return False
    if joint_id < 0:
        return False
    jnt_type = int(model.jnt_type[joint_id])
    if jnt_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return False
    v = float(value)
    if model.jnt_limited[joint_id]:
        lo, hi = float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1])
        v = min(max(v, lo), hi)
    data.qpos[int(model.jnt_qposadr[joint_id])] = v
    data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
    mujoco.mj_forward(model, data)
    return True


def get_named_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
) -> float | None:
    """Read back qpos of a named hinge/slide joint; ``None`` if missing or wrong type."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
    if joint_id < 0:
        return None
    jnt_type = int(model.jnt_type[joint_id])
    if jnt_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return None
    return float(data.qpos[int(model.jnt_qposadr[joint_id])])


def parse_sim_set_body_pose_action(raw: Any) -> tuple[str | None, list[float] | None, list[float] | None]:
    """Parse ``sim_set_body_pose`` recv action payload."""
    if not isinstance(raw, dict):
        return None, None, None
    body = raw.get("body")
    pos = raw.get("pos")
    if not body or pos is None:
        return None, None, None
    pos_list = [float(x) for x in np.asarray(pos, dtype=np.float64).reshape(-1)[:3]]
    quat_raw = raw.get("quat")
    quat_list = None
    if quat_raw is not None:
        quat_list = [float(x) for x in np.asarray(quat_raw, dtype=np.float64).reshape(-1)[:4]]
    return str(body), pos_list, quat_list


def robot_zmq_set_body_pose(
    robot: Any,
    body: str,
    pos: np.ndarray | list[float],
    *,
    quat: np.ndarray | list[float] | None = None,
) -> None:
    """Send ``sim_set_body_pose`` on the robot ZMQ client (OVMM full + dynamic exploration)."""
    from emet.core.zmq_protocol import build_sim_set_body_pose_action

    p = np.asarray(pos, dtype=np.float64).reshape(3)
    step = int(getattr(robot, "_last_step", -1)) + 1
    if step < 1:
        step = 1
    quat_arg = None
    if quat is not None:
        quat_arg = [float(x) for x in np.asarray(quat, dtype=np.float64).reshape(4)]
    action = build_sim_set_body_pose_action(step, body, p.tolist(), quat=quat_arg)
    _send_meta_action(robot, action)


def parse_sim_set_joint_qpos_action(raw: Any) -> tuple[str | None, float | None]:
    """Parse ``sim_set_joint_qpos`` recv action payload."""
    if not isinstance(raw, dict):
        return None, None
    joint = raw.get("joint")
    value = raw.get("value")
    if not joint or value is None:
        return None, None
    return str(joint), float(value)


def robot_zmq_set_joint_qpos(robot: Any, joint: str, value: float) -> None:
    """Send ``sim_set_joint_qpos`` on the robot ZMQ client (doors/drawers in dynamic benchmarks)."""
    from emet.core.zmq_protocol import build_sim_set_joint_qpos_action

    step = int(getattr(robot, "_last_step", -1)) + 1
    if step < 1:
        step = 1
    action = build_sim_set_joint_qpos_action(step, joint, float(value))
    _send_meta_action(robot, action)


def _send_meta_action(robot: Any, action: dict[str, Any]) -> None:
    """Send a meta action reliably and give the sim a beat to apply it."""
    send_action = getattr(robot, "send_action", None)
    if callable(send_action):
        send_action(action, reliable=True)
    else:
        robot.send_message(action)
    wait_obs = getattr(robot, "wait_for_obs", None)
    if callable(wait_obs):
        wait_obs(timeout=5.0)
    time.sleep(0.25)


def robot_sim_body_pose_teleport_supported(robot: Any) -> bool:
    """True when the ZMQ session is simulation and advertises ``sim_set_body_pose``."""
    get_sess = getattr(robot, "get_emet_session", None)
    if not callable(get_sess):
        return False
    session = get_sess()
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return False
    caps = session.get("capabilities") or {}
    return bool(caps.get("sim_set_body_pose", False))


def prefer_sim_teleport_manip(robot: Any, *, visual_servo: bool = False) -> bool:
    """Use GT body teleport for pick/place instead of Stretch visual-servo / AnyGrasp.

    Stretch MuJoCo also advertises ``sim_set_body_pose``; keep visual-servo when enabled.
    """
    if visual_servo:
        return False
    return robot_sim_body_pose_teleport_supported(robot)


def resolve_agent_manip_mode(
    *,
    config_mode: str | None = None,
    visual_servo: bool = False,
) -> str:
    """Resolve ``teleport`` | ``kinematic`` from env then config (default teleport)."""
    if visual_servo:
        return "stretch_visual_servo"
    from emet.simulation.env_flags import env_manip_mode

    env_m = env_manip_mode()
    if env_m:
        return env_m
    mode = str(config_mode or "teleport").strip().lower()
    if mode in ("teleport", "kinematic"):
        return mode
    return "teleport"


def resolve_agent_manip_collision(*, config_mode: str | None = None) -> str:
    from emet.simulation.env_flags import env_manip_collision

    env_c = env_manip_collision()
    if env_c:
        return env_c
    mode = str(config_mode or "none").strip().lower()
    if mode in ("none", "voxel"):
        return mode
    return "none"


def prefer_kinematic_manip(
    robot: Any,
    *,
    manip_mode: str,
    visual_servo: bool = False,
) -> bool:
    """True when agent should run IK + attach pick/place (not teleport / visual-servo).

    Requires both the server ``kinematic_manip`` capability **and** a registered
    :class:`~emet.motion.arm_manip_profile.ArmManipProfile` for the robot id.
    Robosuite may advertise the cap broadly; profile gating prevents KeyError on
    innate_mars / xlerobot / stretch.
    """
    if visual_servo or str(manip_mode).lower() != "kinematic":
        return False
    get_sess = getattr(robot, "get_emet_session", None)
    if not callable(get_sess):
        return False
    session = get_sess()
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return False
    caps = session.get("capabilities") or {}
    if not caps.get("kinematic_manip", False):
        return False
    from emet.core.zmq_protocol import EMET_ZMQ_ROBOT_ID_KEY, read_emet_robot_id_from_message_or_session
    from emet.motion.arm_manip_profile import has_arm_manip_profile, robot_id_from_client

    rid = None
    try:
        rid = robot_id_from_client(robot)
    except ValueError:
        rid = read_emet_robot_id_from_message_or_session(session) or session.get(EMET_ZMQ_ROBOT_ID_KEY)
    if not rid or not has_arm_manip_profile(str(rid)):
        return False
    return True


def can_use_sim_gt_manip(
    robot: Any,
    *,
    manip_mode: str = "teleport",
    visual_servo: bool = False,
) -> bool:
    """True when pick/place can run from session GT (kinematic or teleport) without visual nav.

    If ``manip_mode=kinematic`` but the server lacks ``kinematic_manip``, still True when
    teleport (``sim_set_body_pose``) is available so the agent can fall back.
    """
    if prefer_kinematic_manip(robot, manip_mode=manip_mode, visual_servo=visual_servo):
        return True
    return prefer_sim_teleport_manip(robot, visual_servo=visual_servo)


def _read_robot_placements(robot: Any) -> dict[str, dict[str, Any]] | None:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    get_sess = getattr(robot, "get_emet_session", None)
    session = get_sess() if callable(get_sess) else None
    return read_sim_object_placements(session)


def resolve_sim_object_body(
    robot: Any,
    object_query: str,
    *,
    start_recep: str | None = None,
    object_gt_body: str | None = None,
) -> str | None:
    """Resolve a GT body name for ``object_query`` from the live sim session placements."""
    from emet.eval.ovmm_find_phase import pick_find_object_gt_body

    placements = _read_robot_placements(robot)
    if not placements:
        return None
    return pick_find_object_gt_body(
        placements,
        object_query,
        start_recep or "",
        object_gt_body=object_gt_body,
    )


def _verify_body_near(
    robot: Any,
    body: str,
    target: np.ndarray,
    *,
    tol_m: float = 0.05,
) -> bool:
    placements = _read_robot_placements(robot)
    if not placements or body not in placements:
        return False
    pos = np.asarray(placements[body]["pos"], dtype=np.float64).reshape(3)
    return float(np.linalg.norm(pos - np.asarray(target, dtype=np.float64).reshape(3))) <= float(tol_m)


def sim_teleport_pickup(
    robot: Any,
    target_object: str,
    *,
    lift_m: float = 0.12,
    object_gt_body: str | None = None,
    verify: bool = True,
) -> str | None:
    """Teleport object body up by ``lift_m`` (sim pick proxy). Returns body name or None."""
    if not robot_sim_body_pose_teleport_supported(robot):
        return None
    body = object_gt_body or resolve_sim_object_body(robot, target_object)
    placements = _read_robot_placements(robot)
    if not body or not placements or body not in placements:
        return None
    pos = np.asarray(placements[body]["pos"], dtype=np.float64).reshape(3).copy()
    pos[2] += float(lift_m)
    robot_zmq_set_body_pose(robot, body, pos)
    if verify and not _verify_body_near(robot, body, pos):
        return None
    return body


def sim_teleport_place(
    robot: Any,
    target_receptacle: str,
    *,
    object_gt_body: str | None = None,
    object_query: str | None = None,
    place_z_offset_m: float = 0.02,
    verify: bool = True,
) -> bool:
    """Teleport held/object body onto a receptacle (sim place proxy)."""
    from emet.eval.ovmm_find_phase import bodies_matching_category

    if not robot_sim_body_pose_teleport_supported(robot):
        return False
    placements = _read_robot_placements(robot)
    if not placements:
        return False
    body = object_gt_body
    if not body and object_query:
        body = resolve_sim_object_body(robot, object_query)
    if not body or body not in placements:
        return False
    recep_bodies = bodies_matching_category(placements, target_receptacle)
    if not recep_bodies:
        return False
    anchor = np.asarray(placements[recep_bodies[0]]["pos"], dtype=np.float64).reshape(3).copy()
    anchor[2] += float(place_z_offset_m)
    robot_zmq_set_body_pose(robot, body, anchor)
    if verify and not _verify_body_near(robot, body, anchor):
        return False
    return True


def parse_sim_attach_body_action(raw: Any) -> tuple[str | None, str | None, list[float] | None]:
    if not isinstance(raw, dict):
        return None, None, None
    body = raw.get("body")
    ee = raw.get("ee_body")
    if not body or not ee:
        return None, None, None
    off = raw.get("offset_pos")
    offset = None
    if off is not None:
        offset = [float(x) for x in np.asarray(off, dtype=np.float64).reshape(-1)[:3]]
    return str(body), str(ee), offset


def parse_sim_detach_body_action(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    body = raw.get("body")
    return str(body) if body else None


def robot_zmq_attach_body(
    robot: Any,
    body: str,
    ee_body: str,
    *,
    offset_pos: np.ndarray | list[float] | None = None,
) -> None:
    from emet.core.zmq_protocol import build_sim_attach_body_action

    step = int(getattr(robot, "_last_step", -1)) + 1
    if step < 1:
        step = 1
    off = None
    if offset_pos is not None:
        off = [float(x) for x in np.asarray(offset_pos, dtype=np.float64).reshape(3)]
    action = build_sim_attach_body_action(step, body, ee_body, offset_pos=off)
    _send_meta_action(robot, action)


def robot_zmq_detach_body(robot: Any, body: str | None = None) -> None:
    from emet.core.zmq_protocol import build_sim_detach_body_action

    step = int(getattr(robot, "_last_step", -1)) + 1
    if step < 1:
        step = 1
    action = build_sim_detach_body_action(step, body)
    _send_meta_action(robot, action)


def sim_teleport_to_grasp_pose(
    robot: Any,
    body: str,
    grasp_xyz_world: np.ndarray | list[float],
    *,
    lift_m: float = 0.12,
    verify: bool = True,
) -> bool:
    """Teleport object to grasp XYZ then lift (oracle-driven teleport pick for Stretch etc.)."""
    if not robot_sim_body_pose_teleport_supported(robot):
        return False
    target = np.asarray(grasp_xyz_world, dtype=np.float64).reshape(3).copy()
    robot_zmq_set_body_pose(robot, body, target)
    if verify and not _verify_body_near(robot, body, target, tol_m=0.08):
        return False
    lifted = target.copy()
    lifted[2] += float(lift_m)
    robot_zmq_set_body_pose(robot, body, lifted)
    if verify and not _verify_body_near(robot, body, lifted, tol_m=0.08):
        return False
    return True
