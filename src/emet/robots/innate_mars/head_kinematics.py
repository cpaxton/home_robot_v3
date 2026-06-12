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

"""Innate Mars head nod: ROS / TF ↔ MJCF ``joint_head`` helpers (hardware vs sim)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY, read_emet_session
from emet.robots.innate_mars import INNATE_MARS_JOINT_NAMES, InnateMarsBackend
from emet.simulation.mujoco_gt_objects import camera_pose_world_opencv
from emet.utils.geometry import nav_xyt_to_world_xyt

# Maurice URDF lists head nod about **−Y**; MJCF sim hinge is **+X** (see innate_mars.xml comment).
MJCF_JOINT_HEAD_LIMITS = (-0.1745, 0.5236)

# ROS / REP-103 optical frame (+Z forward, +X right, +Y down) in MJCF head-body frame (wxyz).
ROS_OPTICAL_QUAT_WXYZ = (0.5, -0.5, -0.5, 0.5)

# Sim ``innate_mars.xml`` visual + camera convention: table-forward **−base Y**; hardware ROS /
# TF uses **+base X** forward. Rerun MJCF meshes need this fixed yaw in ``base_link`` (cameras
# are patched via :func:`patch_innate_mars_head_cameras_for_hardware`).
HARDWARE_MJCF_VISUAL_YAW_RAD = float(np.pi / 2.0)

# Sim ``head_visual`` places the STL at the neck pivot; hardware URDF cameras sit ~70 mm
# forward (+head X). Shift the visual subtree so Rerun head mesh aligns with stereo TF (Herman 2026-06).
HARDWARE_HEAD_VISUAL_POS: tuple[float, float, float] = (0.07, 0.01, -0.005)

# Sim ``innate_mars.xml`` applies a table-forward Rz hack on stereo mounts; hardware TF uses
# maurice.urdf optical frames. Mounts are in the MJCF ``head`` body frame, calibrated against
# Herman TF in **base_link** frame (2026-06, ~0.5 mm vs ZMQ ``camera_pose*``).
HARDWARE_HEAD_CAMERA_MOUNTS: dict[str, dict[str, tuple[float, float, float]]] = {
    "head_left": {
        "pos": (0.043, 0.03, 0.0),
        "quat_wxyz": ROS_OPTICAL_QUAT_WXYZ,
    },
    "head_right": {
        "pos": (0.0435, -0.03, 0.0),
        "quat_wxyz": ROS_OPTICAL_QUAT_WXYZ,
    },
}


def is_hardware_innate_mars_obs(obs: dict[str, Any]) -> bool:
    """True when ZMQ session indicates live innate Mars (not MuJoCo sim)."""
    sess = obs.get(EMET_ZMQ_SESSION_KEY)
    if isinstance(sess, dict) and sess.get("is_simulation") is False:
        return True
    if obs.get("is_simulation") is False:
        return True
    sess = sess if isinstance(sess, dict) else {}
    return sess.get("runtime_kind") == "innate_mars_ros2_bridge"


def patch_innate_mars_head_cameras_for_hardware(model: Any) -> bool:
    """Replace sim table-forward stereo mounts with hardware TF mounts on *model* (in place)."""
    import mujoco

    patched = False
    for cam_name, spec in HARDWARE_HEAD_CAMERA_MOUNTS.items():
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            continue
        model.cam_pos[cid] = np.asarray(spec["pos"], dtype=np.float64)
        model.cam_quat[cid] = np.asarray(spec["quat_wxyz"], dtype=np.float64)
        patched = True
    return patched


def patch_innate_mars_head_visual_for_hardware(model: Any) -> bool:
    """Shift ``head_visual`` so the head STL encloses hardware stereo cameras (not sim neck pivot)."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_visual")
    if bid < 0:
        return False
    model.body_pos[bid] = np.asarray(HARDWARE_HEAD_VISUAL_POS, dtype=np.float64)
    return True


def patch_innate_mars_model_for_hardware_replay(model: Any) -> bool:
    """Apply all hardware MJCF display patches (cameras + head visual)."""
    c = patch_innate_mars_head_cameras_for_hardware(model)
    v = patch_innate_mars_head_visual_for_hardware(model)
    return c or v


def _planar_xyt_to_matrix(xyt: np.ndarray) -> np.ndarray:
    """4×4 world pose for planar base (x, y, yaw); z translation 0."""
    x, y, theta = (float(xyt[0]), float(xyt[1]), float(xyt[2])) if xyt.size >= 3 else (0.0, 0.0, 0.0)
    c, s = np.cos(theta), np.sin(theta)
    t = np.eye(4, dtype=np.float64)
    t[0, 0], t[0, 1] = c, -s
    t[1, 0], t[1, 1] = s, c
    t[0, 3], t[1, 3] = x, y
    return t


def camera_pose_in_base_link(
    gps: np.ndarray | list[float],
    compass: np.ndarray | list[float],
    camera_pose_world: np.ndarray,
    *,
    session: dict[str, Any] | None = None,
) -> np.ndarray:
    """OpenCV camera-to-base from ``gps``/``compass``, absolute ``camera_pose_world``, and session.

    Sim ZMQ uses episode-relative ``gps``/``compass`` plus ``navigation_origin_xyt`` while
    ``camera_pose`` is absolute MuJoCo world — same convention as Rerun ``log_robot_xyt``.
    """
    xy = np.asarray(gps, dtype=np.float64).reshape(-1)[:2]
    comp = np.asarray(compass, dtype=np.float64).ravel()
    theta = float(comp[0]) if comp.size else 0.0
    local = np.array([float(xy[0]), float(xy[1]), theta], dtype=np.float64)
    world_xyt = nav_xyt_to_world_xyt(local, session)
    t_base = _planar_xyt_to_matrix(world_xyt)
    t_cam = np.asarray(camera_pose_world, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(t_base) @ t_cam


def camera_pose_in_base_link_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """``camera_pose_in_base_link`` from a full ZMQ observation dict."""
    return camera_pose_in_base_link(
        obs["gps"],
        obs["compass"],
        obs["camera_pose"],
        session=read_emet_session(obs),
    )


def ros_head_deg_to_mjcf_rad(deg: float) -> float:
    """Convert ``/mars/head/current_position`` degrees to MJCF ``joint_head`` radians."""
    return float(-np.deg2rad(float(deg)))


def head_hinge_rad_from_base_head_tf(T_base_head: np.ndarray) -> float:
    """Extract head nod (MJCF ``joint_head``) from ``base_link`` → ``head`` transform."""
    r = np.asarray(T_base_head, dtype=np.float64).reshape(4, 4)[:3, :3]
    # URDF hinge −Y; MJCF replay uses +X with opposite sign.
    theta_urdf = float(np.arctan2(-r[0, 2], r[0, 0]))
    return float(-theta_urdf)


def opencv_gaze_error_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Angle between OpenCV +Z gaze directions of two camera-to-base poses (degrees)."""
    ga = np.asarray(T_a, dtype=np.float64).reshape(4, 4)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    gb = np.asarray(T_b, dtype=np.float64).reshape(4, 4)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    na, nb = float(np.linalg.norm(ga)), float(np.linalg.norm(gb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos = float(np.clip(np.dot(ga, gb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _mjcf_camera_in_base(
    joint: np.ndarray,
    joint_head: float,
    *,
    gps: np.ndarray | list[float] | None = None,
    compass: np.ndarray | list[float] | None = None,
    session: dict[str, Any] | None = None,
    mjcf_path: str | None = None,
    use_hardware_cameras: bool = False,
) -> np.ndarray:
    import mujoco

    from emet.visualization.mjcf_rerun_robot import _body_T_world, apply_zmq_obs_to_mujoco_data

    spec = InnateMarsBackend().get_spec()
    path = mjcf_path or spec.mjcf_path
    model = mujoco.MjModel.from_xml_path(path)
    if use_hardware_cameras:
        patch_innate_mars_head_cameras_for_hardware(model)
    data = mujoco.MjData(model)
    gps_v = np.zeros(2, dtype=np.float64) if gps is None else np.asarray(gps, dtype=np.float64).reshape(-1)[:2]
    comp_v = np.zeros(1, dtype=np.float64) if compass is None else np.asarray(compass, dtype=np.float64).ravel()[:1]
    obs_pose: dict[str, Any] = {
        "gps": gps_v,
        "compass": comp_v,
        "joint": np.asarray(joint, dtype=np.float64).reshape(-1),
        "joint_head": float(joint_head),
    }
    if session is not None:
        obs_pose[EMET_ZMQ_SESSION_KEY] = session
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs_pose,
        joint_names=tuple(INNATE_MARS_JOINT_NAMES),
        dof=spec.dof,
        base_link_name=spec.base_link_name,
        nav_origin_slot=[None],
        free_qadr=None,
    )
    mujoco.mj_forward(model, data)
    t_world = camera_pose_world_opencv(model, data, "head_left")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link_name)
    t_base = _body_T_world(data, int(root))
    return np.linalg.inv(t_base) @ t_world


def infer_joint_head_from_obs_on_model(
    model: Any,
    obs: dict[str, Any],
    *,
    n_samples: int = 21,
) -> float:
    """Fast ``joint_head`` search on an already-loaded (optionally hardware-patched) MJCF model."""
    import mujoco

    from emet.visualization.mjcf_rerun_robot import _body_T_world, apply_zmq_obs_to_mujoco_data

    spec = InnateMarsBackend().get_spec()
    data = mujoco.MjData(model)
    sess = read_emet_session(obs)
    target = camera_pose_in_base_link(obs["gps"], obs["compass"], obs["camera_pose"], session=sess)
    joint = np.asarray(obs["joint"], dtype=np.float64).reshape(-1)
    lo, hi = MJCF_JOINT_HEAD_LIMITS
    angles = np.linspace(lo, hi, max(3, int(n_samples)), dtype=np.float64)
    best_ang = float(lo)
    best_err = float("inf")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link_name)
    obs_base = {
        "gps": obs["gps"],
        "compass": obs["compass"],
        "joint": joint,
        EMET_ZMQ_SESSION_KEY: sess,
    }
    for ang in angles:
        pose = {**obs_base, "joint_head": float(ang)}
        apply_zmq_obs_to_mujoco_data(
            model,
            data,
            pose,
            joint_names=tuple(INNATE_MARS_JOINT_NAMES),
            dof=spec.dof,
            base_link_name=spec.base_link_name,
            nav_origin_slot=[None],
            free_qadr=None,
        )
        mujoco.mj_forward(model, data)
        t_world = camera_pose_world_opencv(model, data, "head_left")
        t_base = np.linalg.inv(_body_T_world(data, int(root))) @ t_world
        err = float(np.linalg.norm(t_base[:3, 3] - target[:3, 3]))
        if err < best_err:
            best_err = err
            best_ang = float(ang)
    return best_ang


def obs_pose_for_base_relative_mjcf_replay(obs_pose: dict[str, Any]) -> dict[str, Any]:
    """Copy *obs_pose* with planar base joints zeroed (``world/robot`` carries global pose)."""
    out = dict(obs_pose)
    jraw = out.get("joint_mjcf")
    if jraw is None:
        jraw = out.get("joint")
    if jraw is None:
        return out
    j = np.asarray(jraw, dtype=np.float64).reshape(-1).copy()
    if j.size >= 3:
        j[:3] = 0.0
    out["joint"] = j
    if "joint_mjcf" in out:
        out["joint_mjcf"] = j.copy()
    return out


def enrich_obs_pose_joint_head_for_hardware_replay(
    model: Any,
    obs_pose: dict[str, Any],
    *,
    n_samples: int = 21,
) -> dict[str, Any]:
    """Return *obs_pose* with ``joint_head`` from ``camera_pose`` for hardware Rerun replay."""
    if not is_hardware_innate_mars_obs(obs_pose):
        return obs_pose
    if obs_pose.get("camera_pose") is None:
        return obs_pose
    inferred = infer_joint_head_from_obs_on_model(model, obs_pose, n_samples=n_samples)
    enriched = dict(obs_pose)
    enriched["joint_head"] = inferred
    return enriched


def infer_joint_head_from_camera_pose(
    joint: np.ndarray,
    camera_pose_world: np.ndarray,
    *,
    gps: np.ndarray | None = None,
    compass: np.ndarray | None = None,
    session: dict[str, Any] | None = None,
    n_samples: int = 41,
    use_hardware_cameras: bool = False,
) -> float:
    """1-D search: MJCF ``joint_head`` that best matches ZMQ ``camera_pose`` (position in base frame)."""
    gps_v = np.zeros(2, dtype=np.float64) if gps is None else np.asarray(gps, dtype=np.float64).reshape(-1)[:2]
    comp_v = np.zeros(1, dtype=np.float64) if compass is None else np.asarray(compass, dtype=np.float64).ravel()[:1]
    target = camera_pose_in_base_link(gps_v, comp_v, camera_pose_world, session=session)
    lo, hi = MJCF_JOINT_HEAD_LIMITS
    angles = np.linspace(lo, hi, max(3, int(n_samples)), dtype=np.float64)
    best_ang = float(lo)
    best_err = float("inf")
    for ang in angles:
        t_pred = _mjcf_camera_in_base(
            joint,
            float(ang),
            gps=gps_v,
            compass=comp_v,
            session=session,
            use_hardware_cameras=use_hardware_cameras,
        )
        err = float(np.linalg.norm(t_pred[:3, 3] - target[:3, 3]))
        if err < best_err:
            best_err = err
            best_ang = float(ang)
    return best_ang


def compare_mjcf_camera_to_zmq(
    obs: dict,
    *,
    joint_head: float | None = None,
    use_hardware_cameras: bool | None = None,
) -> dict[str, float]:
    """Return position/gaze errors between MJCF FK and ZMQ ``camera_pose`` (base frame)."""
    hw = is_hardware_innate_mars_obs(obs) if use_hardware_cameras is None else bool(use_hardware_cameras)
    sess = read_emet_session(obs)
    jhead = float(obs.get("joint_head") or 0.0) if joint_head is None else float(joint_head)
    joint = np.asarray(obs["joint"], dtype=np.float64).reshape(-1)
    target = camera_pose_in_base_link(obs["gps"], obs["compass"], obs["camera_pose"], session=sess)
    t_mj = _mjcf_camera_in_base(
        joint,
        jhead,
        gps=obs["gps"],
        compass=obs["compass"],
        session=sess,
        use_hardware_cameras=hw,
    )
    pos_err = float(np.linalg.norm(t_mj[:3, 3] - target[:3, 3]))
    gaze_err_deg = opencv_gaze_error_deg(t_mj, target)
    r_err = t_mj[:3, :3].T @ target[:3, :3]
    rot_err_deg = float(np.degrees(np.arccos(np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0))))
    return {
        "joint_head_rad": jhead,
        "pos_err_m": pos_err,
        "gaze_err_deg": gaze_err_deg,
        "rot_err_deg": rot_err_deg,
        "use_hardware_cameras": float(hw),
    }
