# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""ZMQ server that wraps a robosuite environment for non-Stretch robots.

For robots natively supported by robosuite (PandaOmron, Tiago, GR1, etc.)
this server keeps the robosuite robot in the scene and exposes the same
ZMQ protocol as MujocoZmqServer.
"""

import math
import os
import threading
import time
from pathlib import Path
from typing import Any, cast

import cv2
import mujoco
import numpy as np
from overrides import override

import emet.utils.compression as compression
import emet.utils.logger as log
from emet.core.server import BaseZmqServer
from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY,
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
)
from emet.robots.base import RobotSpec
from emet.simulation import molmospaces_spawn, scene_base_spawn
from emet.simulation.env_flags import env_sim_nav_debug, warn_sim_nav_env_flags
from emet.simulation.head_look_action import apply_head_to_robosuite, apply_stretch_posture_to_robosuite
from emet.simulation.molmospaces_env import molmospaces_nav_teleport_enabled
from emet.simulation.molmospaces_mobile_autoplace import apply_molmospaces_freejoint_base_autoplace
from emet.simulation.mujoco_ground_truth import (
    mujoco_ground_truth_write_path,
    parse_ground_truth_dump_action_field,
)
from emet.simulation.mujoco_stationary_control import (
    DefaultMujocoStationaryControl,
    MujocoStationaryControl,
    compute_stationary_ctrl_vector,
)
from emet.simulation.robosuite_load_utils import (
    apply_home_keyframe_preserving_base,
    apply_home_keyframe_preserving_planar_base,
    update_robot_qpos0_from_data,
    log_post_load_diagnostics,
    probe_max_qvel_unforced_steps,
    robosuite_post_load_debug_enabled,
)
from emet.simulation.stereo_camera_utils import stereo_right_camera_name_from_spec
from emet.utils.geometry import xyt_global_to_base
from emet.utils.observation_layout import rgb_height_width_for_zmq
from emet.utils.pinhole_intrinsics import apply_pinhole_pixel_ops, chain_pinhole_K_pixel_ops, scale_pinhole_K

logger = log.Logger(__name__)

# One ``mujoco.Renderer`` / GL context: multiple resolutions each call ``mjr_makeContext`` and often
# hit GL_INVALID_OPERATION (0x502) on EGL. Primary + servo reuse one renderer; servo resizes in CPU.
_PRIMARY_RW, _PRIMARY_RH = 640, 480
_SERVO_RW, _SERVO_RH = 320, 240


class RobosuiteZmqServer(BaseZmqServer):
    """ZMQ server backed by a robosuite environment.

    Unlike MujocoZmqServer (which uses stretch_mujoco), this server
    drives the robosuite env directly via ``env.step(action)``.
    """

    hz = 20

    def __init__(
        self,
        robot_spec: RobotSpec,
        *args,
        scene_xml: str | None = None,
        scene_model: mujoco.MjModel | None = None,
        simulation_rate: int = 80,
        environment: dict[str, Any] | None = None,
        scene_source_basename: str | None = None,
        session_extra: dict[str, Any] | None = None,
        **kwargs,
    ):
        max_sim_steps = kwargs.pop("max_sim_steps", None)
        debug_molmospaces_spawn = bool(kwargs.pop("debug_molmospaces_spawn", False))
        scene_disk_path = kwargs.pop("scene_disk_path", None)
        mujoco_stationary_control: MujocoStationaryControl | None = kwargs.pop(
            "mujoco_stationary_control", None
        )
        super().__init__(*args, **kwargs)
        self._spec = robot_spec
        self._scene_xml = scene_xml
        self._scene_model = scene_model
        self.simulation_rate = simulation_rate
        self._environment_descriptor = dict(environment) if environment else None
        self._scene_source_basename = scene_source_basename
        self._session_extra = dict(session_extra) if session_extra else None

        self._mjmodel: mujoco.MjModel | None = None
        self._mjdata: mujoco.MjData | None = None
        self._mj_lock = threading.RLock()
        self._initial_xyt: np.ndarray | None = None
        self._running = False
        self.control_mode = "navigation"
        self._at_goal = False
        self._emet_session: dict[str, Any] | None = None
        # World-frame (x, y, yaw) holonomic drive goal (velocity before mj_step): free joint or planar joints.
        self._nav_goal_world: np.ndarray | None = None
        self._nav_tol_xy = 0.07
        self._nav_tol_theta = 0.15
        self._nav_kp_xy = 0.95
        self._nav_kp_theta = 2.2
        self._nav_v_max = 0.42
        self._nav_w_max = 0.95
        self._render_lock = threading.Lock()
        self._primary_renderer: Any | None = None
        self._last_full_obs_for_servo: dict[str, Any] | None = None
        self._max_sim_steps: int | None = (
            int(max_sim_steps) if max_sim_steps is not None and int(max_sim_steps) > 0 else None
        )
        self._debug_molmospaces_spawn = debug_molmospaces_spawn
        self._scene_disk_path: str | None = (
            str(scene_disk_path).strip() if scene_disk_path and str(scene_disk_path).strip() else None
        )
        self._physics_steps_executed = 0
        # After MolmoSpaces autoplace, ``qpos0`` holds the chosen free-joint pose; see
        # :meth:`_restore_merged_base_freejoint_from_qpos0` after physics stabilize.
        self._molmospaces_autoplace_snap_qpos0 = False
        self._planar_autoplace_snap_qpos0 = False
        self._planar_autoplace_world_xyt: np.ndarray | None = None
        self._planar_base_actuator_ids_cache: tuple[int, int, int] | None = None
        # Robocasa walkable AABB (xmin, xmax, ymin, ymax); clamps teleport / velocity nav goals.
        self._nav_world_clip_rect: tuple[float, float, float, float] | None = None
        self._nav_drive_debug_ticks = 0
        # After spawn / resettle, lock the base free joint while idle (no nav goal) so gravity does not
        # drop a floating base through the floor (wheels are visual-only on Galaxea / rby1).
        self._stationary_base_freejoint_qpos: np.ndarray | None = None
        # Stationary joint-position targets mirrored each physics step onto ``data.ctrl`` (see
        # :meth:`_apply_joint_ctrl_hold_to_actuators`). Same idea as MolmoSpaces ``set_to_stationary``
        # + ``compute_control`` (see comments on that method).
        self._joint_ctrl_hold: np.ndarray | None = None
        # When False, :meth:`_refresh_unpinned_joint_ctrl_hold_from_stationary` copies that spec row
        # from the transmission stationary vector (current ``q``) into ``_joint_ctrl_hold``.
        # ZMQ ``joint`` actions set True so streamed / one-shot targets are not overwritten on refresh.
        self._joint_ctrl_hold_client_pin: np.ndarray | None = None
        self._mujoco_stationary: MujocoStationaryControl = (
            mujoco_stationary_control
            if mujoco_stationary_control is not None
            else DefaultMujocoStationaryControl()
        )
        # ``mj_step`` calls per outer server tick (see :meth:`_configure_mj_substeps_per_tick`).
        self._mj_substeps_per_tick: int = 1
        # When ``EMET_MUJOCO_CTRL_DEBUG=1``: log first N stationary apply cycles, then periodic summaries.
        self._ctrl_debug_initial_logs_remaining: int = 0
        self._ctrl_debug_periodic_counter: int = 0
        self._ctrl_debug_emit_apply_logs: bool = False

    @staticmethod
    def _mujoco_ctrl_debug_enabled() -> bool:
        return os.environ.get("EMET_MUJOCO_CTRL_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _mujoco_ctrl_debug_verbose() -> bool:
        return os.environ.get("EMET_MUJOCO_CTRL_DEBUG_VERBOSE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _mujoco_ctrl_debug_pd_tracking(self) -> str:
        """Torso / limb diagnostics (``Δq0`` vs MJCF ``qpos0``, ``dq``, actuator / constraint torques).

        In the realtime sim loop, unpinned :attr:`_joint_ctrl_hold` rows are **not** overwritten from
        ``q`` each tick (that would retarget PD every frame and kill restoring torque). They advance
        only via :meth:`_sync_actuator_ctrl_from_joint_positions`, ZMQ ``joint`` (pinned rows), or
        :meth:`_snapshot_spec_hold_from_ctrl`. ``ctrl`` is fixed across substeps; ``q - ctrl`` shows
        tracking error for unpinned joints.
        """
        if self._mjmodel is None or self._mjdata is None:
            return ""
        m, d = self._mjmodel, self._mjdata
        hold = self._joint_ctrl_hold

        def one_joint(jn: str, an: str, spec_i: int | None) -> str | None:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
            if jid < 0 or aid < 0:
                return None
            jt = int(m.jnt_type[jid])
            if jt not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                return None
            qadr = int(m.jnt_qposadr[jid])
            vadr = int(m.jnt_dofadr[jid])
            q = float(d.qpos[qadr])
            q0 = float(m.qpos0[qadr])
            dq = float(d.qvel[vadr]) if 0 <= vadr < int(m.nv) else float("nan")
            c = float(d.ctrl[aid])
            d_q0 = q - q0
            h = (
                float(hold[spec_i])
                if spec_i is not None and hold is not None and 0 <= spec_i < int(hold.shape[0])
                else float("nan")
            )
            tau_dof = (
                float(d.qfrc_actuator[vadr])
                if 0 <= vadr < int(d.qfrc_actuator.shape[0])
                else float("nan")
            )
            tau_c = (
                float(d.qfrc_constraint[vadr])
                if 0 <= vadr < int(d.qfrc_constraint.shape[0])
                else float("nan")
            )
            f_act = float(d.actuator_force[aid]) if 0 <= aid < int(d.actuator_force.shape[0]) else float("nan")
            return (
                f"{an}({jn}):q={q:.4f} q0={q0:.4f} Δq0={d_q0:.4f} dq={dq:.4f} "
                f"ctrl={c:.4f} hold={h:.4f} F_act={f_act:.3f} τ_dof={tau_dof:.3f} τ_con={tau_c:.3f}"
            )

        # Resolve spec index for torso1 / torso_joint1
        hip_spec_i: int | None = None
        n = min(len(self._spec.actuator_names), len(self._spec.joint_names))
        for i in range(n):
            if self._spec.actuator_names[i] == "torso1" and self._spec.joint_names[i] == "torso_joint1":
                hip_spec_i = i
                break

        hip_line = one_joint("torso_joint1", "torso1", hip_spec_i)
        if hip_line is None:
            hip_line = "torso_hip: (torso_joint1/torso1 not found in model)"

        max_abs_dq0 = 0.0
        max_abs_dq = 0.0
        torso_chunks: list[str] = []
        for i in range(n):
            an = self._spec.actuator_names[i]
            if not an.startswith("torso"):
                continue
            jn = self._spec.joint_names[i]
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                continue
            jt = int(m.jnt_type[jid])
            if jt not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                continue
            qadr = int(m.jnt_qposadr[jid])
            vadr = int(m.jnt_dofadr[jid])
            q = float(d.qpos[qadr])
            q0 = float(m.qpos0[qadr])
            dq = float(d.qvel[vadr]) if 0 <= vadr < int(m.nv) else 0.0
            max_abs_dq0 = max(max_abs_dq0, abs(q - q0))
            max_abs_dq = max(max_abs_dq, abs(dq))
            if self._mujoco_ctrl_debug_verbose():
                ol = one_joint(jn, an, i)
                if ol:
                    torso_chunks.append(ol)
        if self._mujoco_ctrl_debug_verbose():
            for i in range(n):
                an = self._spec.actuator_names[i]
                if an not in ("left_arm1", "right_arm1"):
                    continue
                jn = self._spec.joint_names[i]
                ol = one_joint(jn, an, i)
                if ol:
                    torso_chunks.append(ol)

        head = (
            f"pd torso_agg max|Δq0|={max_abs_dq0:.4f} max|dq|={max_abs_dq:.4f} "
            f"(unpinned hold fixed until sync / joint / head_to; Δq0 vs qpos0 = keyframe drift) | hip: {hip_line}"
        )
        if torso_chunks:
            return head + " | " + " ; ".join(torso_chunks)
        return head

    def _mujoco_ctrl_debug_base_stability(self) -> str:
        """Base free-joint orientation (deg), twist, COM height — for tipping / fall-over diagnosis.

        Call only with ``_mj_lock`` held. Uses ``base_link`` free-joint quaternion and qvel layout
        (angular wx,wy,wz then linear vx,vy,vz in world frame).
        """
        if self._mjmodel is None or self._mjdata is None:
            return "base: (no model)"
        addrs = self._base_freejoint_addrs()
        if addrs is None:
            return "base: (no free joint on base_link)"
        qadr, vadr = addrs
        d = self._mjdata
        qw = float(d.qpos[qadr + 3])
        qx = float(d.qpos[qadr + 4])
        qy = float(d.qpos[qadr + 5])
        qz = float(d.qpos[qadr + 6])
        norm = math.hypot(qw, qx, qy, qz)
        if norm < 1e-9:
            return "base: (invalid quaternion)"
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        # Tait–Bryan ZYX (yaw, pitch, roll) in radians; pitch = rotation about body Y.
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        r_deg, p_deg, y_deg = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
        v0 = int(vadr)
        wv = np.asarray(d.qvel[v0 : v0 + 3], dtype=np.float64).ravel()
        lv = np.asarray(d.qvel[v0 + 3 : v0 + 6], dtype=np.float64).ravel()
        wn = float(np.linalg.norm(wv))
        ln = float(np.linalg.norm(lv))
        bid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, self._spec.base_link_name)
        z = float(d.xpos[bid, 2]) if 0 <= bid < int(self._mjmodel.nbody) else float("nan")
        # Body X (forward) and Z (up) axes in world: xmat rows are body axes expressed in world (MuJoCo).
        R = np.asarray(d.xmat[bid], dtype=np.float64).reshape(3, 3)
        up_z = float(R[2, 2])
        fwd_z = float(R[0, 2])
        floor_z = float("nan")
        zb_place = float("nan")
        if bid >= 0:
            try:
                from emet.simulation.molmospaces_spawn import (
                    effective_floor_geom_name,
                    robot_placement_bottom_z,
                    walkable_floor_z_at_xy,
                )

                floor_nm = effective_floor_geom_name(self._mjmodel)
                x, y = float(d.xpos[bid, 0]), float(d.xpos[bid, 1])
                zf = walkable_floor_z_at_xy(
                    self._mjmodel,
                    d,
                    x,
                    y,
                    floor_geom_name=floor_nm,
                    exclude_body_id=int(bid),
                )
                if zf is not None:
                    floor_z = float(zf)
                rb = molmospaces_spawn._bodies_descending_from(self._mjmodel, int(bid))
                zb = robot_placement_bottom_z(
                    self._mjmodel,
                    d,
                    base_body_name=self._spec.base_link_name,
                    robot_bodies=rb,
                )
                if zb is not None:
                    zb_place = float(zb)
            except Exception:
                pass
        pen = zb_place - floor_z if np.isfinite(zb_place) and np.isfinite(floor_z) else float("nan")
        return (
            f"base rpy_deg=({r_deg:.2f},{p_deg:.2f},{y_deg:.2f}) z={z:.4f} "
            f"z_floor={floor_z:.4f} zb_place={zb_place:.4f} pen={pen:+.4f} "
            f"|ω|={wn:.4f} |v|={ln:.4f} up·ẑ={up_z:.3f} fwd·ẑ={fwd_z:.3f}"
        )

    def _mujoco_ctrl_debug_summary(self) -> str:
        """One-line diagnostics (call with ``_mj_lock`` held)."""
        if self._mjmodel is None or self._mjdata is None:
            return "no model/data"
        d = self._mjdata
        m = self._mjmodel
        qv = float(np.max(np.abs(d.qvel))) if d.qvel.size else 0.0
        cu = float(np.max(np.abs(d.ctrl))) if d.ctrl.size else 0.0
        parts = [
            f"phys_steps={self._physics_steps_executed}",
            f"max|qvel|={qv:.5g}",
            f"max|ctrl|={cu:.5g}",
        ]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self._spec.base_link_name)
        if 0 <= bid < int(m.nbody):
            z = float(np.asarray(d.xpos)[bid, 2])
            parts.append(f"base_z={z:.4f}")
        for aname in ("torso1", "steer1", "wheel1"):
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                parts.append(f"ctrl[{aname}]={float(d.ctrl[aid]):.5g}")
        hip_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "torso_joint1")
        if hip_jid >= 0:
            hq = int(m.jnt_qposadr[hip_jid])
            parts.append(f"q[torso_joint1]={float(d.qpos[hq]):.5g}")
        hold = self._joint_ctrl_hold
        if hold is not None and hold.size >= 3:
            parts.append(f"hold0..2={float(hold[0]):.4g},{float(hold[1]):.4g},{float(hold[2]):.4g}")
        return " ".join(parts)

    def _maybe_log_mujoco_ctrl_debug_after_apply(self) -> None:
        if not self._mujoco_ctrl_debug_enabled() or not self._ctrl_debug_emit_apply_logs:
            return
        self._ctrl_debug_periodic_counter += 1
        if self._ctrl_debug_initial_logs_remaining > 0:
            self._ctrl_debug_initial_logs_remaining -= 1
            logger.info(f"[mujoco_ctrl_debug] after_apply {self._mujoco_ctrl_debug_summary()}")
            logger.info(f"[mujoco_ctrl_debug] after_apply {self._mujoco_ctrl_debug_pd_tracking()}")
            logger.info(f"[mujoco_ctrl_debug] after_apply_base {self._mujoco_ctrl_debug_base_stability()}")
            return
        # ~0.5 s at 80 Hz × 6 substeps
        if self._ctrl_debug_periodic_counter % 240 == 0:
            logger.info(f"[mujoco_ctrl_debug] periodic {self._mujoco_ctrl_debug_summary()}")
            logger.info(f"[mujoco_ctrl_debug] periodic {self._mujoco_ctrl_debug_pd_tracking()}")
            logger.info(f"[mujoco_ctrl_debug] periodic_base {self._mujoco_ctrl_debug_base_stability()}")

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @override
    def is_running(self) -> bool:
        return self._running

    @override
    def get_control_mode(self) -> str:
        return self.control_mode

    def _load_model(self) -> None:
        if self._scene_model is not None:
            self._mjmodel = self._scene_model
        elif self._scene_xml is not None:
            self._mjmodel = mujoco.MjModel.from_xml_string(self._scene_xml)
        else:
            raise ValueError("Either scene_xml or scene_model must be provided")
        self._mjdata = mujoco.MjData(self._mjmodel)
        self._molmospaces_autoplace_snap_qpos0 = False
        self._planar_autoplace_snap_qpos0 = False
        self._planar_autoplace_world_xyt = None
        with self._mj_lock:
            mujoco.mj_forward(self._mjmodel, self._mjdata)
            self._molmospaces_autoplace_free_base_after_load()
            self._robocasa_planar_autoplace_after_load()
            self._robocasa_freejoint_autoplace_after_load()
        self._planar_base_actuator_ids_cache = None
        self._configure_mj_substeps_per_tick()

    def _configure_mj_substeps_per_tick(self) -> None:
        """Run several ``mj_step`` calls per server tick so wall-clock pacing matches ``opt.timestep``.

        At 80 Hz one wall tick is 12.5 ms; with MuJoCo ``timestep`` = 2 ms, ~6 integration steps belong
        in that interval (same idea as MolmoSpaces ``_n_ctrl_steps_per_policy``). Otherwise the sim
        advances too slowly in sim-time and joint PD + contacts look artificially sloppy.
        """
        self._mj_substeps_per_tick = 1
        if self._mjmodel is None:
            return
        dt = float(self._mjmodel.opt.timestep)
        if dt <= 0.0:
            return
        tick = 1.0 / max(1.0, float(self.simulation_rate))
        n = int(tick / dt + 0.5)
        self._mj_substeps_per_tick = max(1, min(64, n))
        if self._mj_substeps_per_tick > 1:
            logger.info(
                f"MuJoCo substeps per server tick: {self._mj_substeps_per_tick} "
                f"(timestep={dt:g}s, server_rate={self.simulation_rate}Hz)"
            )

    def _molmospaces_autoplace_free_base_after_load(self) -> None:
        """Move merged MolmoSpaces + mobile robot off origin when the base starts inside scene clutter."""
        if self._mjmodel is None or self._mjdata is None:
            return
        self._molmospaces_autoplace_snap_qpos0 = apply_molmospaces_freejoint_base_autoplace(
            self._mjmodel,
            self._mjdata,
            merged_mjcf_path=self._scene_disk_path,
            base_body_name=self._spec.base_link_name,
            environment=self._environment_descriptor,
            scene_source_basename=self._scene_source_basename,
            robot_key=self._spec.name,
            debug=self._debug_molmospaces_spawn,
        )

    def _robocasa_planar_autoplace_after_load(self) -> None:
        """Reposition planar (slide X/Y + yaw) base away from Robocasa clutter when enabled."""
        if not scene_base_spawn.want_robocasa_planar_autoplace(
            environment=self._environment_descriptor,
            robot_spec=self._spec,
        ):
            return
        if self._mjmodel is None or self._mjdata is None:
            return
        if self._base_freejoint_addrs() is not None:
            return
        names = getattr(self._spec, "planar_base_joint_names", None)
        if not names or len(names) != 3:
            return
        base_name = self._spec.base_link_name
        joint_names = (str(names[0]), str(names[1]), str(names[2]))
        if self._debug_molmospaces_spawn:
            logger.info(
                f"Robocasa planar spawn debug: environment={self._environment_descriptor!r} "
                f"base_body_name={base_name!r} joints={joint_names!r}"
            )
        fp = self._spec.footprint
        base_margin = float(
            0.5
            * np.hypot(
                float(fp.length) + abs(float(fp.length_offset)),
                float(fp.width) + abs(float(fp.width_offset)),
            )
            + 0.10
        )
        extra_xy = float(self._spec.planar_spawn_xy_extra_margin_m)
        margin = base_margin + extra_xy
        clip_pad = self._spec.planar_spawn_clip_edge_pad_m
        if clip_pad is None:
            clip_pad = float(0.22 + 0.5 * extra_xy)
        guard_names = self._spec.planar_spawn_clip_guard_body_names
        if not guard_names and self._spec.planar_spawn_clip_guard_body_name:
            guard_names = (self._spec.planar_spawn_clip_guard_body_name,)
        guard_pad = float(self._spec.planar_spawn_clip_guard_pad_m)
        try:
            placed = scene_base_spawn.find_planar_base_xyt(
                self._mjmodel,
                self._mjdata,
                base_body_name=base_name,
                joint_names=joint_names,
                spawn_profile="robocasa",
                scene_label=self._scene_source_basename,
                merged_mjcf_path=self._scene_disk_path,
                environment=self._environment_descriptor,
                footprint_xy_margin_m=margin,
                clip_edge_pad_m=clip_pad,
                clip_guard_body_names=guard_names,
                clip_guard_pad_m=guard_pad,
                robocasa_first_clearance_m=self._spec.planar_spawn_robocasa_first_clearance_m,
            )
        except Exception as e:
            logger.warning(f"Robocasa planar autoplace skipped ({e!r}).")
            return
        if placed is None:
            logger.info("Robocasa planar autoplace: no safer (x,y,yaw) found; keeping MJCF default base pose.")
            return
        wx, wy, wt = placed
        self._planar_autoplace_world_xyt = np.array([float(wx), float(wy), float(wt)], dtype=np.float64)
        logger.info(
            f"Robocasa planar autoplace: moved base to x={wx:.3f} y={wy:.3f} theta={wt:.3f} "
            f"(joints {joint_names!r}) for clearance from scene geometry."
        )
        if self._debug_molmospaces_spawn:
            try:
                mujoco.mj_forward(self._mjmodel, self._mjdata)
                for ln in molmospaces_spawn.format_spawn_contact_report(
                    self._mjmodel,
                    self._mjdata,
                    base_body_name=base_name,
                    floor_geom_name="floor",
                    max_lines=40,
                    dist_report_threshold=0.12,
                ):
                    logger.info(f"[scene_base_spawn/post-place] {ln}")
            except Exception as e:
                logger.warning(f"Robocasa planar spawn debug contact report failed: {e!r}")
        for jn, _val in zip(joint_names, (wx, wy, wt), strict=True):
            jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                qadr = int(self._mjmodel.jnt_qposadr[jid])
                self._mjmodel.qpos0[qadr] = float(self._mjdata.qpos[qadr])
        self._planar_autoplace_snap_qpos0 = True

    def _robocasa_freejoint_autoplace_after_load(self) -> None:
        """Reposition freejoint base away from Robocasa clutter (Galaxea R1 / RB-Y1, etc.)."""
        if not scene_base_spawn.want_robocasa_freejoint_autoplace(
            environment=self._environment_descriptor,
            robot_spec=self._spec,
        ):
            return
        if self._mjmodel is None or self._mjdata is None:
            return
        if self._base_freejoint_addrs() is None:
            return
        base_name = self._spec.base_link_name
        if self._debug_molmospaces_spawn:
            logger.info(
                f"Robocasa freejoint spawn debug: environment={self._environment_descriptor!r} "
                f"base_body_name={base_name!r}"
            )
        try:
            placed = scene_base_spawn.find_molmospaces_freejoint_xyz(
                self._mjmodel,
                self._mjdata,
                base_body_name=base_name,
                scene_label=self._scene_source_basename,
                merged_mjcf_path=self._scene_disk_path,
                environment=self._environment_descriptor,
            )
        except Exception as e:
            logger.warning(f"Robocasa freejoint autoplace skipped ({e!r}).")
            return
        if placed is None:
            logger.info(
                "Robocasa freejoint autoplace: no safer (x,y,z) found; keeping MJCF default base pose."
            )
            return
        x, y, z = placed
        logger.info(
            f"Robocasa freejoint autoplace: moved base on {base_name!r} to "
            f"({x:.3f}, {y:.3f}, {z:.3f}) for clearance from scene geometry."
        )
        if self._debug_molmospaces_spawn:
            try:
                mujoco.mj_forward(self._mjmodel, self._mjdata)
                for ln in molmospaces_spawn.format_spawn_contact_report(
                    self._mjmodel,
                    self._mjdata,
                    base_body_name=base_name,
                    floor_geom_name="floor",
                    max_lines=40,
                    dist_report_threshold=0.12,
                ):
                    logger.info(f"[scene_base_spawn/post-place] {ln}")
            except Exception as e:
                logger.warning(f"Robocasa freejoint spawn debug contact report failed: {e!r}")
        addrs = self._base_freejoint_addrs()
        if addrs is not None:
            qadr = int(addrs[0])
            self._mjmodel.qpos0[qadr : qadr + 7] = self._mjdata.qpos[qadr : qadr + 7]
            self._molmospaces_autoplace_snap_qpos0 = True

    def _snapshot_stationary_base_freejoint_pose(self) -> None:
        """Remember base free-joint ``qpos`` for idle sim (see :meth:`_hold_stationary_base_freejoint_if_idle`)."""
        if self._mjmodel is None or self._mjdata is None:
            return
        addrs = self._base_freejoint_addrs()
        if addrs is None:
            self._stationary_base_freejoint_qpos = None
            return
        qadr, _ = addrs
        self._stationary_base_freejoint_qpos = np.array(
            self._mjdata.qpos[qadr : qadr + 7], dtype=np.float64, copy=True
        )

    def _hold_stationary_base_freejoint_if_idle(self) -> None:
        """While there is no navigation goal, pin the base free joint to the post-spawn snapshot."""
        if self._nav_goal_world is not None:
            return
        if self._mjmodel is None or self._mjdata is None:
            return
        snap = self._stationary_base_freejoint_qpos
        if snap is None or int(snap.shape[0]) != 7:
            return
        addrs = self._base_freejoint_addrs()
        if addrs is None:
            return
        qadr, vadr = addrs
        self._mjdata.qpos[qadr : qadr + 7] = snap
        if vadr >= 0:
            self._mjdata.qvel[vadr : vadr + 6] = 0.0

    def _restore_merged_base_freejoint_from_qpos0(self) -> None:
        """Put ``base_link`` free joint back to ``qpos0`` after :meth:`_stabilize_physics_state_after_load`.

        Stabilize runs a few ``mj_step`` calls with PD actuators synced to ``qpos``. For a floating
        base that can **drift** the robot away from the MolmoSpaces spawn chosen from occupancy,
        while logs and ``qpos0`` still show the intended pose — so the viewer no longer matches the
        top-down map. Restoring the 7 free-joint coordinates from ``qpos0`` preserves spawn XY/Z.

        Callers should run additional ``mj_step`` + hold re-apply after this (see ``start()``) so the
        snap does not leave the articulated body in a bad transient.
        """
        if self._mjmodel is None or self._mjdata is None:
            return
        addrs = self._base_freejoint_addrs()
        if addrs is None:
            return
        qadr, vadr = int(addrs[0]), int(addrs[1])
        self._mjdata.qpos[qadr : qadr + 7] = self._mjmodel.qpos0[qadr : qadr + 7]
        if vadr >= 0:
            self._mjdata.qvel[vadr : vadr + 6] = 0.0

    def _restore_planar_base_from_qpos0(self) -> None:
        """Restore planar slide+yaw ``qpos`` from ``qpos0`` after stabilize (same idea as free joint)."""
        if self._mjmodel is None or self._mjdata is None:
            return
        names = getattr(self._spec, "planar_base_joint_names", None)
        if not names or len(names) != 3:
            return
        for jn in names:
            jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, str(jn))
            if jid < 0:
                return
            qadr = int(self._mjmodel.jnt_qposadr[jid])
            vadr = int(self._mjmodel.jnt_dofadr[jid])
            self._mjdata.qpos[qadr] = float(self._mjmodel.qpos0[qadr])
            if vadr >= 0:
                self._mjdata.qvel[vadr] = 0.0
        mujoco.mj_forward(self._mjmodel, self._mjdata)

    def _spawn_footprint_xy_margin_m(self) -> float:
        fp = self._spec.footprint
        base_margin = float(
            0.5
            * np.hypot(
                float(fp.length) + abs(float(fp.length_offset)),
                float(fp.width) + abs(float(fp.width_offset)),
            )
            + 0.10
        )
        extra = float(getattr(self._spec, "planar_spawn_xy_extra_margin_m", 0.0) or 0.0)
        return base_margin + extra

    def _compute_robocasa_spawn_floor_map(self) -> dict[str, Any] | None:
        if self._environment_descriptor and self._environment_descriptor.get("spawn_floor_map") is not None:
            return dict(self._environment_descriptor["spawn_floor_map"])
        if self._mjmodel is None or self._mjdata is None:
            return None
        with self._mj_lock:
            return scene_base_spawn.compute_spawn_walkable_map_metrics(
                self._mjmodel,
                self._mjdata,
                base_body_name=self._spec.base_link_name,
                grid_resolution_m=0.10,
                footprint_xy_margin_m=self._spawn_footprint_xy_margin_m(),
            )

    def _is_molmospaces_session(self) -> bool:
        env = self._environment_descriptor
        if isinstance(env, dict) and env.get("kind") == "molmospaces":
            return True
        bn = self._scene_source_basename or ""
        return bn.startswith("molmospaces_merged")

    def _use_molmospaces_nav_teleport(self) -> bool:
        return self._is_molmospaces_session() and molmospaces_nav_teleport_enabled()

    def _build_emet_session(
        self,
        *,
        robocasa: bool,
        spawn_floor_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mj_name: str | None = None
        if self._mjmodel is not None:
            try:
                if self._mjmodel.nnames > 0:
                    n0 = self._mjmodel.names[0]
                    mj_name = n0.decode("utf-8") if isinstance(n0, (bytes, bytearray)) else str(n0)
            except Exception:
                mj_name = None
        if self._environment_descriptor:
            env = dict(self._environment_descriptor)
        elif robocasa:
            env = {"kind": "robocasa"}
        else:
            env = {"kind": "default_table"}
        caps: dict[str, Any] = {
            "teleport_base": self._use_molmospaces_nav_teleport(),
            "nav_velocity_drive": True,
            "depth": bool(self._spec.camera_names),
            "num_cameras": len(self._spec.camera_names),
            "dof": int(self._spec.dof),
        }
        session: dict[str, Any] = {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "robosuite_sim",
            "is_simulation": True,
            EMET_ZMQ_ROBOT_ID_KEY: self._spec.name,
            "capabilities": caps,
            "environment": env,
        }
        if mj_name:
            session["mjcf_model_name"] = mj_name
        if self._scene_source_basename:
            session["scene_source_basename"] = self._scene_source_basename
        if self._session_extra:
            session.update(self._session_extra)
        if self._initial_xyt is not None:
            ixy = np.asarray(self._initial_xyt, dtype=np.float64).reshape(-1)[:3]
            session["navigation_origin_xyt"] = [float(ixy[0]), float(ixy[1]), float(ixy[2])]
        if spawn_floor_map is not None:
            session["spawn_floor_map"] = spawn_floor_map
        session["nav_contract"] = {
            "gps_compass_frame": "episode_relative_to_navigation_origin_xyt",
            "xyt_with_nav_world": "absolute_mujoco_world_xyt",
            "xyt_default_non_relative": "episode_relative_xyt_then_compose_with_navigation_origin",
            "xyt_with_nav_relative": "delta_xy_in_current_world_heading",
            "client_sets_nav_world": "GenericZmqClient move_base_to(..., world_frame=True) for voxel planner goals",
            "sim_teleport": "action.nav_teleport or EMET_SIM_NAV_TELEPORT=1 on client",
            "sim_motion_default": "velocity_drive via planar actuators (_nav_goal_world P-control)",
            "sim_debug": "EMET_SIM_NAV_DEBUG=1 for verbose per-action nav logs on server",
        }
        if self._nav_world_clip_rect is not None:
            session["nav_walkable_clip_eroded_xy"] = list(self._nav_world_clip_rect)
        return session

    def _attach_emet_session(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._emet_session is not None:
            message[EMET_ZMQ_SESSION_KEY] = self._emet_session
        return message

    def get_scene_summary(self) -> str:
        """Return a short text summary of the scene: robot, position, and notable objects."""
        if self._mjmodel is None or self._mjdata is None:
            return "Scene not loaded."
        lines = [
            "--- Scene summary ---",
            f"Robot: {self._spec.name}",
        ]
        try:
            with self._mj_lock:
                xyt = self.get_base_xyt()
                lines.append(f"Robot position (x, y, theta): ({xyt[0]:.3f}, {xyt[1]:.3f}, {xyt[2]:.3f})")
                body_id = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, self._spec.base_link_name)
                if body_id >= 0:
                    z = float(self._mjdata.body(body_id).xpos[2])
                    lines.append(f"Robot height (z): {z:.3f}")
                for bid in range(self._mjmodel.nbody):
                    name = mujoco.mj_id2name(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, bid)
                    if name is None or name == self._spec.base_link_name:
                        continue
                    xpos = self._mjdata.body(bid).xpos
                    if "object1" in (name or ""):
                        lines.append(f"  Blue cube (object1): pos ({xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f})")
                    elif "object2" in (name or ""):
                        lines.append(f"  Red cylinder (object2): pos ({xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f})")
                    elif name in ("table", "floor"):
                        lines.append(f"  {name}: pos ({xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f})")
        except Exception as e:
            logger.warning(f"get_scene_summary partial failure: {e!r}")
            lines.append("Robot position: (unknown)")
        lines.append("-------------------")
        return "\n".join(lines)

    def get_base_xyt(self) -> np.ndarray:
        base_name = self._spec.base_link_name
        if self._mjdata is None:
            return np.zeros(3)
        try:
            with self._mj_lock:
                xpos = self._mjdata.body(base_name).xpos
                xmat = self._mjdata.body(base_name).xmat.reshape(3, 3)
                theta = np.arctan2(xmat[1, 0], xmat[0, 0])
                return np.array([xpos[0], xpos[1], theta])
        except Exception as e:
            logger.warning(f"get_base_xyt failed for body {base_name!r}: {e!r}")
            return np.zeros(3)

    def get_base_pose(self) -> np.ndarray | None:
        if self._initial_xyt is None:
            return None
        xyt = self.get_base_xyt()
        return xyt_global_to_base(xyt, self._initial_xyt)

    def get_joint_state(self):
        dof = self._spec.dof
        positions = np.zeros(dof)
        velocities = np.zeros(dof)
        efforts = np.zeros(dof)

        if self._mjdata is None:
            return positions, velocities, efforts

        with self._mj_lock:
            for i, jname in enumerate(self._spec.joint_names):
                try:
                    jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid < 0:
                        continue
                    qadr = self._mjmodel.jnt_qposadr[jid]
                    vadr = self._mjmodel.jnt_dofadr[jid]
                    positions[i] = self._mjdata.qpos[qadr]
                    velocities[i] = self._mjdata.qvel[vadr]
                except Exception:
                    continue

        return positions, velocities, efforts

    def _close_renderers(self) -> None:
        with self._render_lock:
            if self._primary_renderer is not None:
                try:
                    self._primary_renderer.close()
                except Exception:
                    pass
                self._primary_renderer = None

    def _apply_optional_mujoco_render_flip_ud(self, img: np.ndarray) -> np.ndarray:
        """Legacy vertical flip when ``EMET_ROBOSUITE_RENDER_FLIPUD=1``.

        Ignored when :attr:`RobotSpec.robosuite_rgb_depth_ops` is non-empty (use ops instead).
        """
        v = os.environ.get("EMET_ROBOSUITE_RENDER_FLIPUD")
        if v is None or not str(v).strip():
            return img
        if str(v).strip().lower() in ("1", "true", "yes", "on"):
            return np.flipud(img).copy()
        return img

    def _render_rgb_raw(self, camera_name: str) -> np.ndarray:
        """RGB uint8 from ``mujoco.Renderer`` at primary resolution (no pixel postprocess)."""
        cam = self._camera_for_renderer(camera_name)
        with self._mj_lock:
            with self._render_lock:
                if self._primary_renderer is None:
                    self._primary_renderer = mujoco.Renderer(self._mjmodel, _PRIMARY_RH, _PRIMARY_RW)
                renderer = self._primary_renderer
                renderer.update_scene(self._mjdata, camera=cam)
                rgb = cast(np.ndarray, renderer.render())
                return np.asarray(rgb, dtype=np.uint8).copy()

    def _render_depth_raw(self, camera_name: str) -> np.ndarray:
        """Depth float32 for ``camera_name`` after ``update_scene`` (enable_depth_rendering)."""
        cam = self._camera_for_renderer(camera_name)
        with self._mj_lock:
            with self._render_lock:
                renderer = self._primary_renderer
                renderer.enable_depth_rendering()
                try:
                    renderer.update_scene(self._mjdata, camera=cam)
                    depth = cast(np.ndarray, renderer.render())
                    return np.asarray(depth, dtype=np.float32).copy()
                finally:
                    renderer.disable_depth_rendering()

    def _postprocess_rgb_depth_and_K(
        self, camera_name: str, rgb: np.ndarray, depth: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Apply ``RobotSpec.robosuite_rgb_depth_ops`` or legacy env flipud; return matching pinhole ``K``."""
        h0, w0 = int(rgb.shape[0]), int(rgb.shape[1])
        K0 = self._get_camera_K(camera_name, w0, h0)
        ops = getattr(self._spec, "robosuite_rgb_depth_ops", ()) or ()
        if ops:
            rgb = apply_pinhole_pixel_ops(rgb, ops)
            if depth is not None:
                depth = apply_pinhole_pixel_ops(depth, ops)
            K, _, _ = chain_pinhole_K_pixel_ops(K0, h0, w0, ops)
        else:
            rgb = self._apply_optional_mujoco_render_flip_ud(rgb)
            if depth is not None:
                depth = self._apply_optional_mujoco_render_flip_ud(depth)
            K = K0
        return rgb, depth, K

    def _primary_rgb_only(self, camera_name: str) -> np.ndarray:
        """RGB at primary resolution (no depth pass). Used for stereo aux camera."""
        rgb = self._render_rgb_raw(camera_name)
        rgb, _, _ = self._postprocess_rgb_depth_and_K(camera_name, rgb, None)
        return rgb

    def _primary_rgb_only_with_K(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        """RGB + intrinsics after the same postprocess as depth observations."""
        rgb = self._render_rgb_raw(camera_name)
        rgb, _, K = self._postprocess_rgb_depth_and_K(camera_name, rgb, None)
        return rgb, K

    def _primary_rgb_and_depth(self, camera_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB + depth + intrinsics ``K`` matching both buffers (primary resolution)."""
        rgb = self._render_rgb_raw(camera_name)
        depth = self._render_depth_raw(camera_name)
        rgb, depth, K = self._postprocess_rgb_depth_and_K(camera_name, rgb, depth)
        return rgb, depth, K

    def _stereo_right_camera_name(self) -> str | None:
        return stereo_right_camera_name_from_spec(list(self._spec.camera_names))

    def _camera_for_renderer(self, camera_name: str) -> int | str:
        """Resolve RobotSpec camera name to a MuJoCo camera id, or free camera if none match.

        Requires a real ``<camera>`` in the MJCF (``mjOBJ_CAMERA``). **Sites** with the same
        name are not used by ``mujoco.Renderer``; falling back to camera ``-1`` is a fixed
        world view and will not follow the robot.
        """
        cid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cid >= 0:
            return cid
        return -1

    def _get_camera_K(self, camera_name: str, width: int = 640, height: int = 480):
        """Compute intrinsic matrix from MuJoCo camera fovy."""
        cam_id = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            return np.eye(3)
        fovy = self._mjmodel.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(np.radians(fovy) / 2)
        return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

    def _ensure_joint_ctrl_hold_buffers(self) -> None:
        """Allocate :attr:`_joint_ctrl_hold` and client pin mask (spec actuator count)."""
        n = min(len(self._spec.actuator_names), len(self._spec.joint_names))
        if self._joint_ctrl_hold is None or int(self._joint_ctrl_hold.shape[0]) != n:
            self._joint_ctrl_hold = np.zeros(n, dtype=np.float64)
        if self._joint_ctrl_hold_client_pin is None or int(self._joint_ctrl_hold_client_pin.shape[0]) != n:
            self._joint_ctrl_hold_client_pin = np.zeros(n, dtype=np.bool_)

    def _seed_joint_ctrl_hold_from_keyframe(self, key_name: str = "home") -> bool:
        """Copy MJCF keyframe ``ctrl`` into :attr:`_joint_ctrl_hold` (position setpoints for PD actuators)."""
        if self._mjmodel is None or self._mjdata is None:
            return False
        kid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if kid < 0:
            return False
        self._ensure_joint_ctrl_hold_buffers()
        assert self._joint_ctrl_hold is not None
        n = min(len(self._spec.actuator_names), int(self._joint_ctrl_hold.shape[0]))
        key_ctrl = np.asarray(self._mjmodel.key_ctrl[kid], dtype=np.float64)
        for i in range(n):
            aname = self._spec.actuator_names[i]
            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0 and aid < int(key_ctrl.shape[0]):
                self._joint_ctrl_hold[i] = float(key_ctrl[aid])
            elif aid >= 0:
                self._joint_ctrl_hold[i] = float(self._mjdata.ctrl[aid])
        if self._joint_ctrl_hold_client_pin is not None:
            self._joint_ctrl_hold_client_pin.fill(False)
        return True

    def _sync_actuator_ctrl_from_joint_positions(self) -> None:
        """Set full ``ctrl`` from joint transmissions + current ``qpos``, then refresh the spec hold buffer.

        Uses :attr:`_mujoco_stationary` (:class:`emet.simulation.mujoco_stationary_control.MujocoStationaryControl`)
        — MolmoSpaces-style stationary targets for **every** actuator index ``nu``, not only :class:`RobotSpec`
        rows (merged scenes often add extra actuators whose default ``ctrl=0`` would fight the pose).
        """
        if self._mjmodel is None or self._mjdata is None:
            return
        self._ensure_joint_ctrl_hold_buffers()
        if self._joint_ctrl_hold_client_pin is not None:
            self._joint_ctrl_hold_client_pin.fill(False)
        self._mujoco_stationary.sync_ctrl_and_spec_hold(
            self._mjmodel, self._mjdata, self._spec, self._joint_ctrl_hold
        )

    def _preserve_joint_ctrl_hold_from_ctrl(self) -> None:
        """Keep PD targets in :attr:`_joint_ctrl_hold` from current ``data.ctrl`` (not from ``qpos``).

        Use after MJCF ``home`` / zero pose setup and after dynamics settle so post-load ``q`` drift does
        not retarget unpinned joints. Contrasts with :meth:`_sync_actuator_ctrl_from_joint_positions`,
        which sets hold from current joint angles (MolmoSpaces ``set_to_stationary`` / hold-current-pose).
        """
        self._ensure_joint_ctrl_hold_buffers()
        self._snapshot_spec_hold_from_ctrl()
        self._apply_joint_ctrl_hold_to_actuators(refresh_unpinned_hold=False)

    def _snapshot_spec_hold_from_ctrl(self) -> None:
        """After code paths that write ``data.ctrl`` directly (e.g. ``head_to``), mirror into the hold buffer.

        Otherwise :meth:`_apply_joint_ctrl_hold_to_actuators` immediately overwrites those actuators on
        the next ``mj_step``.
        """
        if self._mjmodel is None or self._mjdata is None:
            return
        self._ensure_joint_ctrl_hold_buffers()
        if self._joint_ctrl_hold is None:
            return
        n = min(len(self._spec.actuator_names), int(self._joint_ctrl_hold.shape[0]))
        for i in range(n):
            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, self._spec.actuator_names[i])
            if aid < 0:
                continue
            self._joint_ctrl_hold[i] = float(self._mjdata.ctrl[aid])
        n_pin = min(len(self._spec.actuator_names), int(self._joint_ctrl_hold.shape[0]))
        if self._joint_ctrl_hold_client_pin is None or int(self._joint_ctrl_hold_client_pin.shape[0]) != n_pin:
            self._joint_ctrl_hold_client_pin = np.zeros(n_pin, dtype=np.bool_)
        else:
            self._joint_ctrl_hold_client_pin.fill(False)

    def _refresh_unpinned_joint_ctrl_hold_from_stationary(self) -> None:
        """Align unpinned :attr:`_joint_ctrl_hold` rows with ``compute_stationary_ctrl_vector`` (current ``q``)."""
        if self._mjmodel is None or self._mjdata is None or self._joint_ctrl_hold is None:
            return
        n = min(len(self._spec.actuator_names), int(self._joint_ctrl_hold.shape[0]))
        if self._joint_ctrl_hold_client_pin is None or int(self._joint_ctrl_hold_client_pin.shape[0]) != n:
            self._joint_ctrl_hold_client_pin = np.zeros(n, dtype=np.bool_)
        pin = self._joint_ctrl_hold_client_pin
        hold = self._joint_ctrl_hold
        full = compute_stationary_ctrl_vector(self._mjmodel, self._mjdata)
        for i in range(n):
            if bool(pin[i]):
                continue
            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, self._spec.actuator_names[i])
            if aid >= 0:
                hold[i] = float(full[aid])

    def _apply_joint_ctrl_hold_to_actuators(self, *, refresh_unpinned_hold: bool = True) -> None:
        """Recompute full stationary ``ctrl`` from ``qpos``, then overlay :attr:`_joint_ctrl_hold`.

        When *refresh_unpinned_hold* is true, unpinned spec rows are copied from
        :func:`~emet.simulation.mujoco_stationary_control.compute_stationary_ctrl_vector` (current ``q``).

        **Realtime sim loop:** call with *refresh_unpinned_hold* **false** only. Do **not** refresh
        unpinned hold each tick or substep from ``q``—that retargets PD continuously and removes
        position stiffness (arms collapse). Unpinned targets are updated by
        :meth:`_sync_actuator_ctrl_from_joint_positions`, ZMQ ``joint`` (pinned rows), or
        :meth:`_snapshot_spec_hold_from_ctrl`.

        See :attr:`_mujoco_stationary` and :meth:`_sync_actuator_ctrl_from_joint_positions`.
        """
        if self._mjmodel is None or self._mjdata is None:
            return
        self._ensure_joint_ctrl_hold_buffers()
        if self._joint_ctrl_hold is None:
            return
        if refresh_unpinned_hold:
            self._refresh_unpinned_joint_ctrl_hold_from_stationary()
        self._mujoco_stationary.write_ctrl_with_spec_hold(
            self._mjmodel, self._mjdata, self._spec, self._joint_ctrl_hold
        )
        self._maybe_log_mujoco_ctrl_debug_after_apply()

    def _stabilize_physics_state_after_load(self) -> None:
        """Zero all velocities, align actuators with ``qpos``, and run a few dynamics steps."""
        if self._mjmodel is None or self._mjdata is None:
            return
        with self._mj_lock:
            self._mjdata.qvel.fill(0.0)
            self._preserve_joint_ctrl_hold_from_ctrl()
            mujoco.mj_forward(self._mjmodel, self._mjdata)
            for _ in range(8):
                self._apply_joint_ctrl_hold_to_actuators(refresh_unpinned_hold=False)
                mujoco.mj_step(self._mjmodel, self._mjdata)
            self._mjdata.qvel.fill(0.0)
            self._preserve_joint_ctrl_hold_from_ctrl()
            mujoco.mj_forward(self._mjmodel, self._mjdata)

    def _camera_pose_world(self, camera_name: str) -> np.ndarray:
        """4x4 **OpenCV** camera-to-world transform for pinhole unprojection (DynaMem voxel code).

        MuJoCo reports ``cam_xmat`` in an OpenGL-style camera frame (+Y up, −Z forward). EMET unprojection
        uses OpenCV-style rays (+Y down image rows, +Z into the scene). For the same physical camera,
        ``R_world_from_cv = R_mujoco @ diag(1,-1,-1)`` so ``p_world = R_mujoco @ (D @ p_cv)``.
        """
        if self._mjmodel is None or self._mjdata is None:
            return np.eye(4, dtype=np.float64)
        with self._mj_lock:
            cam_id = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                return np.eye(4, dtype=np.float64)
            R = np.asarray(self._mjdata.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
            pos = np.asarray(self._mjdata.cam_xpos[cam_id], dtype=np.float64).reshape(3)
            # MuJoCo camera frame: +Y up, −Z forward (OpenGL). Point unprojection in emet uses OpenCV
            # camera coordinates (+Y down, +Z forward). World = R_mj @ p_mj; with p_mj = D @ p_cv and
            # D = diag(1,-1,-1), we have p_world = (R_mj @ D) @ p_cv.
            d = np.diag([1.0, -1.0, -1.0])
            r_cv = R @ d
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = r_cv
            T[:3, 3] = pos
            return T

    def _base_freejoint_addrs(self) -> tuple[int, int] | None:
        """Return ``(qposadr, dofadr)`` for the free joint on ``base_link``, if any."""
        if self._mjmodel is None or self._mjdata is None:
            return None
        bid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, self._spec.base_link_name)
        if bid < 0:
            return None
        for j in range(self._mjmodel.njnt):
            if int(self._mjmodel.jnt_bodyid[j]) != bid:
                continue
            if self._mjmodel.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            return (int(self._mjmodel.jnt_qposadr[j]), int(self._mjmodel.jnt_dofadr[j]))
        return None

    @staticmethod
    def _spawn_rel_xyt_to_world(goal_rel: np.ndarray, init_world_xyt: np.ndarray) -> np.ndarray:
        """SE(2) compose: pose of goal in spawn frame ``goal_rel`` → world ``(x,y,theta)``."""
        x0, y0, t0 = float(init_world_xyt[0]), float(init_world_xyt[1]), float(init_world_xyt[2])
        gx, gy, gt = float(goal_rel[0]), float(goal_rel[1]), float(goal_rel[2])
        ca, sa = np.cos(t0), np.sin(t0)
        wx = x0 + ca * gx - sa * gy
        wy = y0 + sa * gx + ca * gy
        wt = float(np.arctan2(np.sin(t0 + gt), np.cos(t0 + gt)))
        return np.array([wx, wy, wt], dtype=np.float64)

    def _planar_base_velocity_actuator_ids(self) -> tuple[int, int, int] | None:
        """MuJoCo actuator indices driving ``planar_base_joint_names`` (velocity transmissions)."""
        if self._planar_base_actuator_ids_cache is not None:
            return self._planar_base_actuator_ids_cache
        names = getattr(self._spec, "planar_base_joint_names", None)
        if not names or len(names) != 3 or self._mjmodel is None:
            return None
        aids: list[int] = []
        for jn in names:
            jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                self._planar_base_actuator_ids_cache = None
                return None
            found = -1
            for a in range(self._mjmodel.nu):
                if int(self._mjmodel.actuator_trnid[a, 0]) == jid:
                    found = int(a)
                    break
            if found < 0:
                self._planar_base_actuator_ids_cache = None
                return None
            aids.append(found)
        self._planar_base_actuator_ids_cache = (aids[0], aids[1], aids[2])
        return self._planar_base_actuator_ids_cache

    def _teleport_planar_base_world_xyt(self, wx: float, wy: float, wt: float) -> bool:
        """Snap planar slide+slide+yaw so ``base_link`` reaches world (x,y,yaw); zero velocity ctrl."""
        names = getattr(self._spec, "planar_base_joint_names", None)
        if not names or len(names) != 3 or self._mjmodel is None or self._mjdata is None:
            return False
        jn = (str(names[0]), str(names[1]), str(names[2]))
        return bool(
            scene_base_spawn.write_planar_base_xyt(
                self._mjmodel,
                self._mjdata,
                joint_names=jn,
                world_x=float(wx),
                world_y=float(wy),
                world_yaw=float(wt),
                base_body_name=self._spec.base_link_name,
            )
        )

    @staticmethod
    def _spawn_rel_xyt_to_world(goal_rel: np.ndarray, init_world_xyt: np.ndarray) -> np.ndarray:
        """SE(2) compose: pose of goal in spawn frame ``goal_rel`` → world ``(x,y,theta)``."""
        x0, y0, t0 = float(init_world_xyt[0]), float(init_world_xyt[1]), float(init_world_xyt[2])
        gx, gy, gt = float(goal_rel[0]), float(goal_rel[1]), float(goal_rel[2])
        ca, sa = np.cos(t0), np.sin(t0)
        wx = x0 + ca * gx - sa * gy
        wy = y0 + sa * gx + ca * gy
        wt = float(np.arctan2(np.sin(t0 + gt), np.cos(t0 + gt)))
        return np.array([wx, wy, wt], dtype=np.float64)

    @staticmethod
    def _sim_nav_debug_enabled() -> bool:
        return env_sim_nav_debug()

    @staticmethod
    def _nav_triplet(xyt: np.ndarray | list | tuple) -> tuple[float, float, float]:
        a = np.asarray(xyt, dtype=np.float64).reshape(-1)
        if a.size < 3:
            a = np.pad(a, (0, max(0, 3 - a.size)), mode="constant")
        return float(a[0]), float(a[1]), float(a[2])

    def _log_nav_startup_banner(self) -> None:
        """Once at server start: frames, spawn pose, walkable clip (grep server log for ``[sim_nav]``)."""
        warn_sim_nav_env_flags()
        init = self._initial_xyt
        init_s = "None" if init is None else f"({init[0]:.3f}, {init[1]:.3f}, {init[2]:.3f})"
        clip = self._nav_world_clip_rect
        clip_s = "None" if clip is None else f"[{clip[0]:.3f}, {clip[1]:.3f}, {clip[2]:.3f}, {clip[3]:.3f}]"
        lines = [
            "--- [sim_nav] navigation frames ---",
            f"  spawn / navigation_origin (world): {init_s}",
            f"  walkable clip_eroded (xmin,xmax,ymin,ymax): {clip_s}",
            "  gps/compass in ZMQ obs: episode-relative to navigation_origin",
            "  xyt default: episode-relative -> compose with origin -> world goal",
            "  xyt + nav_world: absolute world (DynaMem / Dynagraph planner)",
            "  default motion: velocity drive (not teleport unless nav_teleport)",
            "  emet_session.nav_contract + nav_last (debug) on each observation",
            "-----------------------------------",
        ]
        block = "\n".join(lines)
        if env_sim_nav_debug():
            for line in lines:
                logger.warning(line)
        else:
            logger.info(block)

    def _log_nav_action(self, meta: dict[str, Any], *, applied: str) -> None:
        """Log one navigation command (always one line; full block if EMET_SIM_NAV_DEBUG)."""
        raw = meta.get("raw_xyt", [])
        goal = meta.get("goal_world", [])
        base_b = meta.get("base_world_before", [])
        gps = meta.get("gps_episode")
        frame = meta.get("frame", "?")
        jump = float(meta.get("jump_from_base_m", 0.0))
        one = (
            f"[sim_nav] {applied} frame={frame} raw={[round(float(raw[0]), 3), round(float(raw[1]), 3), round(float(raw[2]), 3)]} "
            f"goal_world={[round(float(goal[0]), 3), round(float(goal[1]), 3), round(float(goal[2]), 3)]} "
            f"base_before={[round(float(base_b[0]), 3), round(float(base_b[1]), 3)]} "
            f"Δxy={jump:.2f}m nav_world={meta.get('nav_world')} nav_relative={meta.get('nav_relative')} "
            f"nav_teleport={meta.get('nav_teleport')}"
        )
        logger.warning(one)
        if self._emet_session is not None:
            self._emet_session["nav_last"] = dict(meta)
            self._emet_session["nav_last"]["applied"] = applied
        if not self._sim_nav_debug_enabled():
            return
        gps_s = "None" if gps is None else f"({gps[0]:.3f}, {gps[1]:.3f}, {gps[2]:.3f})"
        pre = meta.get("goal_world_pre_clamp", goal)
        extra = ""
        if "compose_jump_m" in meta:
            extra += f"\n  compose_jump_m: {meta['compose_jump_m']:.3f}"
        if meta.get("clamped"):
            extra += f"\n  clamped from ({pre[0]:.3f}, {pre[1]:.3f})"
        logger.info(
            "[sim_nav] detail action_step=%s\n"
            "  frame: %s\n"
            "  flags: nav_world=%s nav_relative=%s nav_teleport=%s\n"
            "  raw_xyt: %s\n"
            "  spawn_world (navigation_origin): %s\n"
            "  base body world (before): %s\n"
            "  gps episode (get_base_pose): %s\n"
            "  goal_world (after clamp): %s%s",
            meta.get("action_step"),
            frame,
            meta.get("nav_world"),
            meta.get("nav_relative"),
            meta.get("nav_teleport"),
            raw,
            meta.get("spawn_world_xyt"),
            base_b,
            gps_s,
            goal,
            extra,
        )

    def _resolve_nav_goal_world_xyt(
        self,
        action: dict[str, Any],
        raw: np.ndarray,
        init: np.ndarray,
    ) -> tuple[float, float, float, dict[str, Any]]:
        """Convert action xyt to world goal; return metadata for :meth:`_log_nav_action`."""
        meta: dict[str, Any] = {
            "action_step": action.get("step"),
            "raw_xyt": np.asarray(raw, dtype=np.float64).reshape(-1)[:3].tolist(),
            "nav_relative": bool(action.get("nav_relative", False)),
            "nav_world": bool(action.get("nav_world", False)),
            "nav_teleport": bool(action.get("nav_teleport", False)),
            "spawn_world_xyt": np.asarray(init, dtype=np.float64).reshape(-1)[:3].tolist(),
        }
        cur = self.get_base_xyt()
        meta["base_world_before"] = [float(cur[0]), float(cur[1]), float(cur[2])]
        pose_ep = self.get_base_pose()
        if pose_ep is not None:
            ep = np.asarray(pose_ep, dtype=np.float64).reshape(-1)[:3]
            meta["gps_episode"] = [float(ep[0]), float(ep[1]), float(ep[2])]

        if meta["nav_world"] and meta["nav_relative"]:
            logger.warning(
                "[sim_nav] action has both nav_world and nav_relative; using nav_world (absolute MuJoCo xyt)."
            )
            meta["nav_relative"] = False
        if meta["nav_relative"]:
            meta["frame"] = "relative_delta_world"
            ct = float(cur[2])
            dx, dy, dt = float(raw[0]), float(raw[1]), float(raw[2])
            wx = float(cur[0]) + np.cos(ct) * dx - np.sin(ct) * dy
            wy = float(cur[1]) + np.sin(ct) * dx + np.cos(ct) * dy
            wt = float(np.arctan2(np.sin(cur[2] + dt), np.cos(cur[2] + dt)))
        elif meta["nav_world"]:
            meta["frame"] = "mujoco_world"
            wx, wy, wt = float(raw[0]), float(raw[1]), float(raw[2])
        else:
            meta["frame"] = "spawn_compose"
            world = self._spawn_rel_xyt_to_world(raw[:3], init)
            wx, wy, wt = float(world[0]), float(world[1]), float(world[2])
            compose_jump = float(np.hypot(wx - float(cur[0]), wy - float(cur[1])))
            meta["compose_jump_m"] = compose_jump
            if compose_jump > 8.0:
                meta["frame"] = "spawn_compose_corrected_world"
                wx, wy, wt = float(raw[0]), float(raw[1]), float(raw[2])
                logger.warning(
                    "[sim_nav] raw xyt looks like world coords without nav_world (compose jump %.2fm); "
                    "using raw as world (%.3f, %.3f). Set action nav_world=true or client world_frame=True.",
                    compose_jump,
                    wx,
                    wy,
                )

        meta["goal_world_pre_clamp"] = [wx, wy, wt]
        wx_c, wy_c, wt_c = self._clamp_world_nav_xyt(wx, wy, wt)
        meta["clamped"] = abs(wx_c - wx) > 1e-6 or abs(wy_c - wy) > 1e-6
        wx, wy, wt = wx_c, wy_c, wt_c
        meta["goal_world"] = [wx, wy, wt]
        meta["jump_from_base_m"] = float(np.hypot(wx - float(cur[0]), wy - float(cur[1])))
        return wx, wy, wt, meta

    def _clamp_world_nav_xyt(self, wx: float, wy: float, wt: float) -> tuple[float, float, float]:
        """Keep holonomic nav goals inside the Robocasa walkable clip (when known)."""
        rect = self._nav_world_clip_rect
        if rect is None:
            return wx, wy, wt
        x0, x1, y0, y1 = rect
        cx = min(max(float(wx), x0), x1)
        cy = min(max(float(wy), y0), y1)
        if abs(cx - wx) > 1e-6 or abs(cy - wy) > 1e-6:
            logger.warning(
                "Sim navigation: clamped world goal (%.3f, %.3f) -> (%.3f, %.3f) to stay in walkable clip %s",
                wx,
                wy,
                cx,
                cy,
                rect,
            )
        return cx, cy, float(np.arctan2(np.sin(wt), np.cos(wt)))

    def _teleport_base_world_xyt(self, wx: float, wy: float, wt: float) -> bool:
        """Teleport base to world (x,y,yaw): planar slide/slide/yaw joints or ``base_link`` free joint."""
        with self._mj_lock:
            if self._mjdata is None or self._mjmodel is None:
                return False
            if self._teleport_planar_base_world_xyt(wx, wy, wt):
                # Align arm/wheel ``ctrl`` with ``qpos`` after snapping planar joints (no extra mj_forward).
                self._sync_actuator_ctrl_from_joint_positions()
                return True
            addrs = self._base_freejoint_addrs()
            if addrs is None:
                return False
            qadr, vadr = addrs
            z = float(self._mjdata.qpos[qadr + 2])
            qw = float(np.cos(wt * 0.5))
            qz = float(np.sin(wt * 0.5))
            self._mjdata.qpos[qadr] = wx
            self._mjdata.qpos[qadr + 1] = wy
            self._mjdata.qpos[qadr + 2] = z
            self._mjdata.qpos[qadr + 3 : qadr + 7] = np.array([qw, 0.0, 0.0, qz], dtype=np.float64)
            nv = 6
            self._mjdata.qvel[vadr : vadr + nv] = 0.0
            mujoco.mj_forward(self._mjmodel, self._mjdata)
        return True

    def _zero_base_free_joint_velocity(self) -> None:
        """Zero base motion: free-joint qvel or planar slide/yaw joint velocities and velocity ctrl."""
        if self._mjdata is None or self._mjmodel is None:
            return
        addrs = self._base_freejoint_addrs()
        if addrs is not None:
            _, vadr = addrs
            v0 = int(vadr)
            self._mjdata.qvel[v0 : v0 + 6] = 0.0
            return
        names = getattr(self._spec, "planar_base_joint_names", None)
        if names:
            for jn in names:
                jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    vadr = int(self._mjmodel.jnt_dofadr[jid])
                    self._mjdata.qvel[vadr] = 0.0
        aids = self._planar_base_velocity_actuator_ids()
        if aids is not None:
            for aid in aids:
                self._mjdata.ctrl[aid] = 0.0

    def _step_base_navigation_drive(self) -> None:
        """P controller in world XY + yaw toward ``_nav_goal_world``; clears goal and sets ``at_goal`` when close."""
        if self._mjmodel is None or self._mjdata is None:
            return
        goal = self._nav_goal_world
        if goal is None:
            return
        free_addrs = self._base_freejoint_addrs()
        planar_aids = self._planar_base_velocity_actuator_ids()
        if free_addrs is None and planar_aids is None:
            self._nav_goal_world = None
            self._at_goal = True
            return

        cur = self.get_base_xyt()
        cx, cy, ct = float(cur[0]), float(cur[1]), float(cur[2])
        wx, wy, wt = float(goal[0]), float(goal[1]), float(goal[2])
        dx, dy = wx - cx, wy - cy
        dist = float(np.hypot(dx, dy))
        eth = float(np.arctan2(np.sin(wt - ct), np.cos(wt - ct)))
        if self._sim_nav_debug_enabled() and dist > 1.0:
            self._nav_drive_debug_ticks += 1
            if self._nav_drive_debug_ticks % 40 == 1:
                logger.info(
                    "[sim_nav] driving dist=%.2fm goal=(%.3f,%.3f) base=(%.3f,%.3f) v_cmd~(%.2f,%.2f)",
                    dist,
                    wx,
                    wy,
                    cx,
                    cy,
                    dx * self._nav_kp_xy,
                    dy * self._nav_kp_xy,
                )
        if dist < self._nav_tol_xy and abs(eth) < self._nav_tol_theta:
            self._nav_drive_debug_ticks = 0
            if self._sim_nav_debug_enabled():
                logger.info(
                    "[sim_nav] at_goal base=(%.3f, %.3f, %.3f) goal was=(%.3f, %.3f, %.3f)",
                    cx,
                    cy,
                    ct,
                    wx,
                    wy,
                    wt,
                )
            self._nav_goal_world = None
            self._at_goal = True
            if free_addrs is not None:
                _, vadr = free_addrs
                v0 = int(vadr)
                self._mjdata.qvel[v0 : v0 + 6] = 0.0
            elif planar_aids is not None:
                self._zero_base_free_joint_velocity()
            return

        vx = self._nav_kp_xy * dx
        vy = self._nav_kp_xy * dy
        sp = float(np.hypot(vx, vy))
        if sp > self._nav_v_max and sp > 1e-9:
            s = self._nav_v_max / sp
            vx *= s
            vy *= s
        if dist < self._nav_tol_xy * 2.0:
            vx = vy = 0.0
        wz = float(np.clip(self._nav_kp_theta * eth, -self._nav_w_max, self._nav_w_max))

        if free_addrs is not None:
            _, vadr = free_addrs
            v0 = int(vadr)
            self._mjdata.qvel[v0 : v0 + 3] = (0.0, 0.0, wz)
            self._mjdata.qvel[v0 + 3 : v0 + 6] = (vx, vy, 0.0)
        else:
            ax, ay, aw = planar_aids
            names = getattr(self._spec, "planar_base_joint_names", None)
            qxd, qyd = float(vx), float(vy)
            qdot_yaw = float(wz)
            if names and len(names) == 3:
                jn = (str(names[0]), str(names[1]), str(names[2]))
                anchor = scene_base_spawn.infer_planar_anchor_body_name(self._mjmodel, jn)
                if anchor:
                    bid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, anchor)
                    if bid >= 0:
                        R = np.asarray(self._mjdata.body(bid).xmat, dtype=np.float64).reshape(3, 3)
                        M = R[:2, :2]
                        v_world = np.array([vx, vy], dtype=np.float64)
                        try:
                            v_joint = np.linalg.solve(M, v_world)
                        except np.linalg.LinAlgError:
                            v_joint, *_ = np.linalg.lstsq(M, v_world, rcond=None)
                        qxd, qyd = float(v_joint[0]), float(v_joint[1])
                        a_axis = R[:, 2]
                        az = float(a_axis[2])
                        if abs(az) >= 0.2:
                            qdot_yaw = float(wz) / az
            self._mjdata.ctrl[ax] = qxd
            self._mjdata.ctrl[ay] = qyd
            self._mjdata.ctrl[aw] = qdot_yaw

    @override
    def handle_action(self, action: dict[str, Any]):
        if EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY in action:
            path_gt, exclude_robot, as_json = parse_ground_truth_dump_action_field(
                action[EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY]
            )
            if path_gt:
                hdr: dict[str, Any] | None = None
                if isinstance(self._environment_descriptor, dict):
                    hdr = {
                        k: self._environment_descriptor[k]
                        for k in sorted(self._environment_descriptor.keys())
                        if k in ("kind", "task", "style", "layout")
                    }

                try:
                    with self._mj_lock:
                        if self._mjmodel is None or self._mjdata is None:
                            logger.warning(
                                "mujoco_ground_truth_dump: model not loaded; cannot write %r",
                                path_gt,
                            )
                        else:
                            out = mujoco_ground_truth_write_path(
                                self._mjmodel,
                                self._mjdata,
                                dest=path_gt,
                                exclude_robot=exclude_robot,
                                robot_base_body_name=str(self._spec.base_link_name),
                                json=as_json,
                                extras=hdr,
                            )
                            logger.info(f"Wrote MuJoCo ground-truth snapshot -> {out}")
                except Exception as e:
                    logger.error("mujoco_ground_truth_dump failed for %r: %s", path_gt, e)

        if "control_mode" in action:
            self.control_mode = action["control_mode"]

        if "posture" in action:
            p = str(action["posture"])
            if p in ("navigation", "manipulation"):
                self.control_mode = p
                if self._spec.name == "stretch" and self._mjmodel is not None and self._mjdata is not None:
                    with self._mj_lock:
                        apply_stretch_posture_to_robosuite(self._spec, self._mjmodel, self._mjdata, p)
            self._at_goal = True

        has_xyt = "xyt" in action
        if has_xyt:
            self._at_goal = False
        try:
            with self._mj_lock:
                if self._mjdata is None:
                    return
                if "joint" in action:
                    joint_targets = action["joint"]
                    n_spec = len(self._spec.actuator_names)
                    if self._joint_ctrl_hold_client_pin is None or int(
                        self._joint_ctrl_hold_client_pin.shape[0]
                    ) != n_spec:
                        self._joint_ctrl_hold_client_pin = np.zeros(n_spec, dtype=np.bool_)
                    for i, aname in enumerate(self._spec.actuator_names):
                        if i < len(joint_targets):
                            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                            if aid >= 0:
                                v = float(joint_targets[i])
                                self._mjdata.ctrl[aid] = v
                                if (
                                    self._joint_ctrl_hold is not None
                                    and i < int(self._joint_ctrl_hold.shape[0])
                                ):
                                    self._joint_ctrl_hold[i] = v
                                    self._joint_ctrl_hold_client_pin[i] = True

                if has_xyt:
                    raw = np.asarray(action["xyt"], dtype=np.float64).reshape(-1)
                    if raw.size < 3:
                        return
                    init = self._initial_xyt
                    if init is None:
                        init = np.zeros(3, dtype=np.float64)
                    wx, wy, wt, nav_meta = self._resolve_nav_goal_world_xyt(action, raw, init)
                    nav_teleport = bool(action.get("nav_teleport", False)) or self._use_molmospaces_nav_teleport()
                    if nav_teleport:
                        if not self._teleport_base_world_xyt(wx, wy, wt):
                            logger.warning(
                                f"Navigation xyt={action['xyt']!r}: no free joint on base_link "
                                f"{self._spec.base_link_name!r}; cannot teleport."
                            )
                            self._log_nav_action(nav_meta, applied="teleport_failed")
                        else:
                            self._log_nav_action(nav_meta, applied="teleport")
                            after = self.get_base_xyt()
                            nav_meta["base_world_after"] = [
                                float(after[0]),
                                float(after[1]),
                                float(after[2]),
                            ]
                            self._sync_actuator_ctrl_from_joint_positions()
                        self._nav_goal_world = None
                        self._zero_base_free_joint_velocity()
                        self._at_goal = True
                    else:
                        self._zero_base_free_joint_velocity()
                        self._sync_actuator_ctrl_from_joint_positions()
                        self._nav_goal_world = np.array([wx, wy, wt], dtype=np.float64)
                        self._nav_drive_debug_ticks = 0
                        self._log_nav_action(nav_meta, applied="velocity_drive")
        except Exception as e:
            if has_xyt:
                logger.error(f"Navigation xyt={action.get('xyt')!r} failed in simulation server: {e!r}")

        if "head_to" in action and self._mjmodel is not None and self._mjdata is not None:
            ht = action["head_to"]
            if isinstance(ht, (list, tuple)) and len(ht) >= 2:
                with self._mj_lock:
                    apply_head_to_robosuite(self._spec, self._mjmodel, self._mjdata, float(ht[0]), float(ht[1]))
                    self._snapshot_spec_hold_from_ctrl()

    @override
    def get_full_observation_message(self) -> dict[str, Any]:
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        try:
            rgb, depth, K = self._primary_rgb_and_depth(primary_cam)
        except Exception as e:
            logger.warning(f"get_full_observation_message render failed ({primary_cam!r}): {e!r}")
            return None

        height, width = rgb_height_width_for_zmq(rgb)
        depth_u16 = (depth * 1000).astype(np.uint16)

        positions, _, _ = self.get_joint_state()
        xyt = self.get_base_pose()
        if xyt is None:
            xyt = np.zeros(3)

        cam_pose = self._camera_pose_world(primary_cam)

        message = {
            "rgb": compression.to_jpg(rgb),
            "depth": compression.to_jp2(depth_u16),
            "camera_K": K,
            "camera_pose": cam_pose,
            "ee_pose": np.eye(4),
            "joint": positions,
            "gps": xyt[:2],
            "compass": np.array([xyt[2]]),
            "rgb_width": width,
            "rgb_height": height,
            "control_mode": self.get_control_mode(),
            "last_motion_failed": False,
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": self._at_goal,
            "is_simulation": True,
            "lidar_points": None,
            "lidar_timestamp": None,
            EMET_ZMQ_ROBOT_ID_KEY: self._spec.name,
        }
        # Extra cameras use the shared EGL renderer; after --steps the sim thread may have stopped
        # and the GL context can be invalid — skip optional views to avoid eglMakeCurrent failures.
        allow_extra_cams = bool(getattr(self, "_running", True))
        right_name = self._stereo_right_camera_name()
        if right_name is not None and allow_extra_cams:
            try:
                rgb_r, K_r = self._primary_rgb_only_with_K(right_name)
                if rgb_r.shape[0] == rgb.shape[0] and rgb_r.shape[1] == rgb.shape[1]:
                    message["rgb_right"] = compression.to_jpg(rgb_r)
                    message["camera_K_right"] = K_r
                    message["camera_pose_right"] = self._camera_pose_world(right_name)
            except Exception as e:
                logger.debug(f"Stereo auxiliary RGB failed for {right_name}: {e!r}")
        if len(cam_names) >= 3 and allow_extra_cams:
            tertiary = cam_names[2]
            if tertiary not in (primary_cam, right_name):
                try:
                    rgb_t, K_t = self._primary_rgb_only_with_K(tertiary)
                    message["rgb_tertiary"] = compression.to_jpg(rgb_t)
                    message["camera_K_tertiary"] = K_t
                    message["camera_pose_tertiary"] = self._camera_pose_world(tertiary)
                    message["camera_name_tertiary"] = tertiary
                except Exception as e:
                    logger.debug(f"Tertiary RGB failed for {tertiary}: {e!r}")
        message = self._attach_emet_session(message)
        with self._render_lock:
            self._last_full_obs_for_servo = message
        return message

    @override
    def get_state_message(self) -> dict[str, Any]:
        if self._mjdata is None:
            return None
        q, dq, eff = self.get_joint_state()
        message = {
            "base_pose": self.get_base_pose(),
            "ee_pose": np.eye(4),
            "joint_positions": q,
            "joint_velocities": dq,
            "joint_efforts": eff,
            "control_mode": self.get_control_mode(),
            "at_goal": self._at_goal,
            "is_homed": True,
            "is_runstopped": False,
            "is_simulation": True,
            "step": self._last_step,
            EMET_ZMQ_ROBOT_ID_KEY: self._spec.name,
        }
        return self._attach_emet_session(message)

    @override
    def get_servo_message(self) -> dict[str, Any]:
        """Low-rate head RGB-D for StretchZmqClient / Rerun (resized from full observation).

        Reuses :meth:`get_full_observation_message` so we do not run a second concurrent MuJoCo
        render (``spin_send`` and ``spin_send_servo`` share one EGL ``Renderer``).
        """
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        try:
            with self._render_lock:
                full = self._last_full_obs_for_servo
            if full is None:
                return None
            rgb_full = compression.from_jpg(full["rgb"])
            raw_depth = full.get("depth")
            if raw_depth is None:
                return None
            depth_full = compression.from_jp2(raw_depth) / 1000.0
            K_full = np.asarray(full["camera_K"], dtype=np.float64)
            rgb = cv2.resize(rgb_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            if not getattr(self, "_servo_publish_error_logged", False):
                logger.warning(f"get_servo_message failed for camera {primary_cam!r}: {e!r}")
                self._servo_publish_error_logged = True
            return None

        depth_u16 = (depth * 1000).astype(np.uint16)
        q, dq, eff = self.get_joint_state()
        xyt = self.get_base_pose()
        if xyt is None:
            xyt = np.zeros(3)

        K_servo = scale_pinhole_K(K_full, rgb_full.shape[1], rgb_full.shape[0], _SERVO_RW, _SERVO_RH)
        cam_pose = full.get("camera_pose")
        if cam_pose is None:
            cam_pose = self._camera_pose_world(primary_cam)

        message = {
            "head_color_image": compression.to_jpg(rgb),
            "head_depth_image": compression.to_jp2(depth_u16),
            "head_camera_K": K_servo,
            "camera_pose": cam_pose,
            "joint_positions": q,
            "joint_velocities": dq,
            "base_pose": xyt,
            "control_mode": self.get_control_mode(),
            "step": self._last_step,
            "at_goal": self._at_goal,
        }
        return self._attach_emet_session(message)

    def _sim_loop(self) -> None:
        """Step the MuJoCo simulation at the configured rate."""
        if self._mujoco_ctrl_debug_enabled():
            self._ctrl_debug_emit_apply_logs = True
            logger.info("[mujoco_ctrl_debug] headless _sim_loop: apply logging enabled")
        while self._running:
            with self._mj_lock:
                self._step_base_navigation_drive()
                for _ in range(self._mj_substeps_per_tick):
                    self._hold_stationary_base_freejoint_if_idle()
                    self._apply_joint_ctrl_hold_to_actuators(refresh_unpinned_hold=False)
                    mujoco.mj_step(self._mjmodel, self._mjdata)
                    self._physics_steps_executed += 1
                    if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                        break
            if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                logger.info(f"MuJoCo step limit reached (--steps {self._max_sim_steps}); stopping simulation loop.")
                self._running = False
                break
            time.sleep(1 / self.simulation_rate)

    def _run_passive_viewer_main_loop(self, show_viewer_ui: bool) -> None:
        """Step physics in the same thread as ``launch_passive`` (required for a stable viewer)."""
        import mujoco.viewer

        dt = 1.0 / max(1, int(self.simulation_rate))
        try:
            with mujoco.viewer.launch_passive(
                self._mjmodel,
                self._mjdata,
                show_left_ui=show_viewer_ui,
                show_right_ui=show_viewer_ui,
            ) as viewer:
                logger.info("MuJoCo passive viewer open (close window or Ctrl+C to stop).")
                if self._mujoco_ctrl_debug_enabled():
                    self._ctrl_debug_emit_apply_logs = True
                    logger.info("[mujoco_ctrl_debug] passive viewer: apply logging enabled")
                while self._running and viewer.is_running():
                    # Keep mj_step and viewer.sync under the same lock: sync uses mj_copyDataVisual
                    # and must not overlap Renderer / mj_forward on other ZMQ threads.
                    with self._mj_lock:
                        self._step_base_navigation_drive()
                        for _ in range(self._mj_substeps_per_tick):
                            self._hold_stationary_base_freejoint_if_idle()
                            self._apply_joint_ctrl_hold_to_actuators(refresh_unpinned_hold=False)
                            mujoco.mj_step(self._mjmodel, self._mjdata)
                            self._physics_steps_executed += 1
                            if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                                break
                        viewer.sync()
                    if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                        logger.info(f"MuJoCo step limit reached (--steps {self._max_sim_steps}); closing viewer loop.")
                        self._running = False
                        break
                    time.sleep(dt)
        except Exception as e:
            logger.warning(
                f"MuJoCo passive viewer failed ({e!r}); falling back to headless background stepping. "
                "Use a desktop session with DISPLAY set, or run with --headless."
            )
            self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._sim_thread.start()
            while self._running:
                time.sleep(dt)
            return
        self._running = False

    def start(
        self,
        robocasa: bool = False,
        headless: bool = True,
        show_viewer_ui: bool = False,
        **kwargs,
    ) -> None:
        self._load_model()
        self._running = True
        pl_debug = robosuite_post_load_debug_enabled(self._debug_molmospaces_spawn)
        if self._mjmodel is not None and self._mjdata is not None:
            with self._mj_lock:
                if pl_debug:
                    log_post_load_diagnostics(
                        logger,
                        model=self._mjmodel,
                        data=self._mjdata,
                        spec=self._spec,
                        stage="after_load",
                        base_body_name=self._spec.base_link_name,
                    )
                home_applied = apply_home_keyframe_preserving_base(
                    self._mjmodel,
                    self._mjdata,
                    base_body_name=self._spec.base_link_name,
                )
                if not home_applied:
                    planar = getattr(self._spec, "planar_base_joint_names", None)
                    if planar is not None and len(planar) == 3:
                        home_applied = apply_home_keyframe_preserving_planar_base(
                            self._mjmodel,
                            self._mjdata,
                            planar_joint_names=(
                                str(planar[0]),
                                str(planar[1]),
                                str(planar[2]),
                            ),
                            base_body_name=self._spec.base_link_name,
                        )
                if home_applied:
                    logger.info("Applied MJCF keyframe 'home' (preserved robot base pose).")
                    if not self._seed_joint_ctrl_hold_from_keyframe("home"):
                        self._preserve_joint_ctrl_hold_from_ctrl()
                    else:
                        self._apply_joint_ctrl_hold_to_actuators(refresh_unpinned_hold=False)
                    mujoco.mj_forward(self._mjmodel, self._mjdata)
        self._stabilize_physics_state_after_load()
        if self._molmospaces_autoplace_snap_qpos0:
            with self._mj_lock:
                self._restore_merged_base_freejoint_from_qpos0()
                self._preserve_joint_ctrl_hold_from_ctrl()
                mujoco.mj_forward(self._mjmodel, self._mjdata)
                # ``restore`` snaps the free base out of whatever pose stabilize drifted to; that jump
                # can leave arms / contacts inconsistent with the spawn frame. Run a longer settle so
                # PD + contacts re-equilibrate before ZMQ clients observe the scene (iTHOR + rby1).
                for _ in range(120):
                    self._apply_joint_ctrl_hold_to_actuators(refresh_unpinned_hold=False)
                    mujoco.mj_step(self._mjmodel, self._mjdata)
                # Avoid a hard velocity zero here: it injects impulses while contacts are loaded and
                # can excite the torso PD chain before the realtime loop starts.
                self._preserve_joint_ctrl_hold_from_ctrl()
                mujoco.mj_forward(self._mjmodel, self._mjdata)
                if molmospaces_spawn.resettle_free_base_z_at_current_xy_preserving_yaw(
                    self._mjmodel,
                    self._mjdata,
                    base_body_name=self._spec.base_link_name,
                    robot_key=self._spec.name,
                ):
                    logger.info(
                        "MolmoSpaces post-settle: re-aligned base height to floor at fixed (x,y) "
                        "(torso/arms match home after dynamics)."
                    )
                    self._preserve_joint_ctrl_hold_from_ctrl()
                    mujoco.mj_forward(self._mjmodel, self._mjdata)
                update_robot_qpos0_from_data(self._mjmodel, self._mjdata, self._spec)
        elif self._planar_autoplace_snap_qpos0:
            with self._mj_lock:
                jn = getattr(self._spec, "planar_base_joint_names", None)
                wxyt = self._planar_autoplace_world_xyt
                reapply_ok = False
                if wxyt is not None and np.asarray(wxyt).size >= 3 and jn is not None and len(jn) == 3:
                    w = np.asarray(wxyt, dtype=np.float64).reshape(-1)[:3]
                    reapply_ok = bool(
                        scene_base_spawn.write_planar_base_xyt(
                            self._mjmodel,
                            self._mjdata,
                            joint_names=(str(jn[0]), str(jn[1]), str(jn[2])),
                            world_x=float(w[0]),
                            world_y=float(w[1]),
                            world_yaw=float(w[2]),
                            base_body_name=self._spec.base_link_name,
                        )
                    )
                if not reapply_ok:
                    self._restore_planar_base_from_qpos0()
                update_robot_qpos0_from_data(self._mjmodel, self._mjdata, self._spec)
                self._preserve_joint_ctrl_hold_from_ctrl()
        if self._mjmodel is not None and self._mjdata is not None:
            with self._mj_lock:
                self._snapshot_stationary_base_freejoint_pose()
        if pl_debug and self._mjmodel is not None and self._mjdata is not None:
            with self._mj_lock:
                log_post_load_diagnostics(
                    logger,
                    model=self._mjmodel,
                    data=self._mjdata,
                    spec=self._spec,
                    stage="after_stabilize",
                    base_body_name=self._spec.base_link_name,
                )
                mx = probe_max_qvel_unforced_steps(
                    self._mjmodel,
                    self._mjdata,
                    n_steps=24,
                    sync_ctrl=self._sync_actuator_ctrl_from_joint_positions,
                    before_physics_step=lambda: self._apply_joint_ctrl_hold_to_actuators(
                        refresh_unpinned_hold=False
                    ),
                )
                if mx is not None:
                    logger.info(f"[robosuite_load] post-stabilize 24-step probe max|qvel|={mx:.4f}")
        self._initial_xyt = self.get_base_xyt()
        self._nav_goal_world = None
        self._at_goal = True
        spawn_floor_map = self._compute_robocasa_spawn_floor_map() if robocasa else None
        self._nav_world_clip_rect = None
        if spawn_floor_map is not None:
            eroded = spawn_floor_map.get("clip_eroded_xy")
            if isinstance(eroded, (list, tuple)) and len(eroded) == 4:
                self._nav_world_clip_rect = tuple(float(v) for v in eroded)
        self._emet_session = self._build_emet_session(
            robocasa=robocasa,
            spawn_floor_map=spawn_floor_map,
        )
        self._log_nav_startup_banner()
        if self._is_molmospaces_session() and not molmospaces_nav_teleport_enabled():
            log.info("MolmoSpaces navigation: wheel/goal drive (EMET_MOLMOSPACES_NAV_TELEPORT=0)")

        # Print scene summary before any rendering (so it appears in headless / no-DISPLAY runs)
        summary = self.get_scene_summary()
        print(summary, flush=True)
        logger.info("\n" + summary)

        if self._mjmodel is not None and self._mjdata is not None:
            with self._mj_lock:
                if self._mujoco_ctrl_debug_enabled():
                    self._ctrl_debug_periodic_counter = 0
                    self._ctrl_debug_initial_logs_remaining = max(
                        48, int(self._mj_substeps_per_tick) * 8
                    )
                self._preserve_joint_ctrl_hold_from_ctrl()
                mujoco.mj_forward(self._mjmodel, self._mjdata)
                if self._mujoco_ctrl_debug_enabled():
                    logger.info(
                        "[mujoco_ctrl_debug] pre_threads (set EMET_MUJOCO_CTRL_DEBUG=0 to disable): "
                        f"nu={int(self._mjmodel.nu)} nv={int(self._mjmodel.nv)} "
                        f"timestep={float(self._mjmodel.opt.timestep):g}s "
                        f"substeps_per_tick={int(self._mj_substeps_per_tick)} "
                        f"stationary={type(self._mujoco_stationary).__name__} "
                        f"| {self._mujoco_ctrl_debug_summary()}"
                    )
                    logger.info(f"[mujoco_ctrl_debug] pre_threads_pd {self._mujoco_ctrl_debug_pd_tracking()}")
                    logger.info(
                        f"[mujoco_ctrl_debug] pre_threads_base {self._mujoco_ctrl_debug_base_stability()}"
                    )
                    logger.info(
                        "[mujoco_ctrl_debug] set EMET_MUJOCO_CTRL_DEBUG_VERBOSE=1 for full lines "
                        "(Δq0 vs qpos0, dq, F_act, τ_dof, τ_con) on all torso joints + arm1 pair."
                    )

        super().start()

        self._sim_thread: threading.Thread | None = None
        use_viewer = not headless

        logger.info(
            f"RobosuiteZmqServer started for robot '{self._spec.name}' "
            f"({self._spec.dof} DOF, {len(self._spec.actuator_names)} actuators)"
        )
        if self._mujoco_ctrl_debug_enabled():
            mode = "passive_viewer_main_loop" if use_viewer else "headless_sim_thread"
            logger.info(
                f"[mujoco_ctrl_debug] physics mode={mode!r}; first "
                f"{self._ctrl_debug_initial_logs_remaining} apply cycles at INFO, then every 240 applies; "
                "second line: torso PD; third line: base rpy/tilt (fall-over). "
                "Unpinned hold: no per-tick q→hold refresh in sim loop."
            )
        print("Server running. Press Ctrl+C to stop.", flush=True)

        if use_viewer:
            self._run_passive_viewer_main_loop(show_viewer_ui)
        else:
            self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._sim_thread.start()
            while self._running:
                time.sleep(1 / self.simulation_rate)

    def stop(self):
        self._close_renderers()
        self._running = False
        self._done = True
        p = self._scene_disk_path
        if p and Path(p).is_file() and Path(p).name.startswith("molmospaces_merged_"):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        self._scene_disk_path = None
