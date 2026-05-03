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
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
)
from emet.robots.base import RobotSpec
from emet.simulation import molmospaces_spawn
from emet.simulation.head_look_action import apply_head_to_robosuite
from emet.utils.geometry import xyt_global_to_base

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
        # World-frame (x, y, yaw) holonomic drive goal for the base free joint (velocity before mj_step).
        self._nav_goal_world: np.ndarray | None = None
        self._nav_tol_xy = 0.07
        self._nav_tol_theta = 0.15
        self._nav_kp_xy = 0.95
        self._nav_kp_theta = 2.2
        self._nav_v_max = 0.42
        self._nav_w_max = 0.95
        self._render_lock = threading.Lock()
        self._primary_renderer: Any | None = None
        self._max_sim_steps: int | None = (
            int(max_sim_steps) if max_sim_steps is not None and int(max_sim_steps) > 0 else None
        )
        self._debug_molmospaces_spawn = debug_molmospaces_spawn
        self._scene_disk_path: str | None = (
            str(scene_disk_path).strip() if scene_disk_path and str(scene_disk_path).strip() else None
        )
        self._physics_steps_executed = 0

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
        with self._mj_lock:
            mujoco.mj_forward(self._mjmodel, self._mjdata)
            self._molmospaces_autoplace_free_base_after_load()

    def _want_molmospaces_spawn_heuristic(self) -> bool:
        """True when we merged a MolmoSpaces house + mobile base (needs placement away from origin)."""
        return molmospaces_spawn.want_molmospaces_autoplace(
            environment=self._environment_descriptor,
            scene_source_basename=self._scene_source_basename,
        )

    def _molmospaces_autoplace_free_base_after_load(self) -> None:
        """Move merged MolmoSpaces + mobile robot off origin when the base starts inside scene clutter."""
        if not self._want_molmospaces_spawn_heuristic():
            return
        if self._mjmodel is None or self._mjdata is None:
            return
        if self._base_freejoint_addrs() is None:
            return
        base_name = self._spec.base_link_name
        if self._debug_molmospaces_spawn:
            logger.info(
                f"MolmoSpaces spawn debug: scene_source_basename={self._scene_source_basename!r} "
                f"environment={self._environment_descriptor!r} base_body_name={base_name!r}"
            )
        try:
            placed = molmospaces_spawn.find_molmospaces_freejoint_xyz(
                self._mjmodel,
                self._mjdata,
                base_body_name=base_name,
                scene_label=self._scene_source_basename,
                merged_mjcf_path=self._scene_disk_path,
                environment=self._environment_descriptor,
            )
        except Exception as e:
            logger.warning(f"MolmoSpaces base autoplace skipped ({e!r}).")
            return
        if placed is None:
            if self._debug_molmospaces_spawn:
                logger.info("MolmoSpaces base autoplace: find_molmospaces_freejoint_xyz returned None (see spawn debug lines above).")
            return
        x, y, z = placed
        logger.info(
            f"MolmoSpaces base autoplace: moved free joint on {base_name!r} to "
            f"({x:.3f}, {y:.3f}, {z:.3f}) to avoid origin clutter."
        )
        if self._debug_molmospaces_spawn:
            try:
                mujoco.mj_forward(self._mjmodel, self._mjdata)
                lines = molmospaces_spawn.format_spawn_contact_report(
                    self._mjmodel,
                    self._mjdata,
                    base_body_name=base_name,
                    floor_geom_name="floor",
                    max_lines=50,
                    dist_report_threshold=0.15,
                )
                for ln in lines:
                    logger.info(f"[molmospaces_spawn/post-place] {ln}")
                for ln in molmospaces_spawn.format_spawn_floor_alignment_report(
                    self._mjmodel,
                    self._mjdata,
                    base_body_name=base_name,
                    floor_geom_name="floor",
                    xy=(float(x), float(y)),
                ):
                    logger.info(f"[molmospaces_spawn/post-place] {ln}")
            except Exception as e:
                logger.warning(f"MolmoSpaces spawn debug contact report failed: {e!r}")
        # Copy placed free-joint pose into qpos0 so resets use autoplace (Python MjModel has no qvel0).
        addrs = self._base_freejoint_addrs()
        if addrs is not None:
            qadr = int(addrs[0])
            self._mjmodel.qpos0[qadr : qadr + 7] = self._mjdata.qpos[qadr : qadr + 7]

    def _build_emet_session(self, *, robocasa: bool) -> dict[str, Any]:
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
            "teleport_base": False,
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
        except Exception:
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
        except Exception:
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

    def _primary_rgb_and_depth(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        """RGB + depth using one ``Renderer`` under ``_render_lock`` (avoids EGL 0x502 on extra contexts)."""
        cam = self._camera_for_renderer(camera_name)
        with self._mj_lock:
            with self._render_lock:
                if self._primary_renderer is None:
                    self._primary_renderer = mujoco.Renderer(self._mjmodel, _PRIMARY_RH, _PRIMARY_RW)
                renderer = self._primary_renderer
                renderer.update_scene(self._mjdata, camera=cam)
                rgb = cast(np.ndarray, renderer.render())
                rgb = np.asarray(rgb, dtype=np.uint8).copy()
                rgb = np.flipud(rgb).copy()
                renderer.enable_depth_rendering()
                try:
                    renderer.update_scene(self._mjdata, camera=cam)
                    depth = cast(np.ndarray, renderer.render())
                    depth = np.asarray(depth, dtype=np.float32).copy()
                    depth = np.flipud(depth).copy()
                finally:
                    renderer.disable_depth_rendering()
                return rgb, depth

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

    def _sync_actuator_ctrl_from_joint_positions(self) -> None:
        """Set ``ctrl`` so actuators command the current pose (and zero for swerve *velocity* wheels).

        After MolmoSpaces autoplace or any MJCF merge, ``ctrl`` defaults to 0 while articulated
        ``qpos`` may not; the first ``mj_step`` then applies large PD errors and the robot can
        collapse. Velocity wheel actuators (Galaxea / rby1 ``wheel*``) must use ``ctrl=0`` at rest.
        """
        if self._mjmodel is None or self._mjdata is None:
            return
        n = min(len(self._spec.actuator_names), len(self._spec.joint_names))
        for i in range(n):
            jname = self._spec.joint_names[i]
            aname = self._spec.actuator_names[i]
            jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if jid < 0 or aid < 0:
                continue
            qadr = int(self._mjmodel.jnt_qposadr[jid])
            if aname.startswith("wheel"):
                self._mjdata.ctrl[aid] = 0.0
            else:
                self._mjdata.ctrl[aid] = float(self._mjdata.qpos[qadr])

    def _stabilize_physics_state_after_load(self) -> None:
        """Zero all velocities, align actuators with ``qpos``, and run a few dynamics steps."""
        if self._mjmodel is None or self._mjdata is None:
            return
        with self._mj_lock:
            self._mjdata.qvel.fill(0.0)
            self._sync_actuator_ctrl_from_joint_positions()
            mujoco.mj_forward(self._mjmodel, self._mjdata)
            for _ in range(8):
                mujoco.mj_step(self._mjmodel, self._mjdata)
            self._mjdata.qvel.fill(0.0)
            self._sync_actuator_ctrl_from_joint_positions()
            mujoco.mj_forward(self._mjmodel, self._mjdata)

    def _camera_pose_world(self, camera_name: str) -> np.ndarray:
        """4x4 camera-to-world transform for depth unprojection and map ``_visited`` updates.

        Previously this server sent ``np.eye(4)``, which left DynaMem with geometry near the origin
        while the robot reported ``gps`` elsewhere, so ``_navigable = ~obstacle & explored`` had no
        overlap at the base and exploration planning failed immediately.
        """
        if self._mjmodel is None or self._mjdata is None:
            return np.eye(4, dtype=np.float64)
        with self._mj_lock:
            cam_id = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                return np.eye(4, dtype=np.float64)
            R = np.asarray(self._mjdata.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
            pos = np.asarray(self._mjdata.cam_xpos[cam_id], dtype=np.float64).reshape(3)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
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

    def _teleport_base_world_xyt(self, wx: float, wy: float, wt: float) -> bool:
        """Teleport ``base_link`` free joint to world (x,y,yaw); preserve height and zero base twist."""
        with self._mj_lock:
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
        """Zero the 6 velocity dofs of the base free joint (world-frame ang then lin; see MuJoCo free joint)."""
        addrs = self._base_freejoint_addrs()
        if addrs is None or self._mjdata is None:
            return
        _, vadr = addrs
        v0 = int(vadr)
        self._mjdata.qvel[v0 : v0 + 6] = 0.0

    def _step_base_navigation_drive(self) -> None:
        """P controller in world XY + yaw toward ``_nav_goal_world``; clears goal and sets ``at_goal`` when close."""
        if self._mjmodel is None or self._mjdata is None:
            return
        goal = self._nav_goal_world
        if goal is None:
            return
        addrs = self._base_freejoint_addrs()
        if addrs is None:
            self._nav_goal_world = None
            self._at_goal = True
            return
        _, vadr = addrs
        v0 = int(vadr)
        cur = self.get_base_xyt()
        cx, cy, ct = float(cur[0]), float(cur[1]), float(cur[2])
        wx, wy, wt = float(goal[0]), float(goal[1]), float(goal[2])
        dx, dy = wx - cx, wy - cy
        dist = float(np.hypot(dx, dy))
        eth = float(np.arctan2(np.sin(wt - ct), np.cos(wt - ct)))
        if dist < self._nav_tol_xy and abs(eth) < self._nav_tol_theta:
            self._mjdata.qvel[v0 : v0 + 6] = 0.0
            self._nav_goal_world = None
            self._at_goal = True
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
        # MuJoCo free joint qvel: (wx, wy, wz) angular then (vx, vy, vz) linear, world frame.
        self._mjdata.qvel[v0 : v0 + 3] = (0.0, 0.0, wz)
        self._mjdata.qvel[v0 + 3 : v0 + 6] = (vx, vy, 0.0)

    @override
    def handle_action(self, action: dict[str, Any]):
        if "control_mode" in action:
            self.control_mode = action["control_mode"]

        has_xyt = "xyt" in action
        if has_xyt:
            self._at_goal = False
        try:
            with self._mj_lock:
                if self._mjdata is None:
                    return
                if "joint" in action:
                    joint_targets = action["joint"]
                    for i, aname in enumerate(self._spec.actuator_names):
                        if i < len(joint_targets):
                            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                            if aid >= 0:
                                self._mjdata.ctrl[aid] = joint_targets[i]

                if has_xyt:
                    raw = np.asarray(action["xyt"], dtype=np.float64).reshape(-1)
                    if raw.size < 3:
                        return
                    init = self._initial_xyt
                    if init is None:
                        init = np.zeros(3, dtype=np.float64)
                    relative = bool(action.get("nav_relative", False))
                    if relative:
                        cur = self.get_base_xyt()
                        dx, dy, dt = float(raw[0]), float(raw[1]), float(raw[2])
                        ct = float(cur[2])
                        wx = cur[0] + np.cos(ct) * dx - np.sin(ct) * dy
                        wy = cur[1] + np.sin(ct) * dx + np.cos(ct) * dy
                        wt = float(np.arctan2(np.sin(cur[2] + dt), np.cos(cur[2] + dt)))
                    else:
                        world = self._spawn_rel_xyt_to_world(raw[:3], init)
                        wx, wy, wt = float(world[0]), float(world[1]), float(world[2])
                    nav_teleport = bool(action.get("nav_teleport", False))
                    if nav_teleport:
                        if not self._teleport_base_world_xyt(wx, wy, wt):
                            logger.warning(
                                f"Navigation xyt={action['xyt']!r}: no free joint on base_link "
                                f"{self._spec.base_link_name!r}; cannot teleport."
                            )
                        else:
                            logger.info(f"Sim navigation (teleport): base at x={wx:.3f} y={wy:.3f} theta={wt:.3f}.")
                        self._nav_goal_world = None
                        self._zero_base_free_joint_velocity()
                        self._at_goal = True
                    else:
                        self._nav_goal_world = np.array([wx, wy, wt], dtype=np.float64)
                        logger.info(
                            f"Sim navigation: driving toward x={wx:.3f} y={wy:.3f} theta={wt:.3f} "
                            f"(set action nav_teleport=true for instant snap)."
                        )
        except Exception as e:
            if has_xyt:
                logger.error(f"Navigation xyt={action.get('xyt')!r} failed in simulation server: {e!r}")

        if "head_to" in action and self._mjmodel is not None and self._mjdata is not None:
            ht = action["head_to"]
            if isinstance(ht, (list, tuple)) and len(ht) >= 2:
                with self._mj_lock:
                    apply_head_to_robosuite(self._spec, self._mjmodel, self._mjdata, float(ht[0]), float(ht[1]))

    @override
    def get_full_observation_message(self) -> dict[str, Any]:
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        try:
            rgb, depth = self._primary_rgb_and_depth(primary_cam)
        except Exception:
            return None

        width, height = rgb.shape[1], rgb.shape[0]
        depth_u16 = (depth * 1000).astype(np.uint16)

        positions, _, _ = self.get_joint_state()
        xyt = self.get_base_pose()
        if xyt is None:
            xyt = np.zeros(3)

        K = self._get_camera_K(primary_cam, width, height)
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
        return self._attach_emet_session(message)

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
            "step": self._last_step,
            EMET_ZMQ_ROBOT_ID_KEY: self._spec.name,
        }
        return self._attach_emet_session(message)

    @override
    def get_servo_message(self) -> dict[str, Any]:
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        try:
            rgb_full, depth_full = self._primary_rgb_and_depth(primary_cam)
            rgb = cv2.resize(rgb_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None

        depth_u16 = (depth * 1000).astype(np.uint16)
        q, dq, eff = self.get_joint_state()
        xyt = self.get_base_pose()
        if xyt is None:
            xyt = np.zeros(3)

        message = {
            "head_color_image": compression.to_jpg(rgb),
            "head_depth_image": compression.to_jp2(depth_u16),
            "head_camera_K": self._get_camera_K(primary_cam, 320, 240),
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
        while self._running:
            with self._mj_lock:
                self._step_base_navigation_drive()
                mujoco.mj_step(self._mjmodel, self._mjdata)
            self._physics_steps_executed += 1
            if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                logger.info(
                    f"MuJoCo step limit reached (--steps {self._max_sim_steps}); stopping simulation loop."
                )
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
                while self._running and viewer.is_running():
                    # Keep mj_step and viewer.sync under the same lock: sync uses mj_copyDataVisual
                    # and must not overlap Renderer / mj_forward on other ZMQ threads.
                    with self._mj_lock:
                        self._step_base_navigation_drive()
                        mujoco.mj_step(self._mjmodel, self._mjdata)
                        viewer.sync()
                    self._physics_steps_executed += 1
                    if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                        logger.info(
                            f"MuJoCo step limit reached (--steps {self._max_sim_steps}); closing viewer loop."
                        )
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
        self._stabilize_physics_state_after_load()
        self._initial_xyt = self.get_base_xyt()
        self._nav_goal_world = None
        self._at_goal = True
        self._emet_session = self._build_emet_session(robocasa=robocasa)

        # Print scene summary before any rendering (so it appears in headless / no-DISPLAY runs)
        summary = self.get_scene_summary()
        print(summary, flush=True)
        logger.info("\n" + summary)

        super().start()

        self._sim_thread: threading.Thread | None = None
        use_viewer = not headless

        logger.info(
            f"RobosuiteZmqServer started for robot '{self._spec.name}' "
            f"({self._spec.dof} DOF, {len(self._spec.actuator_names)} actuators)"
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
