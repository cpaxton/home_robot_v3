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
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
)
from emet.robots.base import RobotSpec
from emet.simulation import molmospaces_spawn, scene_base_spawn
from emet.simulation.head_look_action import apply_head_to_robosuite
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
        # World-frame (x, y, yaw) holonomic drive goal (velocity before mj_step): either a base **free**
        # joint or **planar** slide_X / slide_Y / hinge_yaw velocity actuators (``planar_base_joint_names``).
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
        # After MolmoSpaces autoplace, ``qpos0`` holds the chosen free-joint pose; see
        # :meth:`_restore_merged_base_freejoint_from_qpos0` after physics stabilize.
        self._molmospaces_autoplace_snap_qpos0 = False
        # After Robocasa planar autoplace, planar joint ``qpos0`` holds chosen (x, y, yaw).
        self._planar_autoplace_snap_qpos0 = False
        # World SE(2) chosen by planar autoplace — reapplied after :meth:`_stabilize_physics_state_after_load`
        # because fixture dynamics can shift the mount while planar qpos stay nearly unchanged.
        self._planar_autoplace_world_xyt: np.ndarray | None = None
        # Actuator indices for :attr:`RobotSpec.planar_base_joint_names` (velocity motors); filled lazily.
        self._planar_base_actuator_ids_cache: tuple[int, int, int] | None = None

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
        self._planar_base_actuator_ids_cache = None

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
                logger.info(
                    "MolmoSpaces base autoplace: find_molmospaces_freejoint_xyz returned None (see spawn debug lines above)."
                )
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
            self._molmospaces_autoplace_snap_qpos0 = True

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
            )
        except Exception as e:
            logger.warning(f"Robocasa planar autoplace skipped ({e!r}).")
            return
        if placed is None:
            logger.info(
                "Robocasa planar autoplace: no safer (x,y,yaw) found; keeping MJCF default base pose."
            )
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

    def _restore_merged_base_freejoint_from_qpos0(self) -> None:
        """Put ``base_link`` free joint back to ``qpos0`` after :meth:`_stabilize_physics_state_after_load`.

        Stabilize runs a few ``mj_step`` calls with PD actuators synced to ``qpos``. For a floating
        base that can **drift** the robot away from the MolmoSpaces spawn chosen from occupancy,
        while logs and ``qpos0`` still show the intended pose — so the viewer no longer matches the
        top-down map. Restoring the 7 free-joint coordinates from ``qpos0`` preserves spawn XY/Z.
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

    def _sync_actuator_ctrl_from_joint_positions(self) -> None:
        """Set ``ctrl`` so actuators command the current pose (and zero for swerve *velocity* wheels).

        After MolmoSpaces autoplace or any MJCF merge, ``ctrl`` defaults to 0 while articulated
        ``qpos`` may not; the first ``mj_step`` then applies large PD errors and the robot can
        collapse. Velocity wheel actuators (Galaxea / rby1 ``wheel*``) must use ``ctrl=0`` at rest.
        """
        if self._mjmodel is None or self._mjdata is None:
            return
        n = min(len(self._spec.actuator_names), len(self._spec.joint_names))
        planar = getattr(self._spec, "planar_base_joint_names", None) or ()
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
            elif jname in planar:
                # Velocity actuators on the planar base: ``ctrl`` is target joint velocity, not position.
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
            # After Robocasa planar autoplace, ``mj_forward`` here (post-actuator sync) can corrupt
            # the slide chain: ``base_link`` xpos collapses onto ``base_root`` while planar qpos are
            # unchanged — only ``mj_fwdPosition`` is safe. Skip dynamics integration regardless.
            # Post–planar-autoplace, avoid ``mj_forward``/``mj_fwdPosition`` after actuator sync: with
            # velocity motors on the base and a freshly chosen ``qpos``, the full forward pass can put
            # ``base_link`` on ``base_root`` while planar joint values stay fixed (MuJoCo 3.x in this merge).
            if self._planar_autoplace_snap_qpos0:
                return
            mujoco.mj_forward(self._mjmodel, self._mjdata)
            for _ in range(8):
                mujoco.mj_step(self._mjmodel, self._mjdata)
            self._mjdata.qvel.fill(0.0)
            self._sync_actuator_ctrl_from_joint_positions()
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

    def _teleport_base_world_xyt(self, wx: float, wy: float, wt: float) -> bool:
        """Teleport base to world (x,y,yaw): planar slide/slide/yaw joints or ``base_link`` free joint."""
        with self._mj_lock:
            if self._mjdata is None or self._mjmodel is None:
                return False
            if self._teleport_planar_base_world_xyt(wx, wy, wt):
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
        if dist < self._nav_tol_xy and abs(eth) < self._nav_tol_theta:
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
            # MuJoCo free joint qvel: (wx, wy, wz) angular then (vx, vy, vz) linear, world frame.
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
                                f"Navigation xyt={action['xyt']!r}: cannot teleport base "
                                f"(no free joint on {self._spec.base_link_name!r} and no planar_base_joint_names / "
                                f"matching MJCF actuators)."
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
            rgb, depth, K = self._primary_rgb_and_depth(primary_cam)
        except Exception:
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
        right_name = self._stereo_right_camera_name()
        if right_name is not None:
            try:
                rgb_r, K_r = self._primary_rgb_only_with_K(right_name)
                if rgb_r.shape[0] == rgb.shape[0] and rgb_r.shape[1] == rgb.shape[1]:
                    message["rgb_right"] = compression.to_jpg(rgb_r)
                    message["camera_K_right"] = K_r
                    message["camera_pose_right"] = self._camera_pose_world(right_name)
            except Exception:
                logger.debug("Stereo auxiliary RGB failed for %s", right_name, exc_info=True)
        if len(cam_names) >= 3:
            tertiary = cam_names[2]
            if tertiary not in (primary_cam, right_name):
                try:
                    rgb_t, K_t = self._primary_rgb_only_with_K(tertiary)
                    message["rgb_tertiary"] = compression.to_jpg(rgb_t)
                    message["camera_K_tertiary"] = K_t
                    message["camera_pose_tertiary"] = self._camera_pose_world(tertiary)
                    message["camera_name_tertiary"] = tertiary
                except Exception:
                    logger.debug("Tertiary RGB failed for %s", tertiary, exc_info=True)
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
            rgb_full, depth_full, K_full = self._primary_rgb_and_depth(primary_cam)
            rgb = cv2.resize(rgb_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None

        depth_u16 = (depth * 1000).astype(np.uint16)
        q, dq, eff = self.get_joint_state()
        xyt = self.get_base_pose()
        if xyt is None:
            xyt = np.zeros(3)

        K_servo = scale_pinhole_K(K_full, rgb_full.shape[1], rgb_full.shape[0], _SERVO_RW, _SERVO_RH)

        message = {
            "head_color_image": compression.to_jpg(rgb),
            "head_depth_image": compression.to_jp2(depth_u16),
            "head_camera_K": K_servo,
            # Same OpenCV camera-to-world convention as full observations (``camera_pose``); required for
            # Rerun head-camera transform + DynaMem when the client only consumes the servo socket.
            "camera_pose": self._camera_pose_world(primary_cam),
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
                while self._running and viewer.is_running():
                    # Keep mj_step and viewer.sync under the same lock: sync uses mj_copyDataVisual
                    # and must not overlap Renderer / mj_forward on other ZMQ threads.
                    with self._mj_lock:
                        self._step_base_navigation_drive()
                        mujoco.mj_step(self._mjmodel, self._mjdata)
                        viewer.sync()
                    self._physics_steps_executed += 1
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
        self._stabilize_physics_state_after_load()
        if self._molmospaces_autoplace_snap_qpos0:
            with self._mj_lock:
                self._restore_merged_base_freejoint_from_qpos0()
                self._sync_actuator_ctrl_from_joint_positions()
                mujoco.mj_forward(self._mjmodel, self._mjdata)
        elif self._planar_autoplace_snap_qpos0:
            with self._mj_lock:
                jn = getattr(self._spec, "planar_base_joint_names", None)
                wxyt = self._planar_autoplace_world_xyt
                reapply_ok = False
                if (
                    wxyt is not None
                    and np.asarray(wxyt).size >= 3
                    and jn is not None
                    and len(jn) == 3
                ):
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
                else:
                    for jname in jn:
                        jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, str(jname))
                        if jid >= 0:
                            qadr = int(self._mjmodel.jnt_qposadr[jid])
                            self._mjmodel.qpos0[qadr] = float(self._mjdata.qpos[qadr])
                self._sync_actuator_ctrl_from_joint_positions()
                # ``write_planar_base_xyt`` already ends with ``mj_forward``. Do not call
                # ``mj_forward`` or ``mj_fwdPosition`` here: ``_sync_actuator_ctrl_from_joint_positions``
                # only touches ``ctrl`` (zeros velocity actuators). A follow-up ``mj_fwdPosition`` has
                # been observed to rewrite ``body_xpos`` without changing ``qpos``, desyncing FK and
                # breaking :meth:`get_base_xyt`; full ``mj_forward`` after sync can likewise corrupt
                # the planar slide chain for this merged MJCF.
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
