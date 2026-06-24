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

"""Robot-agnostic ZMQ client driven by RobotSpec.

This client communicates with a MuJoCo ZMQ server using the same protocol as
StretchZmqClient, but derives joint indexing and DOF from a RobotSpec rather
than hardcoding Stretch-specific constants.
"""

import os
import sys
import threading
import time
import timeit
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import zmq

import emet.utils.compression as compression
from emet.core.interfaces import ContinuousNavigationAction, Observations
from emet.core.parameters import Parameters, get_parameters
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.core.zmq_protocol import (
    EMET_ZMQ_ROBOT_ID_KEY,
    build_mujoco_ground_truth_dump_action,
    emet_session_cache_update,
    read_emet_robot_id_from_message_or_session,
    read_emet_session,
    robot_ids_match,
)
from emet.motion import constants as motion_constants
from emet.robots.base import RobotSpec
from emet.robots.spec_robot_model import SpecRobotModel
from emet.simulation.env_flags import env_sim_nav_teleport, warn_sim_nav_env_flags
from emet.utils.image import align_camera_matrix_to_image_size
from emet.utils.logger import Logger
from emet.utils.memory import lookup_address

logger = Logger(__name__)


def _decode_jpg_field(msg: dict[str, Any], *keys: str) -> np.ndarray | None:
    """Return RGB uint8 array from the first present JPEG-compressed ZMQ field."""
    for key in keys:
        buf = msg.get(key)
        if buf is None:
            continue
        try:
            return compression.from_jpg(buf)
        except Exception:
            continue
    return None


def _align_camera_k_to_rgb(
    camera_K: np.ndarray | None,
    rgb: np.ndarray | None,
) -> np.ndarray | None:
    """Scale ``camera_info`` K to match decoded RGB when stream resolution differs."""
    if camera_K is None or rgb is None or getattr(rgb, "ndim", 0) != 3:
        return camera_K
    k = np.asarray(camera_K, dtype=np.float64).reshape(3, 3)
    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    calib_w = max(1, int(round(2.0 * float(k[0, 2]) + 1.0)))
    calib_h = max(1, int(round(2.0 * float(k[1, 2]) + 1.0)))
    if calib_w == iw and calib_h == ih:
        return k
    return align_camera_matrix_to_image_size(
        k,
        calib_height=calib_h,
        calib_width=calib_w,
        image_height=ih,
        image_width=iw,
    )


def _first_present(msg: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        val = msg.get(key)
        if val is not None:
            return val
    return None


def enrich_zmq_observation_ee_fields(msg: dict[str, Any]) -> None:
    """Fill ``ee_rgb`` / ``ee_camera_*`` on a ZMQ dict (full obs or servo; in-place)."""
    if msg.get("ee_rgb") is None:
        ee_rgb = _decode_jpg_field(
            msg,
            "ee_cam/image",
            "ee_cam/color_image",
            "rgb_tertiary",
        )
        if ee_rgb is not None:
            msg["ee_rgb"] = ee_rgb
    if msg.get("ee_camera_pose") is None:
        pose = _first_present(msg, "ee_camera_pose", "ee_cam/pose", "camera_pose_tertiary")
        if pose is not None:
            msg["ee_camera_pose"] = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if msg.get("ee_camera_K") is None:
        k = _first_present(msg, "ee_camera_K", "ee_cam/color_camera_K", "camera_K_tertiary")
        if k is not None:
            msg["ee_camera_K"] = np.asarray(k, dtype=np.float64).reshape(3, 3)


def _decode_servo_message_to_observations(
    msg: dict[str, Any],
    state: dict[str, Any] | None,
    full_obs: dict[str, Any] | None,
) -> Observations | None:
    """Build `Observations` for Rerun from servo dict (Stretch or Innate Mars bridge)."""
    rgb = _decode_jpg_field(msg, "head_color_image", "head_cam_left/color_image")
    if rgb is None:
        return None
    depth = None
    raw_d = msg.get("head_depth_image")
    if raw_d is not None:
        try:
            depth = compression.from_jp2(raw_d) / 1000.0
        except Exception:
            depth = None
    K = _first_present(msg, "head_camera_K", "head_cam_left/color_camera_K")
    if K is not None:
        K = _align_camera_k_to_rgb(np.asarray(K, dtype=np.float64).reshape(3, 3), rgb)
    joint: np.ndarray | None = None
    if msg.get("joint_positions") is not None:
        joint = np.asarray(msg["joint_positions"], dtype=float)
    elif state is not None and state.get("joint_positions") is not None:
        joint = np.asarray(state["joint_positions"], dtype=float)
    elif full_obs is not None and full_obs.get("joint") is not None:
        joint = np.asarray(full_obs["joint"], dtype=float)

    bp = msg.get("base_pose")
    if bp is None and state is not None:
        bp = state.get("base_pose")
    if bp is not None:
        bp = np.asarray(bp, dtype=float).ravel()
        if bp.size >= 3:
            gps, compass = bp[:2].copy(), np.asarray([float(bp[2])], dtype=float)
        else:
            gps, compass = np.zeros(2, dtype=float), np.zeros(1, dtype=float)
    elif full_obs is not None:
        g = full_obs.get("gps")
        c = full_obs.get("compass")
        if g is not None and c is not None:
            gps = np.asarray(g, dtype=float).reshape(-1)[:2]
            cc = np.asarray(c, dtype=float).ravel()
            compass = cc[:1].copy() if cc.size else np.zeros(1, dtype=float)
        else:
            gps, compass = np.zeros(2, dtype=float), np.zeros(1, dtype=float)
    else:
        gps, compass = np.zeros(2, dtype=float), np.zeros(1, dtype=float)

    cp = _first_present(msg, "camera_pose", "head_cam_left/pose")
    if cp is None and full_obs is not None:
        cp = full_obs.get("camera_pose")
    if cp is not None:
        cp = np.asarray(cp, dtype=np.float64).reshape(4, 4)

    ee_rgb = _decode_jpg_field(msg, "ee_cam/color_image", "ee_cam/image", "rgb_tertiary")
    ee_k = _first_present(msg, "ee_cam/color_camera_K", "ee_camera_K", "camera_K_tertiary")
    ee_pose = _first_present(msg, "ee_cam/pose", "ee_camera_pose", "camera_pose_tertiary")
    if ee_k is not None:
        ee_k = np.asarray(ee_k, dtype=np.float64).reshape(3, 3)
    if ee_pose is not None:
        ee_pose = np.asarray(ee_pose, dtype=np.float64).reshape(4, 4)

    step = msg.get("step")
    seq_id = int(step) if step is not None else -1

    sess = read_emet_session(msg)
    if sess is None:
        sess = read_emet_session(full_obs)

    return Observations(
        gps=gps,
        compass=compass,
        rgb=rgb,
        depth=depth,
        camera_K=K,
        camera_pose=cp,
        ee_rgb=ee_rgb,
        ee_camera_K=ee_k,
        ee_camera_pose=ee_pose,
        joint=joint,
        seq_id=seq_id,
        is_simulation=bool(msg.get("is_simulation", True)),
        emet_session=sess,
    )


def get_observation_from_zmq_dict(obs: dict[str, Any]) -> Observations | None:
    """Build :class:`Observations` from a decoded full-observation ZMQ dict."""
    rgb = obs.get("rgb")
    if rgb is None:
        return None
    enrich_zmq_observation_ee_fields(obs)
    joint_head = obs.get("joint_head")
    return Observations(
        rgb=rgb,
        depth=obs.get("depth"),
        camera_K=obs.get("camera_K"),
        camera_pose=obs.get("camera_pose"),
        head_rgb_right=obs.get("rgb_right"),
        head_camera_K_right=obs.get("camera_K_right"),
        head_camera_pose_right=obs.get("camera_pose_right"),
        ee_rgb=obs.get("ee_rgb"),
        ee_camera_K=obs.get("ee_camera_K"),
        ee_camera_pose=obs.get("ee_camera_pose"),
        ee_pose=obs.get("ee_pose"),
        joint=obs.get("joint"),
        joint_velocities=obs.get("joint_velocities"),
        joint_head=float(joint_head) if joint_head is not None else None,
        gps=obs.get("gps", np.zeros(2)),
        compass=obs.get("compass", np.zeros(1)),
        seq_id=int(obs.get("step", -1)) if obs.get("step") is not None else -1,
        is_simulation=bool(obs.get("is_simulation", False)),
        emet_session=read_emet_session(obs),
    )


class GenericZmqClient(AbstractRobotClient):
    """ZMQ client parameterised by a RobotSpec.

    Provides the same ZMQ protocol as StretchZmqClient but without
    Stretch-specific joint indices, kinematics, or camera assumptions.
    """

    num_state_report_steps: int = 10000

    def __init__(
        self,
        robot_spec: RobotSpec,
        robot_ip: str = "",
        recv_port: int = 4401,
        send_port: int = 4402,
        recv_state_port: int = 4403,
        recv_servo_port: int = 4404,
        port_offset: int = 0,
        parameters: Parameters | None = None,
        use_remote_computer: bool = True,
        start_immediately: bool = True,
        *,
        zmq_startup_timeout: float | None = None,
        allow_missing_depth: bool = False,
        enable_rerun_server: bool = False,
        rerun_headless: bool = False,
        rerun_native_viewer: bool = False,
        rerun_show_panels: bool = False,
        rerun_debug: bool = False,
        output_path: Path | str | None = None,
    ):
        super().__init__()
        self._robot_ip = robot_ip
        self._use_remote_computer = use_remote_computer
        if port_offset:
            recv_port += port_offset
            send_port += port_offset
            recv_state_port += port_offset
            recv_servo_port += port_offset
        self._spec = robot_spec
        self.recv_port = recv_port
        self.send_port = send_port
        self.recv_state_port = recv_state_port
        self.recv_servo_port = recv_servo_port
        if zmq_startup_timeout is not None:
            self._zmq_startup_timeout = max(1.0, float(zmq_startup_timeout))
        else:
            env = os.environ.get("EMET_ZMQ_STARTUP_TIMEOUT", "").strip()
            self._zmq_startup_timeout = max(1.0, float(env)) if env else 60.0

        self._joint_index: dict[str, int] = {name: i for i, name in enumerate(robot_spec.joint_names)}

        if parameters is None:
            parameters = get_parameters("default_planner.yaml")
        self._parameters = parameters
        self._allow_missing_depth = allow_missing_depth

        self._iter = -1
        self._seq_id = 0
        self._started = False
        self._finish = False
        self._zmq_closed = False

        self._obs: dict[str, Any] | None = None
        self._state: dict[str, Any] | None = None
        self._servo: dict[str, Any] | None = None
        self._servo_obs_rerun: Observations | None = None
        self._last_step = -1

        self._obs_lock = Lock()
        self._act_lock = Lock()
        self._mapping_depth_lock = Lock()
        self._mapping_depth_for_rerun: np.ndarray | None = None

        self._emet_session_cache: dict[str, Any] | None = None
        self._emet_session_cache_step: int = -1

        self._base_xyt = np.zeros(3)
        self._nav_goal_timeout_log_streak = 0

        self._rerun_debug = bool(rerun_debug) if enable_rerun_server else False
        self._rerun: Any = None
        self._rerun_thread: threading.Thread | None = None
        if enable_rerun_server:
            from emet.config.rerun_config import build_rerun_visualizer_kwargs
            from emet.visualization.rerun import RerunVisualizer

            out_p = Path(output_path) if output_path is not None else None
            if out_p is not None and not out_p.exists():
                out_p.mkdir(parents=True, exist_ok=True)
            mjcf_p = getattr(self._spec, "mjcf_path", None)
            use_mjcf = bool(mjcf_p and Path(str(mjcf_p)).is_file())
            mjcf_robot = None
            if use_mjcf:
                mjcf_robot = (
                    str(Path(str(mjcf_p)).resolve()),
                    tuple(self._spec.joint_names),
                    int(self._spec.dof),
                    str(self._spec.base_link_name),
                )
            rerun_kwargs = build_rerun_visualizer_kwargs(
                self._parameters,
                output_path=out_p,
                display_robot_mesh=use_mjcf,
                mjcf_robot=mjcf_robot,
                cli_headless=rerun_headless,
                cli_native_viewer=rerun_native_viewer,
                cli_show_panels=rerun_show_panels,
            )
            self._rerun = RerunVisualizer(**rerun_kwargs)
        else:
            from emet.visualization.rerun import NullVisualizer

            self._rerun = NullVisualizer()

        # ZMQ sockets
        self.context = zmq.Context()
        self.recv_socket = self._create_recv_socket(recv_port, robot_ip, use_remote_computer, "observations")
        self.recv_state_socket = self._create_recv_socket(
            recv_state_port, robot_ip, use_remote_computer, "low level state"
        )
        self.recv_servo_socket = self._create_recv_socket(
            recv_servo_port, robot_ip, use_remote_computer, "visual servoing data"
        )

        ip_address = lookup_address(robot_ip, use_remote_computer)
        send_address = ip_address + ":" + str(send_port)
        logger.debug(f"Connecting to {send_address} to send action messages...")
        self.send_socket = self.context.socket(zmq.PUB)
        self.send_socket.setsockopt(zmq.SNDHWM, 1)
        self.send_socket.setsockopt(zmq.RCVHWM, 1)
        self.send_socket.connect(send_address)
        logger.debug("...connected.")

        self._recv_threads_started = False
        self._recv_thread: threading.Thread | None = None
        self._state_thread: threading.Thread | None = None
        self._servo_thread: threading.Thread | None = None

        if start_immediately:
            if not self.start(log_startup_timeout=False):
                with self._obs_lock:
                    streams_ready = self._obs is not None and self._state is not None
                if not streams_ready:
                    msg = self._zmq_startup_failure_message()
                    logger.error(msg)
                    raise ConnectionError(msg)
                raise RuntimeError(
                    f"ZMQ streams are up but client startup failed (likely {EMET_ZMQ_ROBOT_ID_KEY} mismatch "
                    f"with server); client expects robot {self._spec.name!r}. Match `emet serve mujoco --robot`."
                )

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    # -- Socket helpers -------------------------------------------------------

    def _create_recv_socket(self, port: int, robot_ip: str, use_remote: bool, message_type: str = "") -> zmq.Socket:
        sock = self.context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.SNDHWM, 1)
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.CONFLATE, 1)

        ip_address = lookup_address(robot_ip, use_remote)
        if ip_address is None:
            logger.error("No robot IP address found.")
            sys.exit(1)
        addr = f"{ip_address}:{port}"
        logger.debug(f"Connecting to {addr} to receive {message_type}...")
        sock.connect(addr)
        return sock

    def send_message(self, message: dict[str, Any]) -> None:
        self.send_socket.send_pyobj(message)

    def request_sim_mujoco_ground_truth_snapshot(
        self,
        path_on_sim_host: str,
        *,
        exclude_robot: bool = True,
        as_json: bool = False,
    ) -> None:
        step_next = int(getattr(self, "_last_step", -1)) + 1
        if step_next < 1:
            step_next = max(1, int(time.time() * 1000.0) % 2_000_000_000)
        self.send_message(
            build_mujoco_ground_truth_dump_action(
                step_next,
                path_on_sim_host,
                exclude_robot=exclude_robot,
                as_json=as_json,
            ),
        )

    # -- Lifecycle ------------------------------------------------------------

    def _zmq_startup_failure_message(self) -> str:
        host = lookup_address(self._robot_ip, self._use_remote_computer) or self._robot_ip or "127.0.0.1"
        return (
            "Timeout waiting for first observations + state from MuJoCo ZMQ server "
            f"(waited {self._zmq_startup_timeout:.0f}s). "
            f"Host {host}: SUB obs={self.recv_port} state={self.recv_state_port} "
            f"servo={self.recv_servo_port}; PUB send={self.send_port} "
            "(use the same `--port-offset` on server and client). "
            "Start `emet serve mujoco` before the client; large MolmoSpaces scenes can take 30–90s to load. "
            "Increase wait: export EMET_ZMQ_STARTUP_TIMEOUT=120 or pass zmq_startup_timeout=120. "
            f"Robot must match: `emet serve mujoco --robot {self._spec.name}`. "
            "Use the same Python environment for client and server (e.g. `uv run emet` from repo root)."
        )

    def _wait_for_zmq_ready(self, timeout: float | None = None) -> bool:
        """Wait until at least one observation and one state message arrived."""
        if timeout is None:
            timeout = self._zmq_startup_timeout
        t0 = timeit.default_timer()
        last_log = t0
        while True:
            with self._obs_lock:
                ready = self._obs is not None and self._state is not None
            if ready:
                return True
            now = timeit.default_timer()
            if now - t0 > timeout:
                return False
            if now - last_log >= 10.0:
                logger.info(
                    f"Still waiting for ZMQ observations/state ({now - t0:.0f} / {timeout:.0f}s) "
                    f"on ports {self.recv_port} / {self.recv_state_port}…"
                )
                last_log = now
            time.sleep(0.05)

    def _verify_emet_robot_id(self) -> bool:
        """Ensure ``emet_robot_id`` from the server matches this client's RobotSpec (if present)."""
        with self._obs_lock:
            msg = self._obs if self._obs is not None else self._state
        rid = read_emet_robot_id_from_message_or_session(msg)
        if rid is None:
            return True
        expected = self._spec.name
        if robot_ids_match(rid, expected):
            return True
        logger.error(
            f"Robot ID mismatch: server reports {EMET_ZMQ_ROBOT_ID_KEY}={rid!r} but this client expects "
            f"{expected!r} (same as `emet serve mujoco --robot`). "
            "Use matching `--robot` on the agent, e.g. `--robot stretch` for the Stretch sim."
        )
        return False

    def start(self, *, log_startup_timeout: bool = True) -> bool:
        if self._started:
            return True
        if not self._recv_threads_started:
            self._recv_threads_started = True
            self._finish = False
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
            self._state_thread.start()
            self._servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
            self._servo_thread.start()
            if getattr(self._rerun, "enabled", False) and self._rerun_thread is None:
                self._rerun_thread = threading.Thread(target=self.blocking_spin_rerun, daemon=True)
                self._rerun_thread.start()
        if not self._wait_for_zmq_ready():
            if log_startup_timeout:
                logger.error(self._zmq_startup_failure_message())
            return False
        self._refresh_emet_session_from_streams()
        if not self._verify_emet_robot_id():
            return False
        self._started = True
        return True

    def _refresh_emet_session_from_streams(self) -> None:
        with self._obs_lock:
            for msg in (self._obs, self._state, self._servo):
                self._emet_session_cache, self._emet_session_cache_step = emet_session_cache_update(
                    self._emet_session_cache,
                    self._emet_session_cache_step,
                    msg,
                )

    def get_emet_session(self) -> dict[str, Any] | None:
        """Copy of the latest ``emet_session`` block from the server, if any (schema v1)."""
        with self._obs_lock:
            if self._emet_session_cache is None:
                return None
            return dict(self._emet_session_cache)

    def _robosuite_sim_zmq(self) -> bool:
        sess = self.get_emet_session()
        return bool(sess and sess.get("runtime_kind") == "robosuite_sim")

    def stop(self) -> None:
        """Signal threads to stop, join them, and close ZMQ sockets (idempotent)."""
        if self._zmq_closed:
            return
        self._zmq_closed = True
        self._finish = True
        for attr in ("_rerun_thread", "_recv_thread", "_state_thread", "_servo_thread"):
            t = getattr(self, attr, None)
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
        for sock_name in ("recv_socket", "recv_state_socket", "recv_servo_socket", "send_socket"):
            s = getattr(self, sock_name, None)
            if s is not None:
                try:
                    s.setsockopt(zmq.LINGER, 0)
                except Exception:
                    pass
                try:
                    s.close(linger=0)
                except TypeError:
                    try:
                        s.close()
                    except Exception:
                        pass
                except Exception:
                    pass
        ctx = getattr(self, "context", None)
        if ctx is not None:
            try:
                ctx.term()
            except Exception:
                pass

    def blocking_spin_rerun(self) -> None:
        """Background thread: stream latest ZMQ obs + decoded servo images to Rerun."""
        last_debug_t = 0.0
        step_count = 0
        while not self._finish:
            if getattr(self._rerun, "enabled", False):
                with self._obs_lock:
                    obs = self._obs
                    servo_obs = self._servo_obs_rerun
                mapping_depth = self.peek_mapping_depth_for_rerun()
                self._rerun.step(obs, servo_obs, mapping_depth=mapping_depth)
                step_count += 1
                if self._rerun_debug:
                    now = time.time()
                    if now - last_debug_t >= 2.0:
                        has_obs = obs is not None
                        has_servo = servo_obs is not None
                        rgb_shape = (
                            servo_obs.rgb.shape
                            if (servo_obs is not None and getattr(servo_obs, "rgb", None) is not None)
                            else None
                        )
                        logger.info(
                            f"[RERUN] generic obs={has_obs} servo_obs={has_servo} servo_rgb_shape={rgb_shape} "
                            f"steps={step_count}" + (" — waiting for MuJoCo ZMQ streams" if not has_obs else "")
                        )
                        last_debug_t = now
                if obs is None and servo_obs is None:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    @property
    def running(self) -> bool:
        """True until ``stop()`` (same contract as ``StretchZmqClient``)."""
        return not self._finish

    def is_running(self) -> bool:
        """True until ``stop()``; use ``robot.is_running()`` in loops (not ``robot.is_running``)."""
        return not self._finish

    def reset(self) -> None:
        self._obs = None
        self._state = None
        self._servo = None
        self._servo_obs_rerun = None
        self._base_xyt = np.zeros(3)
        self._base_control_mode = ControlMode.IDLE
        self._emet_session_cache = None
        self._emet_session_cache_step = -1
        with self._mapping_depth_lock:
            self._mapping_depth_for_rerun = None

    def set_mapping_depth_for_rerun(self, depth: np.ndarray | None) -> None:
        with self._mapping_depth_lock:
            self._mapping_depth_for_rerun = None if depth is None else np.asarray(depth, dtype=np.float32).copy()

    def peek_mapping_depth_for_rerun(self) -> np.ndarray | None:
        with self._mapping_depth_lock:
            return self._mapping_depth_for_rerun

    # -- Receive loops --------------------------------------------------------

    def _recv_loop(self) -> None:
        while not self._finish:
            try:
                output = self.recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue
            if output is None:
                continue
            self._seq_id += 1
            output["rgb"] = compression.from_jpg(output["rgb"])
            if output.get("camera_K") is not None:
                output["camera_K"] = _align_camera_k_to_rgb(
                    np.asarray(output["camera_K"], dtype=np.float64).reshape(3, 3),
                    output["rgb"],
                )
            if "rgb_right" in output and output["rgb_right"] is not None:
                output["rgb_right"] = compression.from_jpg(output["rgb_right"])
                if output.get("camera_K_right") is not None:
                    output["camera_K_right"] = _align_camera_k_to_rgb(
                        np.asarray(output["camera_K_right"], dtype=np.float64).reshape(3, 3),
                        output["rgb_right"],
                    )
            if "rgb_tertiary" in output and output["rgb_tertiary"] is not None:
                output["rgb_tertiary"] = compression.from_jpg(output["rgb_tertiary"])
            raw_depth = output.get("depth")
            if raw_depth is None:
                if not self._allow_missing_depth:
                    logger.warning(
                        "Observation missing depth; skipping frame (use allow_missing_depth for RGB-only servers)."
                    )
                    continue
                output["depth"] = None
            else:
                output["depth"] = compression.from_jp2(raw_depth) / 1000
            enrich_zmq_observation_ee_fields(output)
            with self._obs_lock:
                self._obs = output
                if "step" in output:
                    self._last_step = output["step"]
                if "gps" in output and "compass" in output:
                    self._base_xyt = np.array([output["gps"][0], output["gps"][1], output["compass"][0]])
                self._emet_session_cache, self._emet_session_cache_step = emet_session_cache_update(
                    self._emet_session_cache,
                    self._emet_session_cache_step,
                    output,
                )

    def _state_loop(self) -> None:
        while not self._finish:
            try:
                msg = self.recv_state_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue
            if msg is None:
                continue
            with self._obs_lock:
                self._state = msg
                if "step" in msg:
                    self._last_step = msg["step"]
                self._emet_session_cache, self._emet_session_cache_step = emet_session_cache_update(
                    self._emet_session_cache,
                    self._emet_session_cache_step,
                    msg,
                )

    def _servo_loop(self) -> None:
        while not self._finish:
            try:
                msg = self.recv_servo_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue
            if msg is None:
                continue
            with self._obs_lock:
                self._servo = msg
                self._servo_obs_rerun = _decode_servo_message_to_observations(msg, self._state, self._obs)
                self._emet_session_cache, self._emet_session_cache_step = emet_session_cache_update(
                    self._emet_session_cache,
                    self._emet_session_cache_step,
                    msg,
                )

    # -- Observations ---------------------------------------------------------

    def wait_for_obs(self, timeout: float = 10.0, *, require_navigation_origin: bool | None = None) -> bool:
        if require_navigation_origin is None:
            require_navigation_origin = self._robosuite_sim_zmq()
        t0 = timeit.default_timer()
        while True:
            with self._obs_lock:
                has_obs = self._obs is not None
            if has_obs:
                if not require_navigation_origin:
                    return True
                sess = self.get_emet_session()
                if sess is None:
                    pass
                elif sess.get("runtime_kind") != "robosuite_sim":
                    return True
                elif sess.get("navigation_origin_xyt") is not None:
                    return True
            if timeit.default_timer() - t0 > timeout:
                if require_navigation_origin:
                    logger.error(
                        "Timeout waiting for observations with navigation_origin_xyt "
                        "(Robocasa sim must finish autoplace before mapping/Rerun align)."
                    )
                else:
                    logger.error("Timeout waiting for observations.")
                return False
            time.sleep(0.05)

    def get_joint_positions(self, timeout: float = 5.0) -> np.ndarray | None:
        state = self._state
        if state is not None and "joint_positions" in state:
            return np.array(state["joint_positions"])
        obs = self._obs
        if obs is not None and "joint" in obs:
            return np.array(obs["joint"])
        return None

    def get_joint_velocities(self, timeout: float = 5.0) -> np.ndarray | None:
        state = self._state
        if state is not None and "joint_velocities" in state:
            return np.array(state["joint_velocities"])
        return None

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        state = self._state
        if state is not None and "base_pose" in state:
            bp = state["base_pose"]
            if bp is not None:
                return np.array(bp)
        return self._base_xyt.copy()

    def _nav_goal_reset_seen(self) -> bool:
        """True if any cached ZMQ message reports ``at_goal`` false (server cleared stale true)."""
        with self._obs_lock:
            chunks = (self._state, self._obs, self._servo)
        for msg in chunks:
            if msg is None:
                continue
            if not bool(msg.get("at_goal", False)):
                return True
        return False

    def at_goal(self) -> bool:
        """True if any of state / full-obs / servo reports at goal (ZMQ CONFLATE can leave one socket stale)."""
        with self._obs_lock:
            chunks = (self._state, self._obs, self._servo)
        for msg in chunks:
            if msg is None:
                continue
            if bool(msg.get("at_goal", False)):
                return True
        return False

    def get_observation(self, max_iter: int = 5) -> Observations | None:
        """Get the latest observation from the server."""
        with self._obs_lock:
            obs = self._obs
        if obs is None:
            return None

        rgb = obs.get("rgb")
        depth = obs.get("depth")
        if rgb is None:
            return None
        if depth is None and not self._allow_missing_depth:
            return None

        camera_K = obs.get("camera_K")
        camera_pose = obs.get("camera_pose")
        ee_pose = obs.get("ee_pose")
        joint = obs.get("joint")
        joint_head = obs.get("joint_head")
        gps = obs.get("gps", np.zeros(2))
        compass = obs.get("compass", np.zeros(1))

        ee_rgb = obs.get("ee_rgb")
        ee_camera_K = obs.get("ee_camera_K")
        ee_camera_pose = obs.get("ee_camera_pose")
        if ee_rgb is None:
            enrich_zmq_observation_ee_fields(obs)
            ee_rgb = obs.get("ee_rgb")
            ee_camera_K = obs.get("ee_camera_K")
            ee_camera_pose = obs.get("ee_camera_pose")

        return Observations(
            rgb=rgb,
            depth=depth,
            camera_K=camera_K,
            camera_pose=camera_pose,
            head_rgb_right=obs.get("rgb_right"),
            head_camera_K_right=obs.get("camera_K_right"),
            head_camera_pose_right=obs.get("camera_pose_right"),
            ee_rgb=ee_rgb,
            ee_camera_K=ee_camera_K,
            ee_camera_pose=ee_camera_pose,
            ee_pose=ee_pose,
            joint=joint,
            joint_velocities=obs.get("joint_velocities"),
            joint_head=float(joint_head) if joint_head is not None else None,
            gps=gps,
            compass=compass,
            emet_session=read_emet_session(obs),
        )

    def peek_emet_robot_id(self) -> str | None:
        """Return ``emet_robot_id`` from the latest full-observation ZMQ dict (logging / CLI)."""
        with self._obs_lock:
            raw = self._obs
        return read_emet_robot_id_from_message_or_session(raw)

    def get_head_pose(self) -> np.ndarray:
        """SE(3) head / primary camera frame; fall back to identity if unknown."""
        obs = self.get_observation()
        if obs is None or obs.camera_pose is None:
            return np.eye(4, dtype=np.float64)
        return np.asarray(obs.camera_pose, dtype=np.float64)

    def get_servo_observation(self) -> Observations | None:
        """Low-rate head/EE stream (Stretch servo port or Innate Mars bridge 4404)."""
        with self._obs_lock:
            if self._servo_obs_rerun is not None:
                return self._servo_obs_rerun
            obs = self._obs
        if obs is None:
            return None
        decoded = get_observation_from_zmq_dict(obs) if isinstance(obs, dict) else None
        return decoded if decoded is not None else self.get_observation()

    # -- Actions --------------------------------------------------------------

    def send_action(
        self,
        action: dict[str, Any],
        timeout: float = 5.0,
        reliable: bool = True,
    ) -> dict[str, Any]:
        with self._act_lock:
            block_id = max(self._iter, self._last_step + 1)
            action["step"] = block_id
            self._iter = block_id + 1
            self.send_message(action)
            while reliable and self._last_step < block_id:
                self.send_message(action)
                time.sleep(0.01)
        return action

    def set_velocity(self, v: float, w: float) -> None:
        """Set base translational (v) and rotational (w) velocity setpoints."""
        self.send_action({"v": v, "w": w})

    def move_base_to(
        self,
        xyt,
        relative=False,
        blocking=False,
        verbose: bool = False,
        timeout: float | None = None,
        *,
        world_frame: bool | None = None,
    ) -> bool:
        if isinstance(xyt, ContinuousNavigationAction):
            xyt = xyt.xyt
        xyt = np.array(xyt, dtype=float).reshape(-1)
        if xyt.size < 3:
            xyt = np.pad(xyt, (0, max(0, 3 - xyt.size)), mode="constant")
        # Default episode-relative (gps frame); only voxel/planner paths pass world_frame=True (nav_world).
        if world_frame is None:
            world_frame = False
        action: dict[str, Any] = {"xyt": xyt[:3].tolist()}
        if relative:
            action["nav_relative"] = True
        elif world_frame:
            action["nav_world"] = True
        frame_tag = "nav_relative" if relative else ("nav_world" if world_frame else "episode_compose")
        logger.info(
            f"move_base_to: goal=[{float(xyt[0]):.3f}, {float(xyt[1]):.3f}, {float(xyt[2]):.3f}] "
            f"frame={frame_tag} blocking={blocking}"
        )
        if env_sim_nav_teleport():
            warn_sim_nav_env_flags()
            action["nav_teleport"] = True
        if world_frame and self._robosuite_sim_zmq():
            sess = read_emet_session(self._obs) or read_emet_session(self._state)
            org = None if sess is None else sess.get("navigation_origin_xyt")
            if org is not None:
                origin = np.asarray(org, dtype=np.float64).reshape(-1)[:3]
                dist = float(np.linalg.norm(xyt[:2] - origin[:2]))
                if dist > 12.0:
                    logger.warning(
                        "move_base_to: refusing world goal [%.3f, %.3f, %.3f] — %.1fm from "
                        "navigation_origin [%.3f, %.3f] (planner/rotate frame bug or unstable sim).",
                        xyt[0],
                        xyt[1],
                        xyt[2],
                        dist,
                        origin[0],
                        origin[1],
                    )
                    return False
        self.send_action(action)
        if blocking:
            # PUB/SUB can drop the first packet; give the server a beat to apply xyt.
            time.sleep(0.02)
            # Avoid succeeding immediately on a stale ``at_goal`` from the previous navigation
            # before the server recv thread clears it for the new goal.
            t_clear = timeit.default_timer()
            while not self._nav_goal_reset_seen() and timeit.default_timer() - t_clear < 1.0:
                time.sleep(0.01)
            return self._wait_at_goal(timeout=timeout or 30.0, target_xyt=xyt)
        return True

    def _wait_at_goal(self, timeout: float = 30.0, target_xyt: np.ndarray | None = None) -> bool:
        t0 = timeit.default_timer()
        while not self.at_goal():
            time.sleep(0.05)
            if timeit.default_timer() - t0 > timeout:
                self._nav_goal_timeout_log_streak += 1
                g = np.asarray(target_xyt, dtype=float).reshape(-1) if target_xyt is not None else None
                goal_s = f"[{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}]" if g is not None and g.size >= 3 else str(target_xyt)
                detail = (
                    f"Navigation goal not reached within {timeout:.0f}s (target_xyt={goal_s}, "
                    f"at_goal={self.at_goal()}). If this repeats, the sim may never set `at_goal`, "
                    "the target may be unreachable, or try tighter `--goal-*` bounds / "
                    "`emet run molmospaces-explore --navigate-every` less often."
                )
                if self._nav_goal_timeout_log_streak <= 2:
                    logger.warning(detail)
                else:
                    logger.debug(
                        f"{detail} (suppressing further identical warnings; streak={self._nav_goal_timeout_log_streak})"
                    )
                return False
        self._nav_goal_timeout_log_streak = 0
        return True

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        """Send a joint position command using actuator names."""
        action = {"joint": list(positions.values())}
        self.send_action(action)

    def open_gripper(self, gripper_name: str = "left_gripper", amount: float = 0.05) -> None:
        if self._spec.name == "xlerobot":
            from emet.robots.xlerobot import parse_xlerobot_gripper_side

            self.gripper_to(1.0, side=parse_xlerobot_gripper_side(gripper_name))
            return
        action = {"gripper": amount}
        self.send_action(action)

    def close_gripper(self, gripper_name: str = "left_gripper") -> None:
        if self._spec.name == "xlerobot":
            from emet.robots.xlerobot import parse_xlerobot_gripper_side

            self.gripper_to(0.0, side=parse_xlerobot_gripper_side(gripper_name))
            return
        action = {"gripper": -0.1}
        self.send_action(action)

    def switch_to_navigation_mode(self) -> None:
        action = {"control_mode": "navigation"}
        self.send_action(action)
        self._base_control_mode = ControlMode.NAVIGATION

    def switch_to_manipulation_mode(self, verbose: bool = False) -> None:
        action = {"control_mode": "manipulation"}
        self.send_action(action)
        self._base_control_mode = ControlMode.MANIPULATION

    def move_to_nav_posture(self) -> None:
        action = {"posture": "navigation"}
        self.send_action(action)

    def move_to_manip_posture(self) -> None:
        action = {"posture": "manipulation"}
        self.send_action(action)

    # -- Stretch / DynaMem API shims (this client is spec-driven, not Stretch-indexed) ------------

    def get_joint_state(
        self, timeout: float = 5.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """Joint positions, velocities, and efforts from the state stream (Stretch-compatible shape)."""
        t0 = timeit.default_timer()
        while timeit.default_timer() - t0 < timeout:
            with self._obs_lock:
                st = self._state
            if st is not None and "joint_positions" in st:
                q = np.asarray(st["joint_positions"], dtype=float)
                dq = np.asarray(st.get("joint_velocities", np.zeros_like(q)), dtype=float)
                tau = np.asarray(st.get("joint_efforts", np.zeros_like(q)), dtype=float)
                return q, dq, tau
            time.sleep(0.01)
        return None, None, None

    def get_six_joints(self, _timeout: float = 5.0) -> np.ndarray:
        """Placeholder for DynaMem's Stretch 6-DOF slice; xlerobot returns head pan/tilt."""
        if self._spec.name == "xlerobot":
            pan, tilt = self.get_pan_tilt()
            return np.array([pan, tilt], dtype=float)
        return np.zeros(6, dtype=float)

    def get_pan_tilt(self) -> tuple[float, float]:
        """Head pan/tilt in radians (xlerobot ``head_pan_joint`` / ``head_tilt_joint``)."""
        if self._spec.name == "xlerobot":
            q, _, _ = self.get_joint_state(timeout=2.0)
            if q is None:
                return (0.0, 0.0)
            pan_i = self._joint_index.get("head_pan_joint")
            tilt_i = self._joint_index.get("head_tilt_joint")
            pan = float(q[pan_i]) if pan_i is not None else 0.0
            tilt = float(q[tilt_i]) if tilt_i is not None else 0.0
            return (pan, tilt)
        return (0.0, 0.0)

    def get_gripper_position(self, side: str = "left") -> float:
        if self._spec.name == "xlerobot":
            from emet.robots.xlerobot import XLEROBOT_GRIPPER_JOINTS, jaw_normalized_from_angle

            q, _, _ = self.get_joint_state(timeout=2.0)
            if q is None:
                return 0.0
            key = side if side in ("left", "right") else "left"
            jname = XLEROBOT_GRIPPER_JOINTS[key]
            idx = self._joint_index.get(jname)
            if idx is None:
                return 0.0
            return jaw_normalized_from_angle(float(q[idx]))
        q, _, _ = self.get_joint_state(timeout=2.0)
        if q is None:
            return 0.5
        for i, name in enumerate(self._spec.joint_names):
            if "gripper" in name and "finger" in name:
                return float(q[i])
        return 0.5

    def arm_to(self, joint_angles=None, gripper=None, head=None, blocking=True, **kwargs) -> bool:
        if not getattr(self, "_logged_arm_to_nonstretch", False):
            logger.warning(
                "arm_to() targets Stretch's 6-DOF arm; this generic robot does not map those commands (no-op). "
                "DynaMem manipulation paths remain Stretch-oriented."
            )
            self._logged_arm_to_nonstretch = True
        return True

    def head_to(self, head_pan: float, head_tilt: float, blocking: bool = False, **kwargs) -> None:
        """Send Stretch-compatible ``head_to`` to the ZMQ server (``RobosuiteZmqServer`` maps it for rby1/galaxea)."""
        next_action: dict[str, Any] = {
            "head_to": [float(head_pan), float(head_tilt)],
        }
        if blocking:
            next_action["manip_blocking"] = True
        self.send_action(
            next_action,
            timeout=float(kwargs.get("timeout", 5.0)),
            reliable=bool(kwargs.get("reliable", True)),
        )
        if blocking:
            time.sleep(0.05)

    def look_front(self, blocking: bool = True, timeout: float = 10.0) -> None:
        self.head_to(
            float(motion_constants.look_front[0]),
            float(motion_constants.look_front[1]),
            blocking=blocking,
            timeout=timeout,
            reliable=True,
        )

    def look_at_ee(self, blocking: bool = True, timeout: float = 10.0) -> None:
        self.head_to(
            float(motion_constants.look_at_ee[0]),
            float(motion_constants.look_at_ee[1]),
            blocking=blocking,
            timeout=timeout,
            reliable=True,
        )

    def say(self, text: str) -> None:
        """Text-to-speech on Stretch; generic client has no audio bridge (no-op)."""

    def say_sync(self, text: str) -> None:
        """Blocking TTS on Stretch; generic client has no audio bridge (no-op)."""

    def navigate_to(self, xyt, relative: bool = False, blocking: bool = True, **kwargs) -> bool:
        xyt_a = np.asarray(xyt, dtype=float).reshape(-1)
        if xyt_a.size != 3:
            logger.error("navigate_to expects a length-3 xyt vector")
            return False
        return self.move_base_to(xyt_a, relative=relative, blocking=blocking, timeout=kwargs.get("timeout"))

    def gripper_to(self, target: float, blocking: bool = True, reliable: bool = True, side: str = "left") -> None:
        """Send gripper command (Stretch single gripper or xlerobot left/right jaw)."""
        if self._spec.name == "xlerobot":
            key = "gripper_right" if side == "right" else "gripper_left"
            action: dict[str, Any] = {key: float(target)}
            if blocking:
                action["gripper_blocking"] = True
            self.send_action(action, reliable=reliable)
            if blocking:
                time.sleep(0.05)
            return
        action = {"gripper": float(target)}
        if blocking:
            action["gripper_blocking"] = True
        self.send_action(action, reliable=reliable)
        if blocking:
            time.sleep(0.05)

    def gripper_both_to(self, target: float, blocking: bool = True, reliable: bool = True) -> None:
        """Set both xlerobot jaws; no-op on single-gripper robots."""
        if self._spec.name != "xlerobot":
            self.gripper_to(target, blocking=blocking, reliable=reliable)
            return
        action: dict[str, Any] = {"gripper_left": float(target), "gripper_right": float(target)}
        if blocking:
            action["gripper_blocking"] = True
        self.send_action(action, reliable=reliable)
        if blocking:
            time.sleep(0.05)

    def get_robot_model(self):
        return SpecRobotModel(self._spec)

    def get_pose_graph(self) -> np.ndarray:
        return np.zeros((0, 3))

    def execute_trajectory(
        self,
        trajectory,
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
        final_timeout: float = 60.0,
        blocking: bool = True,
        *,
        world_frame: bool | None = None,
    ) -> bool:
        for waypoint in trajectory:
            if not self.move_base_to(
                waypoint,
                relative=relative,
                blocking=blocking,
                timeout=per_waypoint_timeout,
                world_frame=world_frame,
            ):
                return False
        return True
