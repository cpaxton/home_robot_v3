# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Stopped-base ZMQ bridge for RoboCasa's native PandaOmron controllers."""

import time

import mujoco
import numpy as np

from emet.core.capture_clock import local_monotonic_clock_id
from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
)
from emet.robots.pandaomron import (
    ARM_JOINTS,
    CONTROL_JOINTS,
    FINGER_JOINTS,
    FRONT_CAMERA,
    GRASP_BODY,
    PANDA_ARM_CHAIN,
    WRIST_CAMERA,
    PandaOmronBackend,
)
from emet.simulation.pandaomron_environment import CONTROL_HZ
from emet.simulation.pandaomron_probe import PandaPlacementProbe
from emet.simulation.physical_execution import audit_physical_action
from emet.simulation.robosuite_server import RobosuiteZmqServer
from emet.simulation.sim_object_placements import apply_navigation_origin_to_session
from emet.utils import compression


class PandaOmronZmqServer(RobosuiteZmqServer):
    """Expose the native arm, torso, and gripper at a stopped scenario reset."""

    # Keep the millimeter values used for collision evidence exact on the wire.
    _depth_jp2_quality = 1000

    def __init__(self, *, env, environment, scene_camera=FRONT_CAMERA, probe_calibration=None, **kwargs):
        spec = PandaOmronBackend().get_spec()
        spec.camera_names[0] = scene_camera
        self._auxiliary_scene_camera = "robot0_agentview_left" if scene_camera != "robot0_agentview_left" else "robot0_agentview_right"
        self._scene_camera = scene_camera
        super().__init__(
            robot_spec=spec,
            simulation_rate=CONTROL_HZ,
            environment=environment,
            **kwargs,
        )
        self._env = env
        self._robot = env.robots[0]
        self._mjmodel = env.sim.model._model
        self._mjdata = env.sim.data._data
        if not self._no_cameras:
            for camera in self._spec.camera_names:
                self._mjmodel.camera(camera)
        self._joint_ids = [self._mjmodel.joint(name).id for name in self._spec.joint_names]
        self._qpos_addresses = self._mjmodel.jnt_qposadr[self._joint_ids]
        self._qvel_addresses = self._mjmodel.jnt_dofadr[self._joint_ids]
        self._limits = self._mjmodel.jnt_range[self._joint_ids[: len(CONTROL_JOINTS)]]
        initial = self._mjdata.qpos[self._qpos_addresses[: len(CONTROL_JOINTS)]]
        self._goal = np.clip(initial, self._limits[:, 0], self._limits[:, 1])
        self._grip = -1.0  # RoboSuite GRIP uses -1 to open and +1 to close.
        self._finger_open = np.array([position for _, position in PANDA_ARM_CHAIN.gripper_open_configuration])
        self._finger_target = self._finger_open
        self._finger_limits = self._mjmodel.jnt_range[self._joint_ids[len(CONTROL_JOINTS) :]]
        grip = self._robot.part_controllers["right_gripper"]
        self._finger_actuator_limits = np.column_stack((grip.actuator_min, grip.actuator_max))
        self._placement_probe = PandaPlacementProbe(
            self._mjmodel, self._robot, self._qpos_addresses[: len(CONTROL_JOINTS)],
            self._limits, probe_calibration,
        )
        self._initial_xyt = self.get_base_xyt()
        self.control_mode = "manipulation"
        self._emet_session = {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            EMET_ZMQ_ROBOT_ID_KEY: self._spec.name,
            "runtime_kind": "robocasa_native",
            "is_simulation": True,
            "environment": dict(environment),
            "scene_camera": self._scene_camera,
            "auxiliary_scene_camera": self._auxiliary_scene_camera,
            "native_joint_goal_limits": dict(zip(CONTROL_JOINTS, self._limits.tolist(), strict=True)),
            "native_finger_goal_limits": dict(zip(FINGER_JOINTS, self._finger_limits.tolist(), strict=True)),
            "capabilities": {
                "depth": not self._no_cameras,
                "num_cameras": len(self._spec.camera_names) if not self._no_cameras else 0,
                "native_joint_position": True,
                "native_gripper": True,
                "native_gripper_position": True,
                "wrist_wrench": True,
                "guarded_placement_probe": probe_calibration is not None,
                "dof": self._spec.dof,
                "nav_velocity_drive": False,
                "head_motion": False,
                "teleport_base": False,
                "kinematic_manip": False,
            },
        }
        if probe_calibration is not None:
            self._emet_session["probe_calibration_id"] = probe_calibration.calibration_id
        apply_navigation_origin_to_session(self._emet_session, self._initial_xyt)

    def start(self, *, headless=True, show_viewer_ui=False, use_glx=False):
        self._configure_offscreen_gl(use_glx=use_glx)
        self._run_initialized_server(headless, show_viewer_ui)

    def _sim_loop(self):
        try:
            while self._running:
                started = time.monotonic()
                self._step_physics_tick()
                if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                    break
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.monotonic() - started)))
        except Exception as exc:
            self._physics_error = exc
            raise
        finally:
            self._running = False

    def _step_physics_tick(self):
        with self._mj_lock:
            self._goal = self._placement_probe.tick(self._mjdata, self._goal)
            arm = self._goal[: len(ARM_JOINTS)]
            torso = self._goal[len(ARM_JOINTS) :]
            grip = self._grip
            if self._finger_target is not None:
                low, high = self._finger_actuator_limits.T
                # The native GRIP controller scales the gripper model's
                # normalized goal into the two position actuator ranges.
                # A zero command retains the explicitly bounded goal.
                self._robot.gripper["right"].current_action = 2 * (self._finger_target - low) / (high - low) - 1
                grip = 0.0
            action = self._robot.create_action_vector({"right": arm, "torso": torso, "right_gripper": [grip]})
            self._env.step(action)
            self._physics_steps_executed += 1
            self._physics_fps_counter.tick(self._mjdata.time)

    def handle_action(self, action):
        probe_keys = {"placement_probe", "placement_probe_keepalive", "placement_probe_cancel"} & action.keys()
        if probe_keys:
            if len(probe_keys) != 1 or set(action) - (probe_keys | {"step"}):
                raise ValueError("placement probe commands must be isolated")
            key = next(iter(probe_keys))
            with self._mj_lock:
                if key == "placement_probe":
                    request = action[key]
                    if not isinstance(request, dict) or set(request) != {
                        "probe_id", "calibration_id", "capture_end_s", "contact_band_m", "waypoints",
                    }:
                        raise ValueError("invalid_probe_request")
                    self._placement_probe.start(request, self._mjdata)
                    self._goal = self._placement_probe.hold_goal(self._mjdata)
                elif key == "placement_probe_keepalive":
                    self._placement_probe.keepalive(action[key])
                else:
                    self._goal = self._placement_probe.cancel(action[key], self._mjdata)
                audit_physical_action(action, source="PandaOmronZmqServer")
            return
        known = {"step", "joint", "gripper", "gripper_blocking", "gripper_joint_targets", "posture", "control_mode"}
        unsupported = set(action) - known
        if unsupported:
            raise ValueError(f"PandaOmron stopped-base bridge does not support {sorted(unsupported)}")
        mode = action.get("posture", action.get("control_mode", self.control_mode))
        if mode not in ("navigation", "manipulation"):
            raise ValueError("unknown control mode")
        joints = None
        if "joint" in action:
            joints = np.asarray(action["joint"], dtype=float)
            if joints.shape != (len(CONTROL_JOINTS),) or not np.isfinite(joints).all():
                raise ValueError("expected seven finite arm joint targets and one torso target")
            if np.any(joints < self._limits[:, 0]) or np.any(joints > self._limits[:, 1]):
                raise ValueError("joint target is outside the native joint limits")
        fingers = None
        if "gripper_joint_targets" in action:
            if "gripper" in action:
                raise ValueError("provide one gripper command")
            fingers = np.asarray(action["gripper_joint_targets"], dtype=float)
            if fingers.shape != (len(FINGER_JOINTS),) or not np.isfinite(fingers).all():
                raise ValueError("expected two finite finger targets")
            for limits in (self._finger_limits, self._finger_actuator_limits):
                if np.any(fingers < limits[:, 0]) or np.any(fingers > limits[:, 1]):
                    raise ValueError("finger target outside the native joint limits")
        grip = None
        if "gripper" in action:
            target = float(action["gripper"])
            if not np.isfinite(target):
                raise ValueError("gripper target must be finite")
            grip = -1.0 if target > 0 else 1.0
        with self._mj_lock:
            if self._placement_probe.active:
                raise ValueError("placement probe is active")
            audit_physical_action(action, source="PandaOmronZmqServer")
            self.control_mode = mode
            if joints is not None:
                self._goal = joints
            if grip is not None:
                self._grip = grip
                self._finger_target = self._finger_open if grip < 0 else None
            elif fingers is not None:
                self._finger_target = fingers

    def start_navigation_command(self, action):
        raise ValueError("PandaOmron bridge currently requires a stopped base")

    def cancel_navigation_command(self):
        # A rejected navigation request never dispatched a controller goal.
        return True

    def navigation_policy_names(self):
        return ()

    def check_base_poses(self, poses):
        raise ValueError("base pose queries are unavailable in the stopped-base bridge")

    def get_base_xyt(self, data=None):
        data = self._mjdata if data is None else data
        with self._mj_lock:
            base = data.body(self._spec.base_link_name)
            return np.r_[base.xpos[:2], np.arctan2(base.xmat[3], base.xmat[0])]

    def get_base_pose(self, data=None):
        current = self.get_base_xyt(data)
        origin = self._initial_xyt
        dx, dy, dtheta = current - origin
        c, s = np.cos(origin[2]), np.sin(origin[2])
        return np.array([c * dx + s * dy, -s * dx + c * dy, np.arctan2(np.sin(dtheta), np.cos(dtheta))])

    def _capture_render_state(self):
        # RoboCasa365's MuJoCo 3.3.1 has no Python mj_copyData binding.
        model = self._mjmodel
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, state_type))
        with self._mj_lock:
            started_ns = time.monotonic_ns()
            mujoco.mj_getState(model, self._mjdata, state, state_type)
            finished_ns = time.monotonic_ns()
        self._capture_interval_ns = [started_ns, finished_ns]
        snapshot = mujoco.MjData(model)
        mujoco.mj_setState(model, snapshot, state, state_type)
        mujoco.mj_forward(model, snapshot)
        return snapshot

    def _configure_renderer_geomgroups_for_camera(self, renderer, camera_name):
        # Match RoboSuite's default view: visual meshes without collision overlays.
        renderer._scene_option.geomgroup[:] = mujoco.MjvOption().geomgroup
        renderer._scene_option.geomgroup[0] = 0
        renderer._scene_option.geomgroup[1] = 1

    def get_joint_state(self, data=None):
        data = self._mjdata if data is None else data
        with self._mj_lock:
            return (
                data.qpos[self._qpos_addresses],
                data.qvel[self._qvel_addresses],
                data.qfrc_actuator[self._qvel_addresses],
            )

    def _ee_pose(self, data):
        body = data.body(GRASP_BODY)
        pose = np.eye(4)
        pose[:3, :3] = body.xmat.reshape(3, 3)
        pose[:3, 3] = body.xpos
        return pose

    def _attach_emet_session(self, message):
        # The native bridge supplies calibrated observations and robot feedback.
        # Object and fixture state belong to the evaluation process.
        message[EMET_ZMQ_SESSION_KEY] = self._emet_session
        message["sim_base_pose_query"] = False
        return message

    def get_state_message(self):
        with self._mj_lock:
            message = super().get_state_message()
            message.update(
                actuator_targets=self._goal.copy(),
                ee_pose=self._ee_pose(self._mjdata),
                joint_sim_time=float(self._mjdata.time),
                wrist_wrench=self._placement_probe.telemetry(),
                placement_probe_status=self._placement_probe.status(),
            )
        return self._attach_emet_session(message)

    def get_full_observation_message(self):
        message = super().get_full_observation_message()
        if message is not None and not self._no_cameras:
            data = self._camera_render_states[self._scene_camera]
            message["ee_pose"] = self._ee_pose(data)
            message["joint_velocities"] = data.qvel[self._qvel_addresses]
            interval = list(self._capture_interval_ns)
            timing = {
                "timestamp_ns": interval[0],
                "capture_interval_ns": interval,
                "clock_domain": "mujoco_render_monotonic",
                "clock_id": local_monotonic_clock_id(),
                "source": "renderer_synchronized_snapshot",
                "available": True,
            }
            message["head_cam_left/image_timing"] = {**timing, "depth_capture_interval_ns": interval}
            message["ee_cam/image_timing"] = {**timing, "depth_capture_interval_ns": interval}
            alternate = self._auxiliary_scene_camera
            alternate_rgb, alternate_depth = self._render_primary_rgb_and_depth_raw(alternate, snapshot=data)
            alternate_rgb, alternate_depth, alternate_K = self._postprocess_rgb_depth_and_K(alternate, alternate_rgb, alternate_depth)
            # Keep additional sensor arrays explicit and lossless. They contain
            # rendered measurements/calibration only, never scene identities.
            message["auxiliary_rgbd_views"] = {"scene_alt": {
                "rgb": alternate_rgb, "depth": alternate_depth,
                "camera_K": alternate_K, "camera_pose": self._camera_pose_world(alternate, data=data),
                "timing": {**timing, "depth_capture_interval_ns": interval},
            }}
            rgb, depth = self._render_primary_rgb_and_depth_raw(WRIST_CAMERA, snapshot=data)
            rgb, depth, intrinsics = self._postprocess_rgb_depth_and_K(WRIST_CAMERA, rgb, depth)
            valid_depth = np.isfinite(depth) & (depth > 0) & (depth <= 65.535)
            depth_mm = np.where(valid_depth, depth * 1000, 0).astype(np.uint16)
            message.update(
                ee_rgb=compression.to_jpg(rgb),
                ee_depth=compression.to_jp2(depth_mm, quality=self._depth_jp2_quality),
                ee_camera_K=intrinsics,
                ee_camera_pose=self._camera_pose_world(WRIST_CAMERA, data=data),
                ee_image_sim_time=float(data.time),
            )
        return message

    def stop(self):
        self._running = False
        thread = getattr(self, "_sim_thread", None)
        if thread is not None:
            thread.join()
        super().stop()
        if self._env is not None:
            with self._mj_lock:
                self._env.close()
                self._env = None
