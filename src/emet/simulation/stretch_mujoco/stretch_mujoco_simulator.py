# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import atexit
import multiprocessing
import os
import platform
import signal
import sys
import threading
import time
from multiprocessing import Lock, Manager, Process
from typing import Any

import numpy as np

from emet.utils.logger import Logger

logger = Logger(__name__)

from emet.utils.mujoco_import import assert_mujoco_available

assert_mujoco_available()
from mujoco import MjModel

import emet.simulation.stretch_mujoco.utils as utils
from emet.simulation.stretch_mujoco.datamodels.status_command import (
    CommandBaseVelocity,
    CommandCoordinateFrameArrowsViz,
    CommandKeyframe,
    CommandMove,
    CommandTeleportBase,
    StatusCommand,
)
from emet.simulation.stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from emet.simulation.stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from emet.simulation.stretch_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from emet.simulation.stretch_mujoco.enums.actuators import Actuators
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
from emet.simulation.stretch_mujoco.mujoco_server_managed import MujocoServerManaged
from emet.simulation.stretch_mujoco.mujoco_server_passive import MujocoServerPassive
from emet.simulation.stretch_mujoco.utils import block_until_check_succeeds, require_connection


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco Server.

    Calling `start()` will spawn a new process that runs `MujocoServer` and the simulator.

    You can specify `start(headless=True)` to run the simulation without a GUI.

    Data from the MujocoServer is sent to StretchMujocoSimulator using proxies.

    Use `pull_status()` and `pull_camera_data()` to access simulation data.
    """

    def __init__(
        self,
        scene_xml_path: str | None = None,
        model: MjModel | None = None,
        camera_hz: float = 30,
        cameras_to_use: list[StretchCameras] = None,
        start_translation: list | None = None,
        start_rotation_quat: list | None = None,
        *,
        molmospaces_environment: dict[str, Any] | None = None,
        debug_molmospaces_spawn: bool = False,
    ) -> None:
        if cameras_to_use is None:
            cameras_to_use = []
        self.scene_xml_path = scene_xml_path
        self.model = model
        self.camera_hz = camera_hz
        self.urdf_model = utils.URDFmodel()
        self._server_process = None
        self._cameras_to_use = cameras_to_use
        self._start_translation = start_translation
        self._start_rotation_quat = start_rotation_quat
        self._molmospaces_environment = molmospaces_environment
        self._debug_molmospaces_spawn = debug_molmospaces_spawn

        self.is_stop_called = False

        self._manager = Manager()
        self._stop_mujoco_process_event = self._manager.Event()

        self.data_proxies = MujocoServerProxies.default(self._manager)

        self._command_lock = Lock()

    def start(
        self,
        show_viewer_ui: bool = False,
        headless: bool = False,
        use_passive_viewer: bool = True,
        use_glx: bool = False,
    ) -> None:
        """
        Start the simulator

        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
            use_passive_viewer: bool, to use the passive or managed mujoco UI viewer.
            use_glx: bool, if True use default/GLX instead of EGL (for Xvfb on WSL).
        """
        self.is_stop_called = False

        mujoco_server = MujocoServer  # Headless

        if not headless:
            mujoco_server = MujocoServerPassive if use_passive_viewer else MujocoServerManaged

        if platform.system() == "Darwin" and mujoco_server is MujocoServerPassive:
            # On a mac, the process for MujocoServerPassive needs to be started with mjpython
            mjpython_path = sys.executable.replace("bin/python3", "bin/mjpython").replace("bin/python", "bin/mjpython")
            print(f"{mjpython_path=}")
            multiprocessing.set_executable(mjpython_path)

        multiprocessing.set_start_method("spawn", force=True)

        # Headless rendering: use EGL on Linux (no display needed), or rely on DISPLAY (e.g. Xvfb).
        # Only disable cameras when headless + no display + not Linux (Mac/Windows have no headless GL).
        if headless and not os.environ.get("DISPLAY"):
            if platform.system() == "Linux":
                if not use_glx:
                    os.environ.setdefault("MUJOCO_GL", "egl")  # GPU headless rendering
                cameras_for_server = self._cameras_to_use
            else:
                cameras_for_server = []  # Mac/Windows: no EGL, disable cameras
        elif headless and os.environ.get("DISPLAY"):
            cameras_for_server = self._cameras_to_use  # Xvfb or real display
        else:
            cameras_for_server = self._cameras_to_use

        # On Linux, use EGL unless --use-glx (e.g. for Xvfb on WSL to get camera images).
        if platform.system() == "Linux" and cameras_for_server and not use_glx:
            if "MUJOCO_GL" not in os.environ:
                print(
                    "Warning: Setting MUJOCO_GL=egl for camera rendering (avoids GLX/gladLoadGL errors). "
                    "Use --use-glx with Xvfb on WSL if EGL hangs."
                )
                os.environ["MUJOCO_GL"] = "egl"

        self._cameras_passed = cameras_for_server  # used so parent only waits for camera data when cameras are enabled
        self._server_process = Process(
            target=mujoco_server.launch_server,
            name="MujocoProcess",
            args=(
                self.scene_xml_path,
                self.model,
                self.camera_hz,
                show_viewer_ui,
                self._stop_mujoco_process_event,
                self.data_proxies,
                cameras_for_server,
                self._start_translation,
                self._start_rotation_quat,
                use_glx,
                self._molmospaces_environment,
                self._debug_molmospaces_spawn,
            ),
            daemon=False,  # We're gonna handle terminating this in stop_mujoco_process()
        )
        self._server_process.start()

        # Handle stopping, in all its various ways:
        signal.signal(signal.SIGTERM, lambda num, sig: self.stop())
        signal.signal(signal.SIGINT, lambda num, sig: self.stop())
        atexit.register(self.stop)

        logger.alert("Starting Stretch MuJoCo Simulator...")
        need_camera = len(self._cameras_passed) > 0
        while self.pull_status().time == 0 or (need_camera and self.pull_camera_data().time == 0):
            time.sleep(1)
            logger.warning("Still waiting to connect to the MuJoCo Simulator.")

            if not self.is_running():
                # is_running() may call stop(), which sets _server_process to None
                if self._server_process is not None:
                    self._server_process.join(timeout=2.0)
                raise RuntimeError(
                    "The MuJoCo simulator process exited before connecting. "
                    "If you have no display (SSH, Docker), use: emet serve mujoco --headless "
                    "(or robocasa with --headless). "
                    "On WSL, try: Xvfb :99 -screen 0 1024x768x24 & then DISPLAY=:99 emet serve mujoco --headless --use-glx. "
                    "Or use --no-cameras if you do not need camera images. "
                    "Check the process output above for OpenGL/EGL/GLX errors."
                )

        logger.alert("The MuJoCo Simulator is connected.")

        self.home()

    def stop(self) -> None:
        """
        This is called at exit to gracefully terminate the simulation and the Mujoco Process, and their many threads.

        Fingers-crossed we get a SIGTERM, and not a SIGKILL..
        """
        if self.is_stop_called:
            return

        self.is_stop_called = True

        try:
            simulation_time_message = self.data_proxies.get_status().time
            simulation_time_message = f" simulated runtime= {simulation_time_message:.1f}s"
        except:
            simulation_time_message = ""

        logger.error(
            f"Stopping Stretch MuJoCo Simulator...{simulation_time_message}",
        )

        self.stop_mujoco_process()

        # We're going to try to wait for threads to end. They might not gracefully stop before hitting an exception. Race conditions are rampant.
        # For example, the main thread or a thread may not be checking `sim.is_running()` and is oblivious that it should stop. Nothing we can do to stop it except sigkill.
        active_threads = threading.enumerate()
        for index, thread in enumerate(active_threads):
            if (
                thread != threading.current_thread()
                and thread != threading.main_thread()
                and not isinstance(thread, threading._DummyThread)
            ):
                logger.warning(
                    f"Stopping thread {index}/{len(active_threads) - 1}.",
                )
                thread.join(timeout=10.0)
                if thread.is_alive():
                    logger.error(
                        f"{thread.name} is not terminating. Make sure to check 'sim.is_running()' in threading loops.",
                    )

        logger.error(
            "The Stretch MuJoCo Simulator has ended. Good-bye!",
        )

    def stop_mujoco_process(self):
        """Stop the MuJoCo server process gracefully: request stop, join with timeout, then SIGTERM, then SIGKILL if needed."""
        if not self._server_process:
            return

        if not self._server_process.is_alive():
            logger.warning(
                "The MuJoCo process has already terminated. Cleaning up.",
            )
            self._server_process.join(timeout=2.0)  # Reap the dead process
            self._server_process = None
            return

        logger.error(
            "Stopping the MuJoCo process (sending stop request)...",
        )
        self._stop_mujoco_process_event.set()

        join_timeout = 10.0
        self._server_process.join(timeout=join_timeout)
        if not self._server_process.is_alive():
            logger.error("The MuJoCo process has ended.")
            self._server_process = None
            return

        logger.warning(
            f"MuJoCo process did not exit after {join_timeout}s. Sending SIGTERM.",
        )
        self._server_process.terminate()
        term_timeout = 5.0
        self._server_process.join(timeout=term_timeout)
        if not self._server_process.is_alive():
            logger.error("The MuJoCo process has ended.")
            self._server_process = None
            return

        logger.error(
            "MuJoCo process did not exit after SIGTERM. Sending SIGKILL.",
        )
        self._server_process.kill()
        self._server_process.join(timeout=2.0)
        self._server_process = None
        logger.error("The MuJoCo process has ended.")

    @require_connection
    def home(self) -> None:
        """
        Move the robot to home position
        """
        with self._command_lock:
            self.data_proxies.set_command(StatusCommand(keyframe=CommandKeyframe(name="home", trigger=True)))
        self.wait_while_is_moving(Actuators.lift)

    @require_connection
    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        with self._command_lock:
            self.data_proxies.set_command(StatusCommand(keyframe=CommandKeyframe(name="stow", trigger=True)))

        self.wait_while_is_moving(Actuators.wrist_pitch)

    def is_reached_set_position(self, actuator: str | Actuators, position_tolerance: float = 0.05):
        """
        Checks if the joint has reached a previously commanded location.

        Only listens to the `move_to` command.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.base_rotate,
            Actuators.base_translate,
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
        ]:
            raise NotImplementedError(f"Check joint reached is not supported for {actuator}.")

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            logger.warning(
                "Position check requested, but the joint was not commanded to move.",
            )
            return True

        set_position = move_command.pos

        current_position = actuator.get_position(self.pull_status())

        return bool(np.isclose(current_position, set_position, atol=position_tolerance))

    def wait_until_at_setpoint(self, actuator: str | Actuators, timeout: float = 5.0, position_tolerance: float = 0.05):
        """Blocks until the actuator reaches its previously set point."""
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        move_command = self.data_proxies.get_command().move_to.get(actuator.name)

        if not move_command:
            return True

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: self.is_reached_set_position(actuator=actuator, position_tolerance=position_tolerance),
            is_alive=self.is_running,
        ):
            pos = move_command.pos
            actual = actuator.get_position(self.pull_status())
            error = pos - actual
            logger.error(
                f"Timeout: Joint {actuator.name} did not reach {pos}. Actual: {actual:.4f} Diff: {error * 100:.4f}cm",
            )
            return False
        return True

    _last_movement_positions: dict[Actuators, float | tuple[float, float, float]] = {}

    def wait_while_is_moving(
        self,
        actuator: str | Actuators,
        timeout: float | None = 5.0,
        check_interval: float = 0.1,
        position_tolerance: float = 0.0005,
    ):
        """
        Checks position after a delay, and blocks if position has changed.
        If `timeout` is None, will block indefinitely.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        def check_if_moved():
            """Checks movement, returns True if movement is detected."""
            time.sleep(check_interval)

            if actuator in [
                Actuators.left_wheel_vel,
                Actuators.right_wheel_vel,
                Actuators.base_rotate,
                Actuators.base_translate,
            ]:
                current_position = actuator.get_position_relative(self.pull_status())
                if actuator == Actuators.left_wheel_vel or actuator == Actuators.base_translate:
                    current_position = current_position[0]
                elif actuator == Actuators.right_wheel_vel:
                    current_position = current_position[1]
                elif actuator == Actuators.base_rotate:
                    current_position = current_position[2]
            else:
                current_position = actuator.get_position(self.pull_status())

            if actuator not in self._last_movement_positions:
                self._last_movement_positions[actuator] = current_position
                return True

            last_position = self._last_movement_positions[actuator]

            is_moved = not np.isclose(current_position, last_position, atol=position_tolerance)

            self._last_movement_positions[actuator] = current_position

            return is_moved

        if not block_until_check_succeeds(
            wait_timeout=timeout,
            check=lambda: not check_if_moved(),
            is_alive=self.is_running,
        ):
            if timeout is not None:
                logger.error(
                    f"Timeout: Joint {actuator.name} is still moving after {timeout}.",
                )
            return False
        return True

    @require_connection
    def move_to(self, actuator: str | Actuators, pos: float) -> None:
        """
        Move the actuator to an absolute position.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, absolute position goal

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
            Actuators.base_rotate,
            Actuators.base_translate,
        ]:
            raise Exception(f"Cannot set an absolute position for a continuous joint {actuator.name}")

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_move_to(CommandMove(actuator_name=actuator.name, pos=pos, trigger=True))

            self.data_proxies.set_command(command)

    @require_connection
    def move_by(self, actuator: str | Actuators, pos: float):
        """
        Move the actuator by a relative amount.
        Args:
            actuator: string name of the actuator or Actuator enum instance
            pos: float, position to increment by

        Use `wait_until_at_setpoint()` or `wait_while_is_moving()` to block until the joint reaches its location.
        """
        if isinstance(actuator, str):
            actuator = Actuators[actuator]

        if actuator in [Actuators.left_wheel_vel, Actuators.right_wheel_vel]:
            logger.error(
                f"Cannot set a position for a velocity joint {actuator.name}",
            )
            raise Exception(f"Cannot set an absolute position for a continuous joint {actuator.name}")

        with self._command_lock:
            command = self.data_proxies.get_command()

            command.set_move_by(
                # We set the pos here, and not new_position, because this relative motion math is handled by mujoco_server:
                CommandMove(actuator_name=actuator.name, pos=pos, trigger=True)
            )

            self.data_proxies.set_command(command)

    @require_connection
    def teleport_base_xyt(self, x: float, y: float, theta: float) -> None:
        """Snap ``base_link`` free joint to world (x, y, theta) in the MuJoCo subprocess."""
        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_teleport_base(CommandTeleportBase(float(x), float(y), float(theta), True))
            self.data_proxies.set_command(command)

    @require_connection
    def set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """

        with self._command_lock:
            command = self.data_proxies.get_command()
            command.set_base_velocity(CommandBaseVelocity(v_linear=v_linear, omega=omega, trigger=True))

            self.data_proxies.set_command(command)

    @require_connection
    def add_world_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Add a world frame to the simulator for visualization.
        Args:
            position: tuple of (x, y, z) coordinates in the world frame
            rotation: tuple of (x, y, z) angle in radians for the rotation around each axis
        """
        with self._command_lock:
            command = self.data_proxies.get_command()
            command.coordinate_frame_arrows_viz.append(
                CommandCoordinateFrameArrowsViz(position=position, rotation=rotation, trigger=True)
            )
            self.data_proxies.set_command(command)

    @require_connection
    def get_base_pose(self):
        """Get the se(2) base pose: x, y, and theta"""
        status = self.pull_status()
        return (status.base.x, status.base.y, status.base.theta)

    @require_connection
    def get_ee_pose(self) -> np.ndarray:
        return self.get_link_pose("link_grasp_center")

    @require_connection
    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Pose of link in world frame"""
        status = self.pull_status()
        cfg = {
            "wrist_yaw": status.wrist_yaw.pos,
            "wrist_pitch": status.wrist_pitch.pos,
            "wrist_roll": status.wrist_roll.pos,
            "lift": status.lift.pos,
            "arm": status.arm.pos,
            "head_pan": status.head_pan.pos,
            "head_tilt": status.head_tilt.pos,
        }
        transform = self.urdf_model.get_transform(cfg, link_name)
        base_xyt = self.get_base_pose()
        base_4x4 = np.eye(4)
        base_4x4[:3, :3] = utils.Rz(base_xyt[2])
        base_4x4[:2, 3] = base_xyt[:2]
        world_coord = np.matmul(base_4x4, transform)
        return world_coord

    @require_connection
    def pull_camera_data(self) -> StatusStretchCameras:
        """
        Pull camera data from the simulator and return as a StatusStretchCameras
        """
        return self.data_proxies.get_cameras()

    @require_connection
    def pull_sensor_data(self) -> StatusStretchSensors:
        """
        Pull sensor data from the simulator and return as a StatusStretchSensors
        """
        return self.data_proxies.get_sensors()

    @require_connection
    def pull_status(self) -> StatusStretchJoints:
        """
        Pull robot joint states from the simulator and return as a StatusStretchJoints
        """
        return self.data_proxies.get_status()

    @require_connection
    def pull_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        """
        Pull robot joint limuts from the simulator and return as a dict
        """
        return self.data_proxies.get_joint_limits()

    def is_mujoco_process_dead_or_stopevent_triggered(self):
        return (
            self._server_process is None
            or not self._server_process.is_alive()
            or self._stop_mujoco_process_event.is_set()
        )

    def is_running(self) -> bool:
        """
        Check if the simulator and mujoco are running, or if the stopevent signal has been triggered.

        Side-effect here is that if the mujoco process is terminated or the stopevent is triggered, `self.stop()` is called.
        """
        if self.is_mujoco_process_dead_or_stopevent_triggered():
            # Send the signal to stop the program:
            self.stop()
            return False

        return not self.is_stop_called
