# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os

import mujoco
import mujoco._callbacks
import mujoco._enums
import mujoco._functions
import mujoco._render
import mujoco.viewer

from emet.simulation.stretch_mujoco.utils import override

try:
    from mujoco import MjData, MjModel
except ImportError:
    from mujoco._structs import MjData, MjModel

from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer


class MujocoServerManaged(MujocoServer):
    """
    Use `MujocoServerManaged.launch_server()` to start the simulator.

    This uses the mujoco managed viewer.

    https://mujoco.readthedocs.io/en/stable/python.html#managed-viewer
    """

    @override
    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # We're using the managed viewer, and don't have access to the UI thread, so use the camera thread to manage camera rendering:
        self.set_camera_manager(
            use_camera_thread=True,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        self._run_ui_simulation(show_viewer_ui)

    @override
    def _run_ui_simulation(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        mujoco._callbacks.set_mjcb_control(self._managed_viewer_loop)
        mujoco.viewer.launch(
            self.mjmodel,
            show_left_ui=show_viewer_ui,
            show_right_ui=show_viewer_ui,
        )

    def _managed_viewer_loop(self, model: MjModel, data: MjData):

        if self._is_requested_to_stop():
            self.close()

            os.kill(os.getpid(), 9)

        self._ctrl_callback(model, data)
