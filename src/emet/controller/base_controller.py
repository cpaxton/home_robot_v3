# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Base class for robot controllers. Shared by InstanceMemoryController and DynamemController."""

from abc import ABC, abstractmethod
from typing import Any

from emet.core.interfaces import Observations
from emet.core.parameters import Parameters
from emet.core.robot import AbstractRobotClient
from emet.mapping.instance import Instance
from emet.motion import PlanResult
from emet.utils.logger import Logger

logger = Logger(__name__)


def _parse_parameters(
    parameters: Parameters | dict[str, Any] | None,
    default_path: str | None = "default_planner.yaml",
) -> Parameters:
    """Parse parameters from dict, Parameters, or None (load from default)."""
    if parameters is None:
        if default_path is None:
            raise RuntimeError("parameters is None and no default_config_path provided")
        from emet.core.parameters import get_parameters

        return get_parameters(default_path)
    if isinstance(parameters, dict):
        return Parameters(**parameters)
    if isinstance(parameters, Parameters):
        return parameters
    raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")


class BaseController(ABC):
    """Base class for robot controllers. Provides common state and methods."""

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters | dict[str, Any] | None = None,
        use_instance_memory: bool = False,
        realtime_updates: bool = False,
        default_config_path: str | None = "default_planner.yaml",
    ):
        self.robot = robot
        self.parameters = _parse_parameters(parameters, default_config_path)

        self.pos_err_threshold = self.parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = self.parameters["trajectory_rot_err_threshold"]

        self.current_receptacle: Instance | None = None
        self.current_object: Instance | None = None
        self.target_object: str | None = None
        self.target_receptacle: str | None = None

        self._is_match_threshold = self.parameters.get("encoder_args/feature_match_threshold", 0.05)
        self._grasp_match_threshold = self.parameters.get("encoder_args/grasp_feature_match_threshold", 0.05)

        self._default_expand_frontier_size = self.parameters["motion_planner"]["frontier"][
            "default_expand_frontier_size"
        ]
        self._frontier_min_dist = self.parameters["motion_planner"]["frontier"]["min_dist"]
        self._frontier_step_dist = self.parameters["motion_planner"]["frontier"]["step_dist"]
        self._manipulation_radius = self.parameters["motion_planner"]["goals"]["manipulation_radius"]
        self._voxel_size = self.parameters.get("voxel_size", 0.05)

        self.obs_count = 0
        self.obs_history: list[Observations] = []

        self.guarantee_instance_is_reachable = self.parameters.guarantee_instance_is_reachable
        self.use_scene_graph = self.parameters["use_scene_graph"]

        self._use_instance_memory = use_instance_memory
        self._realtime_updates = realtime_updates
        self._sweep_head_on_update = self.parameters.get("agent/sweep_head_on_update", False)

        self.scene_graph = None
        self._previous_goal = None
        self._running = True

        self.reset_object_plans()

    def reset_object_plans(self) -> None:
        """Clear stored object planning information."""
        self._object_attempts: dict[int, int] = {}
        self._cached_plans: dict[int, PlanResult] = {}
        self.unreachable_instances = set()

    def get_robot(self) -> AbstractRobotClient:
        """Return the robot in use by this controller."""
        return self.robot

    @property
    def feature_match_threshold(self) -> float:
        """Return the feature match threshold."""
        return self._is_match_threshold

    @property
    def grasp_feature_match_threshold(self) -> float:
        """Return the feature match threshold for grasping."""
        return self._grasp_match_threshold

    @property
    def voxel_size(self) -> float:
        """Return the voxel size in meters."""
        return self._voxel_size

    def start(
        self,
        goal: str | None = None,
        visualize_map_at_start: bool = False,
        can_move: bool = True,
        verbose: bool = True,
    ) -> None:
        """Start the robot and put it in a ready state (nav posture, navigation mode)."""
        started = self.robot.start()
        if not started:
            client = type(self.robot).__name__
            raise RuntimeError(
                f"Robot failed to start ({client}). No ZMQ observations in time — start the "
                "MuJoCo or robot server first (e.g. `emet serve mujoco --robot rby1 --headless`, "
                "or `--robot stretch` for Stretch). The agent `--robot` must match the server "
                "(default agent robot is stretch; use `--robot rby1` if the server is rby1). "
                "Check IP, `--port-offset`, and firewall."
            )
        if verbose:
            logger.debug("ZMQ connection to robot started.")
        if can_move:
            self.robot.switch_to_manipulation_mode()
            self.robot.open_gripper()
            if verbose:
                logger.debug("Sending arm to home...")
            self.robot.move_to_nav_posture()
            if verbose:
                logger.debug("... done.")
        self.robot.switch_to_navigation_mode()
        if verbose:
            logger.debug("- Update map after switching to navigation posture")

    @abstractmethod
    def get_voxel_map(self):
        """Return the voxel map in use by this controller. Subclasses implement."""
        pass

    def robot_say(self, msg: str) -> str:
        """Say a message: optional TTS on robot, send to Discord if bot is set, return for chatbot.

        Subclasses can override. Base implementation: strip quotes, call robot.say if available,
        push to Discord via self.discord_bot if set, and return the message string.
        """
        msg = msg.strip("'\"").strip()
        if hasattr(self.robot, "say") and callable(self.robot.say):
            self.robot.say(msg)
        discord_bot = getattr(self, "discord_bot", None)
        if discord_bot is not None and hasattr(discord_bot, "push_task_to_all_channels"):
            discord_bot.push_task_to_all_channels(message=msg)
        return msg
