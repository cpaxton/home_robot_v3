# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import colored

from emet.agent.env_flags import env_base_rotate_only
from emet.config.embodied_agent_config import (
    EmbodiedAgentConfig,
    coerce_embodied_agent_for_memory_backend,
    normalize_memory_backend,
)
from emet.controller.operations import GraspObjectOperation
from emet.controller.task.emote import EmoteTask
from emet.controller.task.pickup.hand_over_task import HandOverTask
from emet.core import AbstractRobotClient, Parameters
from emet.memory.utils import print_memory_saved_help
from emet.perception import create_semantic_sensor
from emet.utils.image import numpy_image_to_bytes
from emet.utils.logger import Logger

if TYPE_CHECKING:
    from emet.controller.controller_dynamem import RobotAgent

logger = Logger(__name__)

# Executor commands that translate the base (blocked when EMET_BASE_ROTATE_ONLY=1).
_BASE_DRIVE_COMMANDS = frozenset(
    {
        "explore",
        "find",
        "move_forward",
        "go_home",
        "pickup",
        "place",
        "hand_over",
    }
)


def compute_tilt(camera_xyz, target_xyz):
    """
    a util function for computing robot head tilts so the robot can look at the target object after navigation
    - camera_xyz: estimated (x, y, z) coordinates of camera
    - target_xyz: estimated (x, y, z) coordinates of the target object
    """
    if not isinstance(camera_xyz, np.ndarray):
        camera_xyz = np.array(camera_xyz)
    if not isinstance(target_xyz, np.ndarray):
        target_xyz = np.array(target_xyz)
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))


class DynamemTaskExecutor:
    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters,
        match_method: str = "feature",
        visual_servo: bool = False,
        device_id: int = 0,
        output_path: str | None = None,
        server_ip: str | None = "127.0.0.1",
        skip_confirmations: bool = True,
        explore_iter: int = 5,
        mllm: bool = False,
        manipulation_only: bool = False,
        cpu_only: bool = False,
        eqa: bool = False,
        defer_eqa_vllm: bool = False,
        discord_bot=None,
        embodied_agent: EmbodiedAgentConfig | None = None,
        memory_backend: str = "dynagraph",
    ) -> None:
        """Initialize the executor.

        *memory_backend*: ``dynagraph`` (default), ``graph_eqa``, ``dynamem``, or ``open_vocab``.
        """
        self.robot = robot
        self.parameters = parameters
        self.discord_bot = discord_bot
        self.cpu_only = cpu_only
        self._last_memory_save_path = None  # set when memory is saved (e.g. after rotate_in_place)
        self._last_sim_picked_body: str | None = None  # GT body after sim-teleport pickup
        self._last_exec_ok = True  # False if last __call__ had pickup/place failure
        self._manip_mode = "teleport"
        self._manip_collision = "none"
        self._manip_planner = "rrt_connect"
        agent_cfg: dict = {}
        if hasattr(parameters, "get"):
            raw = parameters.get("agent")
            if isinstance(raw, dict):
                agent_cfg = raw
        elif isinstance(parameters, dict):
            agent_cfg = parameters.get("agent") or {}
            if not isinstance(agent_cfg, dict):
                agent_cfg = {}
        self._manip_mode = str(agent_cfg.get("manip_mode") or "teleport")
        self._manip_collision = str(agent_cfg.get("manip_collision") or "none")
        self._manip_planner = str(agent_cfg.get("manip_planner") or "rrt_connect")
        from emet.motion.arm_rrt import resolve_agent_manip_planner
        from emet.simulation.sim_manipulation import resolve_agent_manip_collision, resolve_agent_manip_mode

        self._manip_mode = resolve_agent_manip_mode(config_mode=self._manip_mode, visual_servo=bool(visual_servo))
        self._manip_collision = resolve_agent_manip_collision(config_mode=self._manip_collision)
        self._manip_planner = resolve_agent_manip_planner(config_mode=self._manip_planner)
        self.memory_backend = normalize_memory_backend(memory_backend)
        self.embodied_agent = coerce_embodied_agent_for_memory_backend(
            embodied_agent, self.memory_backend
        )
        # If there is no GPU, we have to use CPU
        if not torch.cuda.is_available():
            print("Setting up to use CPU as there is no GPU!")
            self.cpu_only = True

        # Other parameters
        self.visual_servo = visual_servo
        self.match_method = match_method
        self.skip_confirmations = skip_confirmations
        self.explore_iter = explore_iter

        self.manipulation_only = manipulation_only

        # Do type checks
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")

        # Create semantic sensor if visual servoing is enabled
        logger.debug("- Create semantic sensor if visual servoing is enabled")
        if self.visual_servo:
            self.parameters["detection"]["module"] = "yoloe" if self.cpu_only else "owlsam"
            self.semantic_sensor = create_semantic_sensor(
                parameters=self.parameters,
                device_id=device_id,
                verbose=False,
            )
        else:
            self.parameters["encoder"] = None
            self.semantic_sensor = None

        logger.debug("- Start robot agent with data collection")
        self.agent = self._build_agent(
            output_path=output_path,
            server_ip=server_ip,
            mllm=mllm,
            eqa=eqa,
            defer_eqa_vllm=defer_eqa_vllm,
        )
        self.agent.start()

        # Create grasp object operation
        if self.visual_servo:
            self.grasp_object = GraspObjectOperation(
                "grasp_the_object",
                self.agent,
            )
        else:
            self.grasp_object = None

        # Task stuff
        self.emote_task = EmoteTask(self.agent)

    def _build_agent(
        self,
        *,
        output_path: str | None,
        server_ip: str | None,
        mllm: bool,
        eqa: bool,
        defer_eqa_vllm: bool,
    ):
        """Construct Dynamem / GraphEQA / Dynagraph controller for the agent loop."""
        from emet.eval.stack import build_memory_agent

        return build_memory_agent(
            robot=self.robot,
            parameters=self.parameters,
            backend=self.memory_backend,
            harness="interactive",
            semantic_sensor=self.semantic_sensor,
            log=output_path,
            server_ip=server_ip,
            mllm=mllm,
            manipulation_only=self.manipulation_only,
            cpu_only=self.cpu_only,
            eqa=eqa,
            defer_eqa_vllm=defer_eqa_vllm,
            embodied_agent=self.embodied_agent,
        )

    def _find(self, target_object: str) -> np.ndarray:
        """Find an object. This is a helper function for the main loop.

        Args:
            target_object: The object to find.

        Returns:
            The point where the object is located.
        """
        self.robot.switch_to_navigation_mode()
        point = self.agent.navigate(target_object)
        # `filename` = None means write to default log path (the datetime you started to run the process)
        self.agent.voxel_map.write_to_pickle(filename=None)
        if point is None:
            logger.error(f"Navigation Failure: Could not find the object {target_object}")
            return None
        cv2.imwrite(target_object + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])
        self.robot.switch_to_navigation_mode()
        xyt = self.robot.get_base_pose()
        xyt[2] = xyt[2] + np.pi / 2
        self.robot.move_base_to(xyt, blocking=True)
        return point

    def _can_sim_gt_manip(self) -> bool:
        from emet.simulation.sim_manipulation import can_use_sim_gt_manip

        return can_use_sim_gt_manip(
            self.robot,
            manip_mode=self._manip_mode,
            visual_servo=self.visual_servo,
        )

    def _voxel_map_for_manip(self) -> Any | None:
        agent = getattr(self, "agent", None)
        if agent is None:
            return None
        get_vm = getattr(agent, "get_voxel_map", None)
        return get_vm() if callable(get_vm) else getattr(agent, "voxel_map", None)

    def _pickup(
        self,
        target_object: str,
        point: np.ndarray | None = None,
        skip_confirmations: bool = False,
    ) -> bool:
        """Pick up an object. Returns True on success (Stretch path always True).

        Stretch visual-servo / AnyGrasp still returns True unconditionally — see TODO.md
        (Stretch / AnyGrasp ``_pickup`` / ``_place`` always return True).
        """
        from emet.simulation.sim_manipulation import (
            prefer_kinematic_manip,
            prefer_sim_teleport_manip,
            sim_teleport_pickup,
        )

        if prefer_kinematic_manip(self.robot, manip_mode=self._manip_mode, visual_servo=self.visual_servo):
            from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

            exe = KinematicPickPlaceExecutor(
                self.robot,
                manip_collision=self._manip_collision,
                manip_planner=self._manip_planner,
                voxel_map=self._voxel_map_for_manip(),
            )
            self._kinematic_executor = exe
            result = exe.grasp_only(target_object)
            if result.success:
                self._last_sim_picked_body = result.object_body
                logger.info(f"Kinematic grasp: {target_object!r} body={result.object_body!r} err={result.grasp_err_m}")
                self.robot.say("Picked up the " + str(target_object) + ".")
                return True
            self._last_sim_picked_body = None
            logger.error(f"Kinematic grasp failed: {result.message}")
            self.robot.say("I could not pick up the " + str(target_object) + ".")
            return False

        # Teleport (default) or fallback when kinematic was requested but server lacks caps.
        if prefer_sim_teleport_manip(self.robot, visual_servo=self.visual_servo):
            if self._manip_mode == "kinematic":
                logger.warning("manip_mode=kinematic but server lacks kinematic_manip; falling back to teleport.")
            body = sim_teleport_pickup(self.robot, target_object)
            if body:
                self._last_sim_picked_body = body
                logger.info(f"Sim teleport pickup: {target_object!r} body={body!r}")
                self.robot.say("Picked up the " + str(target_object) + ".")
                return True
            self._last_sim_picked_body = None
            logger.error(
                f"Sim teleport pickup failed for {target_object!r} "
                "(no matching freejoint GT body, or pose verify failed)."
            )
            self.robot.say("I could not pick up the " + str(target_object) + ".")
            return False

        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        if point is not None:
            theta = compute_tilt(camera_xyz, point)
        else:
            theta = -0.6

        # Grasp the object using operation if it's available
        if self.grasp_object is not None:
            self.robot.say("Grasping the " + str(target_object) + ".")
            print("Using operation to grasp object:", target_object)
            print(" - Point:", point)
            print(" - Theta:", theta)
            state = self.robot.get_six_joints()
            state[1] = 1.0
            self.robot.arm_to(state, blocking=True)
            self.grasp_object(
                target_object=target_object,
                object_xyz=point,
                match_method=self.match_method,
                show_object_to_grasp=False,
                show_servo_gui=False,
                delete_object_after_grasp=False,
            )
            # This retracts the arm
            self.robot.move_to_nav_posture()
        else:
            # Otherwise, use the self.agent's manipulation method
            # This is from OK Robot
            print("Using self.agent to grasp object:", target_object)
            self.agent.manipulate(target_object, theta, skip_confirmation=skip_confirmations)
        self.robot.look_front()
        return True

    def _place(self, target_receptacle: str, point: np.ndarray | None) -> bool:
        """Place an object. Returns True on success (Stretch path always True).

        Stretch ``agent.place`` still returns True unconditionally — see TODO.md
        (Stretch / AnyGrasp ``_pickup`` / ``_place`` always return True).
        """
        from emet.simulation.sim_manipulation import (
            prefer_kinematic_manip,
            prefer_sim_teleport_manip,
            sim_teleport_place,
        )

        if prefer_kinematic_manip(self.robot, manip_mode=self._manip_mode, visual_servo=self.visual_servo):
            exe = getattr(self, "_kinematic_executor", None)
            if exe is None:
                from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

                exe = KinematicPickPlaceExecutor(
                    self.robot,
                    manip_collision=self._manip_collision,
                    manip_planner=self._manip_planner,
                    voxel_map=self._voxel_map_for_manip(),
                )
                self._kinematic_executor = exe
            result = exe.place_only(target_receptacle, object_gt_body=self._last_sim_picked_body)
            if result.success:
                logger.info(
                    f"Kinematic place: body={result.object_body!r} onto {target_receptacle!r} err={result.place_err_m}"
                )
                self.robot.say("Placing object on the " + str(target_receptacle) + ".")
                self._last_sim_picked_body = None
                return True
            logger.error(f"Kinematic place failed: {result.message}")
            self.robot.say("I could not place the object on the " + str(target_receptacle) + ".")
            return False

        if prefer_sim_teleport_manip(self.robot, visual_servo=self.visual_servo):
            if self._manip_mode == "kinematic":
                logger.warning("manip_mode=kinematic but server lacks kinematic_manip; falling back to teleport.")
            ok = sim_teleport_place(
                self.robot,
                target_receptacle,
                object_gt_body=self._last_sim_picked_body,
            )
            if ok:
                logger.info(f"Sim teleport place: body={self._last_sim_picked_body!r} onto {target_receptacle!r}")
                self.robot.say("Placing object on the " + str(target_receptacle) + ".")
                self._last_sim_picked_body = None
                return True
            logger.error(
                f"Sim teleport place failed for receptacle {target_receptacle!r} "
                f"(held body={self._last_sim_picked_body!r})."
            )
            self.robot.say("I could not place the object on the " + str(target_receptacle) + ".")
            return False

        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        if point is not None:
            theta = compute_tilt(camera_xyz, point)
        else:
            theta = -0.6

        self.robot.say("Placing object on the " + str(target_receptacle) + ".")
        # If you run this stack with visual servo, run it locally
        self.agent.place(target_receptacle, init_tilt=theta, local=self.visual_servo)
        self.robot.move_to_nav_posture()
        return True

    def _take_picture(self, channel=None) -> None:
        """Take a picture with the head camera. Optionally send it to Discord."""

        obs = self.robot.get_observation()
        if channel is None:
            # Just save it to the disk
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(obs.rgb).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel, message="Head camera:", content=numpy_image_to_bytes(obs.rgb)
            )

    def _take_ee_picture(self, channel=None) -> None:
        """Take a picture of the end effector."""

        obs = self.robot.get_servo_observation()
        ee = None if obs is None else getattr(obs, "ee_rgb", None)
        if ee is None:
            logger.warning("take_ee_picture: no ee_rgb on servo observation (common on Mars).")
            return
        arr = np.asarray(ee)
        if channel is None:
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(arr).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel,
                message="End effector camera:",
                content=numpy_image_to_bytes(arr),
            )

    def _hand_over(self) -> None:
        """Create a task to find a person, navigate to them, and extend the arm toward them"""
        logger.alert("[Pickup task] Hand Over")

        # After the robot has started...
        try:
            hand_over_task = HandOverTask(self.agent)
            task = hand_over_task.get_task()
        except Exception as e:
            print(f"Error creating task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def __call__(self, response: list[tuple[str, str]], channel=None) -> bool:
        """Execute the list of commands given by the LLM bot.

        Args:
            response: A list of tuples, where the first element is the command and the second is the argument.

        Returns:
            True if we should keep going, False if we should stop (quit).

        Task success for the last batch is in ``_last_exec_ok`` (False if pickup/place
        failed). Agent loop uses that for tool summaries without treating failure as quit.
        """
        i = 0
        self._last_exec_ok = True

        if response is None or len(response) == 0:
            logger.error("No commands to execute!")
            said = self.agent.robot_say("I'm sorry, I didn't understand that.")
            if said:
                print(colored("Robot:", "blue"), said)
            return True

        # Dynamem aims to life long robot, we should not reset the robot's memory.
        # logger.info("Resetting agent...")
        # self.agent.reset()

        # Loop over every command we have been given
        # Pull out pickup and place as a single arg if they are in a row
        # Else, execute things as they come
        while i < len(response):
            command, args = response[i]
            logger.info(f"Command: {i} {command} {args}")
            if env_base_rotate_only() and command in _BASE_DRIVE_COMMANDS:
                logger.warning(
                    f"EMET_BASE_ROTATE_ONLY: skipping command {command!r} "
                    "(yaw-only / in-place scan allowed; no XY drive)."
                )
                i += 1
                continue
            if command == "say":
                # Use TTS to say the text
                logger.info(f"Saying: {args}")
                self.agent.robot_say(args)
                if channel is not None:
                    # Optionally strip quotes from args
                    if args[0] == '"' and args[-1] == '"':
                        args = args[1:-1]
                    self.discord_bot.send_message(channel=channel, message=args)
            elif command == "pickup":
                logger.info(f"[Pickup task] Pickup: {args}")
                target_object = args

                # Navigation

                # Either we wait for users to confirm whether to run navigation, or we just directly control the robot to navigate.
                if not self.manipulation_only and (
                    self.skip_confirmations
                    or (not self.skip_confirmations and input("Do you want to run navigation? [Y/n]: ").upper() != "N")
                ):
                    self.robot.move_to_nav_posture()
                    point = self._find(args)
                # Or the user explicitly tells that he or she does not want to run navigation.
                else:
                    point = None

                # Pick up (sim GT kinematic/teleport can run without a nav point).
                pickup_ok = True
                if self.skip_confirmations:
                    if point is not None or self._can_sim_gt_manip():
                        if point is None:
                            logger.info(f"Nav find missed {args!r}; running sim GT pickup without visual nav point.")
                        pickup_ok = self._pickup(target_object, point=point)
                    else:
                        logger.error("Could not find the object.")
                        self.robot.say("I could not find the " + str(args) + ".")
                        pickup_ok = False
                else:
                    if input("Do you want to run picking? [Y/n]: ").upper() != "N":
                        pickup_ok = self._pickup(target_object, point=point)
                    else:
                        logger.info("Skip picking!")
                        i += 1
                        continue

                if not pickup_ok:
                    self._last_exec_ok = False
                    i += 1
                    # Do not run a following place in the same command batch.
                    if i < len(response) and response[i][0] == "place":
                        logger.error("Skipping place after failed pickup.")
                        i += 1
                    continue

            elif command == "place":
                logger.info(f"[Pickup task] Place: {args}")
                target_object = args

                # Navigation

                # Either we wait for users to confirm whether to run navigation, or we just directly control the robot to navigate.
                if not self.manipulation_only and (
                    self.skip_confirmations
                    or (not self.skip_confirmations and input("Do you want to run navigation? [Y/n]: ").upper() != "N")
                ):
                    point = self._find(args)
                # Or the user explicitly tells that he or she does not want to run navigation.
                else:
                    point = None

                # Placing
                place_ok = True
                if self.skip_confirmations:
                    if point is not None or self._can_sim_gt_manip():
                        if point is None:
                            logger.info(f"Nav find missed {args!r}; running sim GT place without visual nav point.")
                        place_ok = self._place(target_object, point=point)
                    else:
                        logger.error("Could not find the object.")
                        self.robot.say("I could not find the " + str(args) + ".")
                        place_ok = False
                else:
                    if input("Do you want to run placement? [Y/n]: ").upper() != "N":
                        place_ok = self._place(target_object, point=point)
                    else:
                        logger.info("Skip placing!")
                        i += 1
                        continue

                if not place_ok:
                    self._last_exec_ok = False
            elif command == "hand_over":
                self._hand_over()
            elif command == "wave":
                logger.info("[Pickup task] Waving.")
                self.agent.move_to_manip_posture()
                self.emote_task.get_task("wave").run()
                self.agent.move_to_manip_posture()
            elif command == "rotate_in_place":
                logger.info("Rotate in place to scan environments.")
                self.agent.rotate_in_place()
                from emet.memory.lifelong import save_lifelong_checkpoint

                save_dir = getattr(getattr(self.agent, "voxel_map", None), "log", None) or getattr(
                    self.agent, "log", "saved_memory"
                )
                save_lifelong_checkpoint(
                    self.agent,
                    save_dir,
                    save_voxel_pickle=True,
                    memory_backend=self.memory_backend,
                )
                self._last_memory_save_path = save_dir
                print_memory_saved_help(save_dir)
            elif command == "rotate_base":
                try:
                    deg = float(args)
                except (TypeError, ValueError):
                    logger.error(f"rotate_base: bad degrees {args!r}")
                    deg = 90.0
                logger.info(f"Rotate base by {deg} degrees.")
                if hasattr(self.agent, "rotate_base_degrees"):
                    self.agent.rotate_base_degrees(deg)
                else:
                    self.robot.move_base_to(
                        [0.0, 0.0, float(np.deg2rad(deg))],
                        relative=True,
                        blocking=True,
                    )
            elif command == "move_forward":
                try:
                    meters = float(args)
                except (TypeError, ValueError):
                    logger.error(f"move_forward: bad meters {args!r}")
                    meters = 0.5
                logger.info(f"Move forward {meters} m.")
                if hasattr(self.agent, "move_forward_meters"):
                    self.agent.move_forward_meters(meters)
                else:
                    self.robot.move_base_to([float(meters), 0.0, 0.0], relative=True, blocking=True)
            elif command == "go_home":
                logger.info("[Pickup task] Going home.")
                if self.agent.get_voxel_map().is_empty():
                    logger.warning("No map data available. Cannot go home.")
                else:
                    self.agent.go_home()
            elif command == "explore":
                logger.info("[Pickup task] Exploring.")
                if hasattr(self.agent, "announce_action"):
                    self.agent.announce_action(f"Exploring… ({self.explore_iter} steps)")
                for i in range(self.explore_iter):
                    if hasattr(self.agent, "announce_motion_progress"):
                        self.agent.announce_motion_progress(f"Exploring… step {i + 1}/{self.explore_iter}")
                    elif hasattr(self.agent, "announce_action"):
                        self.agent.announce_action(f"Exploring… step {i + 1}/{self.explore_iter}", discord=False)
                    self.agent.run_exploration()
                if hasattr(self.agent, "announce_motion_progress"):
                    self.agent.announce_motion_progress("Exploring… done")
            elif command == "find":
                logger.info(f"[Pickup task] Finding {args}.")
                point = self._find(args)
            elif command == "nod_head":
                logger.info("[Pickup task] Nodding head.")
                self.emote_task.get_task("nod_head").run()
            elif command == "shake_head":
                logger.info("[Pickup task] Shaking head.")
                self.emote_task.get_task("shake_head").run()
            elif command == "avert_gaze":
                logger.info("[Pickup task] Averting gaze.")
                self.emote_task.get_task("avert_gaze").run()
            elif command == "quit":
                logger.info("[Pickup task] Quitting.")
                self.robot.stop()
                return False
            elif command == "take_picture":
                self._take_picture(channel)
            elif command == "take_ee_picture":
                self._take_ee_picture(channel)
            elif command == "end":
                logger.info("[Pickup task] Ending.")
                break
            else:
                logger.error(f"Skipping unknown command: {command}")

            i += 1
        # If we did not explicitly receive a quit command, we are not yet done.
        return True


class EQAExecuter:
    def __init__(self, agent: RobotAgent, discord_bot=None) -> None:
        """
        Initialize the executor. Make sure EQA module can be used in the same way as DynaMem module
        TODO: Itegrate this module with DynaMem
        """

        self.agent = agent
        self.discord_bot = discord_bot

    def rotate_in_place(self):
        self.agent.rotate_in_place()

    def __call__(self, response: str | list[tuple[str, str]], channel=None) -> tuple[str, list[Image.Image]]:
        """Run EQA for one user question (string) or Discord-style payload (list of tuples)."""
        discord_text, relevant_images = self.agent.run_eqa(response)
        if channel is not None:
            self.discord_bot.send_message(
                channel=channel,
                message=discord_text,
                content=numpy_image_to_bytes(relevant_images),
            )
        return discord_text, relevant_images
