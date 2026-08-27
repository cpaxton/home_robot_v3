# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""DynaMem robot controller facade.

Methods live in sibling modules and are bound onto this class. Subclasses
(GraphEQA / Dynagraph / LazyGraph) keep inheriting from ``DynamemController``.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
import zmq

from emet.audio.text_to_speech import PiperTextToSpeech
from emet.config.embodied_agent_config import EmbodiedAgentConfig, legacy_embodied_agent_off
from emet.controller.base_controller import BaseController
from emet.controller.generic_zmq_client import GenericZmqClient
from emet.controller.manipulation.dynamem_manipulation.dynamem_manipulation import (
    DynamemManipulationWrapper as ManipulationWrapper,
)
from emet.controller.zmq_client import StretchZmqClient
from emet.core.parameters import Parameters
from emet.core.robot import AbstractRobotClient
from emet.perception.depth.lingbot_estimator import LingBotDepthEstimator
from emet.perception.wrapper import OvmmPerception
from emet.utils.bind_methods import bind_module_methods
from emet.utils.logger import Logger
from emet.visualization.rerun import NullVisualizer

from . import describe, eqa, look, manipulation, mapping, navigation, perception, pose
from .constants import DYNAMEM_HEAD_SETTLE_S, _env_truthy

logger = Logger(__name__)


class DynamemController(BaseController):
    """
    DynaMem robot controller. Extends base with DynaMem-specific mapping and manipulation.
    https://dynamem.github.io
    """

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters | dict[str, Any],
        semantic_sensor: OvmmPerception | None = None,
        save_rerun: bool = False,
        use_instance_memory: bool = False,
        realtime_updates: bool = False,
        re: int = 3,
        manip_port: int = 5557,
        log: str | None = None,
        server_ip: str | None = "127.0.0.1",
        mllm: bool = False,
        manipulation_only: bool = False,
        cpu_only: bool = False,
        eqa: bool = False,
        defer_eqa_vllm: bool = False,
        embodied_agent: EmbodiedAgentConfig | None = None,
    ):
        super().__init__(
            robot=robot,
            parameters=parameters,
            use_instance_memory=use_instance_memory,
            realtime_updates=realtime_updates,
            default_config_path=None,
        )
        self.semantic_sensor = semantic_sensor
        # StretchZmqClient and GenericZmqClient set ``_rerun`` when Rerun is enabled; otherwise NullVisualizer.
        self.rerun_visualizer = getattr(self.robot, "_rerun", None) or NullVisualizer()
        # Last navigation / EQA markdown for Rerun ``robot_monologue``; ``update()`` appends live status.
        self._rerun_monologue_base = ""
        # Human gate before execute_trajectory (CLI ``--confirm-nav`` / ``EMET_CONFIRM_NAV``).
        self.confirm_navigation = False
        self.nav_confirm_timeout_s: float | None = None
        self._nav_confirm_input_queue = None
        self._nav_confirm_auto_yes = False
        self._last_nav_plan = None
        self.setup_custom_blueprint()

        self.mllm = mllm
        self.manipulation_only = manipulation_only
        self.eqa = eqa
        self.defer_eqa_vllm = defer_eqa_vllm
        self.owl_sam_detector = None

        self.embodied_agent = embodied_agent if embodied_agent is not None else legacy_embodied_agent_off()
        self._open_vocab_sg_processor = None
        self.graph_memory = None
        self.sensor_builder = None
        self._last_nav_attempt = None
        self._episode_diagnostics_recorder = None
        self._graph_eqa_use_instance_graph = True
        self._graph_eqa_use_sensor_perception = True
        self._graph_dedup_xy_m = 0.0

        self.cpu_only = cpu_only
        if self.cpu_only:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._depth_source = str(self.parameters.get("depth_source", "sensor")).lower()
        self._da3_estimator = None
        self._da3_infer_every_n = max(1, int(self.parameters.get("da3_infer_every_n", 2) or 1))
        self._da3_last_depth: np.ndarray | None = None
        self._da3_use_stereo = bool(self.parameters.get("da3_stereo", False))
        self._lingbot_estimator: LingBotDepthEstimator | None = None
        self._lingbot_infer_every_n = max(1, int(self.parameters.get("lingbot_infer_every_n", 2) or 1))
        self._lingbot_last_depth: np.ndarray | None = None
        self._lingbot_last_pose: np.ndarray | None = None
        self._lingbot_use_pose = bool(self.parameters.get("lingbot_use_pose", True))
        # Heavy perception (YoloE detection + SigLIP dense features + instance memory
        # + graph update) runs every N updates; occupancy/clearance update every frame
        # so navigation is always current. The VLM/graph verification reads the latest
        # full-perception observation, so a skipped frame only defers object recall by
        # one cadence — a huge wall-time cut in teleport eval (each update was ~15-20s).
        self._perception_every_n = max(1, int(self.parameters.get("perception_every_n", 2) or 1))
        self._debug_perfect_sensor_depth = bool(
            self.parameters.get("debug_perfect_sensor_depth", False)
        ) or _env_truthy("EMET_DYNAMEM_PERFECT_DEPTH")
        if self._debug_perfect_sensor_depth:
            logger.info(
                "Dynamem: debug perfect sensor depth — when ZMQ observation includes depth, it is used for "
                "mapping and DA3 is skipped (YAML ``debug_perfect_sensor_depth: true`` or EMET_DYNAMEM_PERFECT_DEPTH=1)."
            )

        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        if log is None:
            current_datetime = datetime.now()
            self.log = os.path.join(logs_dir, "memory_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            self.log = os.path.join(logs_dir, log) if not os.path.isabs(log) else log

        self.create_obstacle_map(parameters)
        logger.info("Agent init: obstacle map ready (encoder/detector loaded)")

        if bool(parameters.get("enable_tts", True)):
            logger.info("Agent init: loading Piper TTS")
            self.tts = PiperTextToSpeech()
        else:
            self.tts = None
            logger.info("Agent init: TTS disabled")
        logger.info("Agent init: starting robot ZMQ streams")

        context = zmq.Context()
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://" + server_ip + ":" + str(manip_port))

        if re == 1 or re == 2:
            stretch_gripper_max = 0.3
            end_link = "link_straight_gripper"
        else:
            stretch_gripper_max = 0.64
            end_link = "link_gripper_s3_body"
        self.transform_node = end_link
        # ManipulationWrapper.__init__ calls get_pan_tilt(); ZMQ clients use start_immediately=False in
        # run_agent/run_dynamem so streams must be started here before any joint reads.
        if isinstance(self.robot, (StretchZmqClient, GenericZmqClient)):
            if not self.robot.start():
                raise RuntimeError(
                    "Robot ZMQ client did not connect (no full observation + state within the startup timeout). "
                    "Start the sim first with the same robot and ports, e.g. "
                    "`emet serve mujoco` (Stretch), `emet serve mujoco --robot innate_mars`, or "
                    "`emet serve mujoco --robot rby1` / MolmoSpaces merge, "
                    "then run the agent. Match `--port-offset` on both sides and use the same `--robot` as serve. "
                    "If the sim is already running, check server logs: missing camera/GL errors can make "
                    "observations None so nothing is published on the observation socket."
                )
        logger.info("Agent init: robot ZMQ streams ready")
        self.manip_wrapper = ManipulationWrapper(self.robot, stretch_gripper_max=stretch_gripper_max, end_link=end_link)
        logger.info(
            "Agent init: nav posture + look_front "
            "(head timeout warnings on a slow sim are OK; lifelong --input-path load starts after this)"
        )
        self.robot.move_to_nav_posture()
        look_front = getattr(self.robot, "look_front", None)
        if callable(look_front):
            # Prefer a short wait: MuJoCo Stretch often settles ~0.03–0.04 rad off target and
            # the default 10s look_front timeout looks like a hang before --input-path load.
            look_front(blocking=True, timeout=3.0)
            time.sleep(DYNAMEM_HEAD_SETTLE_S)
        logger.info("Agent init: robot ready")

        self.re = re
        self.save_rerun = save_rerun
        self.rerun_iter = 0
        self._cached_navigation_origin_xyt: np.ndarray | None = None


for _mod in (pose, mapping, perception, describe, look, navigation, manipulation, eqa):
    bind_module_methods(DynamemController, _mod)


RobotAgent = DynamemController
