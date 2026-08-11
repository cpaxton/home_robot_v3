# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zmq
from PIL import Image

from emet.agent.env_flags import env_agent_camera_debug, env_agent_model_debug
from emet.audio.text_to_speech import PiperTextToSpeech
from emet.config.embodied_agent_config import EmbodiedAgentConfig, legacy_embodied_agent_off
from emet.controller.base_controller import BaseController
from emet.controller.generic_zmq_client import GenericZmqClient
from emet.controller.habitat_nav import (
    NavAttemptResult,
    NavOutcome,
    goal_key_xy,
    habitat_navmesh_navigate,
    habitat_perfect_nav_enabled,
    is_habitat_robot_client,
    pick_uncovered_explore_target,
)
from emet.controller.manipulation.dynamem_manipulation.dynamem_manipulation import (
    DynamemManipulationWrapper as ManipulationWrapper,
)
from emet.controller.manipulation.dynamem_manipulation.grasper_utils import (
    capture_and_process_image,
    move_to_point,
    pickup,
    process_image_for_placing,
)
from emet.controller.zmq_client import StretchZmqClient
from emet.controller.zmq_stream_control import paused_robot_streams
from emet.core.parameters import Parameters
from emet.core.robot import AbstractRobotClient
from emet.mapping.instance import instances_to_text
from emet.mapping.scene_graph import SceneGraph
from emet.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
from emet.mapping.voxel import (
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from emet.mapping.voxel.voxel import _instance_memory_kwargs_from_params
from emet.memory.graph_eqa import GraphEQAMemory, SensorGraphBuilder
from emet.memory.graph_eqa.instance_observations import DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M
from emet.motion import constants as motion_constants
from emet.motion.algo.a_star import AStar, default_min_clearance_m
from emet.perception.depth import create_da3_estimator_from_parameters, resolve_depth_map
from emet.perception.depth.da3_estimator import apply_da3_sky_row_mask, apply_depth_speckle_filter, sensor_depth_usable
from emet.perception.depth.lingbot_estimator import LingBotDepthEstimator, create_lingbot_estimator_from_parameters
from emet.perception.detection.owl import OwlPerception
from emet.perception.detection.yoloe import YoloEPerception

# from emet.perception.encoders.mobile_clip_encoder import MaskMobileClipEncoder
from emet.perception.encoders.clip_encoder import MaskClipEncoder
from emet.perception.wrapper import OvmmPerception
from emet.utils.geometry import nav_xyt_to_world_xyt
from emet.utils.logger import Logger
from emet.utils.vram_debug import print_vram_snapshot
from emet.visualization.rerun import NullVisualizer, has_display

logger = Logger(__name__)

# Env truthy check (same tokens as ``EMET_DYNAMEM_PERFECT_DEPTH``).
_TRUEISH = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUEISH


# Manipulation hyperparameters
INIT_LIFT_POS = 0.45
INIT_WRIST_PITCH = -1.57
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.57
INIT_HEAD_TILT = -0.65

# After look_front / move_to_nav_posture, wait briefly so the head reaches goal and depth/RGB
# stabilize before base motion (Stretch ZMQ + mapping).
DYNAMEM_HEAD_SETTLE_S = 0.25
# Head sweep: command non-blocking; exit on near-goal or settled motion.
# Soft-wait is for client settle (not because real Stretch is slow — Dynamixel head ~3 rad/s).
# Sim MJCF used to use head kp=10 (crawl); assets now use higher kp. Keep a short max wait anyway.
DYNAMEM_HEAD_SWEEP_MAX_WAIT_S = 0.75
DYNAMEM_HEAD_SWEEP_MIN_MOVE_S = 0.08
DYNAMEM_HEAD_SWEEP_STOPPED_HOLD_S = 0.05
DYNAMEM_HEAD_SWEEP_SPEED_TOL = 0.20
DYNAMEM_HEAD_SWEEP_POS_DELTA_TOL = 0.04
DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD = 0.35
DYNAMEM_HEAD_SWEEP_FRAME_SETTLE_S = 0.08


def _finite_xyz_traj_target(traj_target_point: Any) -> bool:
    """True if traj tail looks like a 3D world point (not a waypoint or NaN sentinel)."""
    if isinstance(traj_target_point, torch.Tensor):
        t = traj_target_point.detach().cpu().reshape(-1)
        return t.numel() >= 3 and bool(torch.isfinite(t[:3]).all())
    if isinstance(traj_target_point, np.ndarray):
        a = np.asarray(traj_target_point, dtype=np.float64).reshape(-1)
        return a.size >= 3 and bool(np.all(np.isfinite(a[:3])))
    if isinstance(traj_target_point, (list, tuple)) and len(traj_target_point) >= 3:
        a = np.asarray(traj_target_point[:3], dtype=np.float64)
        return bool(np.all(np.isfinite(a)))
    return False


# Batched OWL text queries for describe_head_camera_scene_text (single forward pass).
_DESCRIBE_SCENE_OWL_QUERIES: tuple[str, ...] = (
    "table",
    "chair",
    "person",
    "cup",
    "bottle",
    "laptop",
    "computer monitor",
    "television",
    "cabinet",
    "shelf",
    "door",
    "window",
    "couch",
    "bed",
    "counter",
    "box",
    "bowl",
    "plate",
    "plant",
    "book",
    "keyboard",
    "microwave",
    "refrigerator",
    "sink",
    "robot arm",
)

# Household-ish labels for user-facing YoloE describe_scene (not full ScanNet-200).
# Mapping still uses ScanNet-200 at a low confidence; describe uses this vocab + a higher bar.
_DESCRIBE_SCENE_YOLOE_LABELS: tuple[str, ...] = _DESCRIBE_SCENE_OWL_QUERIES + (
    "desk",
    "sofa",
    "lamp",
    "pillow",
    "curtain",
    "picture",
    "mirror",
    "trash can",
    "bag",
    "phone",
    "remote",
    "wall",
    "floor",
    "ceiling",
    "stairs",
    "rug",
    "blanket",
    "towel",
    "toilet",
    "bathtub",
    "oven",
    "dishwasher",
    "washer",
    "dryer",
    "fan",
    "clock",
    "vase",
    "apple",
    "banana",
    "orange",
    "mouse",
    "tv stand",
    "nightstand",
    "dresser",
    "wardrobe",
    "stool",
    "bench",
    "fireplace",
    "radiator",
)

# User-facing describe_scene: mapping keeps detection.confidence_threshold low for proposals;
# chat-only filtering uses describe_confidence_threshold (safe to raise; does not affect mapping).
_DEFAULT_DESCRIBE_CONFIDENCE = 0.30
_DEFAULT_DESCRIBE_MAX_LABELS = 12
_DESCRIBE_SCENE_STRUCTURE_LABELS = frozenset({"floor", "wall", "ceiling"})


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

    def _start_threads(self) -> None:
        """DynamemController does not use realtime update threads."""
        pass

    def move_to_manip_posture(self) -> None:
        """Move the robot to manipulation posture (delegates to the robot client)."""
        self.robot.move_to_manip_posture()

    def move_to_nav_posture(self) -> None:
        """Move the robot to navigation posture (delegates to the robot client)."""
        self.robot.move_to_nav_posture()

    def _robot_emet_session(self) -> dict[str, Any] | None:
        get_sess = getattr(self.robot, "get_emet_session", None)
        if get_sess is None:
            return None
        return get_sess()

    def _navigation_origin_xyt(self) -> np.ndarray | None:
        """World spawn pose from ZMQ ``emet_session`` (cached from first observation if needed)."""
        sess = self._robot_emet_session()
        if sess is not None:
            org = sess.get("navigation_origin_xyt")
            if org is not None:
                origin = np.asarray(org, dtype=np.float64).reshape(-1)[:3]
                self._cached_navigation_origin_xyt = origin.copy()
                return origin
        if self._cached_navigation_origin_xyt is not None:
            return self._cached_navigation_origin_xyt
        return None

    def _planning_base_xyt(self, local_xyt: np.ndarray | list | tuple) -> np.ndarray:
        """Episode-relative ZMQ base pose → world frame for voxel-grid planning."""
        xyt = np.asarray(local_xyt, dtype=np.float64).reshape(-1)
        if xyt.size < 3:
            xyt = np.pad(xyt, (0, max(0, 3 - xyt.size)), mode="constant")
        sess = self._robot_emet_session()
        if sess is None and self._cached_navigation_origin_xyt is not None:
            sess = {"navigation_origin_xyt": self._cached_navigation_origin_xyt.tolist()}
        return nav_xyt_to_world_xyt(xyt[:3], sess)

    def world_base_xy(self) -> tuple[float, float] | None:
        """Robot base (x, y) in the voxel-map / world frame (not raw ZMQ gps)."""
        if self.robot is None or not hasattr(self.robot, "get_base_pose"):
            return None
        try:
            wxyt = self._planning_base_xyt(self.robot.get_base_pose())
            return float(wxyt[0]), float(wxyt[1])
        except Exception:
            return None

    def _sync_graph_frontier_nodes(self) -> None:
        gm = self.graph_memory
        if gm is None or not getattr(gm, "frontier_nodes_enabled", False):
            return
        from emet.memory.graph_eqa.dynamem_graph_hooks import sync_graph_frontier_nodes

        question = getattr(self, "_eqa_question", None)
        sync_graph_frontier_nodes(
            graph_memory=gm,
            voxel_map=self.voxel_map,
            planner=self.planner,
            base_xyt=self._planning_base_xyt(self.robot.get_base_pose()),
            question=question,
        )

    def _exploration_text(self, text: str | None) -> str | None:
        """Text used for question-guided frontier scoring (explicit query or active EQA question)."""
        if text is not None and str(text).strip():
            return str(text).strip()
        q = getattr(self, "_eqa_question", None)
        if q is not None and str(q).strip():
            return str(q).strip()
        return None

    def _localize_point_from_graph_memory(self, text: str) -> np.ndarray | None:
        """Resolve a nav goal from graph nodes (GT or perception) when voxel localize misses."""
        gm = getattr(self, "graph_memory", None)
        if gm is None or not (text or "").strip():
            return None
        from emet.memory.graph_eqa.graph_memory import heuristic_relevant_objects

        query = text.lower().strip()
        tokens = heuristic_relevant_objects(text)
        best_node = None
        best_score = -1
        for node in gm.get_nodes():
            if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                continue
            labels = [str(label).lower() for label in (node.labels or []) if str(label).strip()]
            if not labels:
                continue
            blob = " ".join(labels)
            score = 0
            if query in blob:
                score += 3
            for tok in tokens:
                if tok.lower() in blob:
                    score += 1
            if score > best_score:
                best_score = score
                best_node = node
        if best_node is None or best_score <= 0:
            return None
        return np.array([float(best_node.xyz[0]), float(best_node.xyz[1]), 1.0], dtype=float)

    def _best_frontier_point_from_graph(self, text: str | None) -> np.ndarray | None:
        """Pick the frontier graph node best matching *text* / the active EQA question."""
        gm = getattr(self, "graph_memory", None)
        if gm is None or not getattr(gm, "frontier_nodes_enabled", True):
            return None
        from emet.memory.graph_eqa.frontier_nodes import exploration_keywords_from_text, keyword_overlap_score

        frontier_nodes = [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)]
        if not frontier_nodes:
            return None
        keywords = exploration_keywords_from_text(text)
        robot = getattr(self, "robot", None)
        if robot is not None and hasattr(robot, "get_base_pose"):
            pose = robot.get_base_pose()
            rx, ry = float(pose[0]), float(pose[1])
        else:
            rx, ry = 0.0, 0.0
        if not keywords:
            node = min(
                frontier_nodes,
                key=lambda n: math.hypot(float(n.xyz[0]) - rx, float(n.xyz[1]) - ry),
            )
            return np.array([float(node.xyz[0]), float(node.xyz[1]), 1.0], dtype=float)
        best_node = None
        best_score = -1.0
        best_dist = float("inf")
        for node in frontier_nodes:
            labels = [str(lbl).strip().lower() for lbl in (node.labels or []) if str(lbl).strip()]
            score = keyword_overlap_score(labels, keywords)
            dist = math.hypot(float(node.xyz[0]) - rx, float(node.xyz[1]) - ry)
            if score > best_score or (score == best_score and dist < best_dist):
                best_score = score
                best_dist = dist
                best_node = node
        if best_node is None or best_score <= 0:
            return None
        return np.array([float(best_node.xyz[0]), float(best_node.xyz[1]), 1.0], dtype=float)

    def create_obstacle_map(self, parameters):
        """
        This function creates the MaskSiglipEncoder, Owlv2 detector, voxel map util class and voxel map navigation space util class
        """

        # Initialize the encoder in different ways depending on the configuration.
        # Dynagraph sets force_eqa_siglip_encoder so SigLIP features are computed even in
        # manipulation_only mode (Habitat EQA), enabling open-vocab text->3D grounding.
        force_siglip = bool(parameters.get("force_eqa_siglip_encoder", False))
        if self.manipulation_only and not force_siglip:
            self.encoder = None
        elif self.cpu_only:
            # Assume we only have CPU, we will use CLIP ViT-B/16 for fast inference
            self.encoder = MaskClipEncoder(version="ViT-B/16", feature_matching_threshold=0.35, device=self.device)
        else:
            # Use SIGLip-so400m for accurate inference. Shared/load-once across episodes so
            # batch runs (Habitat) do not reload the weights every controller construction.
            from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

            print("Dynamem create_obstacle_map: loading SigLIP…", flush=True)
            self.encoder = get_shared_mask_siglip_encoder(
                version="so400m", device=self.device, feature_matching_threshold=0.14
            )
            print("Dynamem create_obstacle_map: SigLIP ready", flush=True)

        # You can see a clear difference in hyperparameter selection in different querying strategies
        # Running gpt4o is time consuming, so we don't want to waste more time on object detection or Siglip or voxelization
        # On the other hand querying by feature similarity is fast and we want more fine grained details in semantic memory
        # When use_instance_memory is True we need a detector that returns instance segmentation (YoloE)
        # so that object icons show in the UI (default MuJoCo scene: red cylinder, blue cube on table).
        det_conf = parameters.get("detection", {}).get("confidence_threshold", 0.05)
        if self.manipulation_only or (self.eqa and not self._use_instance_memory):
            # No object detection for pure manipulation or voxel EQA without instances.
            # GraphEQA may set eqa=False and use_instance_memory=True so YoloE runs for instance labels.
            self.detection_model = None
            semantic_memory_resolution = 0.1
            image_shape = (360, 270)
        elif self.mllm:
            # Use GPT4o to localize objects, we use OWLV2-B for fast inference
            self.detection_model = OwlPerception(version="owlv2-B-p16", device=self.device, confidence_threshold=0.01)
            semantic_memory_resolution = 0.1
            image_shape = (360, 270)
        elif self._use_instance_memory or self.cpu_only:
            # YoloE returns instance segmentation so instance memory and UI icons work (sim + real).
            # Lower confidence (e.g. from config) helps detect small default-sim objects (red cylinder, blue cube).
            print("Dynamem create_obstacle_map: loading YoloE…", flush=True)
            from emet.perception.detection.yoloe import get_shared_yoloe_perception

            self.detection_model = get_shared_yoloe_perception(
                confidence_threshold=det_conf,
                device=self.device,
                size="l",
            )
            print("Dynamem create_obstacle_map: YoloE ready", flush=True)
            semantic_memory_resolution = 0.05
            image_shape = (360, 270)
        else:
            self.detection_model = OwlPerception(
                version="owlv2-L-p14-ensemble", device=self.device, confidence_threshold=0.15
            )
            semantic_memory_resolution = 0.05
            image_shape = (480, 360)

        _eqa = parameters.get("eqa", {}) or {}
        print("Dynamem create_obstacle_map: building SparseVoxelMap…", flush=True)
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            semantic_memory_resolution=semantic_memory_resolution,
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            grid_resolution=0.1,
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get("add_local_radius_points", True),
            remove_visited_from_obstacles=parameters.get("remove_visited_from_obstacles", False),
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
            voxel_pcd_dbscan_min_samples=int(parameters.get("filters/voxel_pcd_dbscan_min_samples", 0) or 0),
            detection=self.detection_model,
            encoder=self.encoder,
            image_shape=image_shape,
            log=self.log,
            mllm=self.mllm,
            run_eqa=self.eqa,
            device=self.device,
            eqa_backend=_eqa.get("backend", "qwen_vl"),
            eqa_vl_model_size=_eqa.get("vl_model_size", "8B"),
            eqa_vl_max_tokens=int(_eqa.get("vl_max_tokens", 512)),
            eqa_vl_quantization=_eqa.get("vl_quantization", "int4"),
            eqa_vl_hf_model_id=_eqa.get("vl_hf_model_id"),
            gemini_model=_eqa.get("gemini_model", "gemini-2.5-flash"),
            eqa_device=self.device,
            vl_family=_eqa.get("vl_family", "qwen3_vl"),
            use_instance_memory=self._use_instance_memory,
            instance_memory_kwargs=_instance_memory_kwargs_from_params(parameters),
            parameters=parameters,
            defer_eqa_vllm=self.defer_eqa_vllm,
        )
        print("Dynamem create_obstacle_map: SparseVoxelMap ready", flush=True)
        print_vram_snapshot("after_create_obstacle_map_sparse_voxel_map")
        print("Dynamem create_obstacle_map: building NavigationSpace…", flush=True)
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            rotation_step_size=parameters.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=parameters.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=parameters.get("motion_planner/frontier/dilate_obstacle_size", 0),
        )
        print("Dynamem create_obstacle_map: NavigationSpace ready", flush=True)
        _min_c = parameters.get("motion_planner/min_clearance_m", None)
        if _min_c is None:
            _fp = getattr(self.space, "_footprint", None) or getattr(self.voxel_map, "_footprint", None)
            _width = float(getattr(_fp, "width", 0.34) or 0.34)
            _min_c = default_min_clearance_m(_width)
        self._min_clearance_m = float(_min_c)
        self._clearance_cost_weight = float(parameters.get("motion_planner/clearance_cost_weight", 1.0))
        print("Dynamem create_obstacle_map: building AStar…", flush=True)
        self.planner = AStar(
            self.space,
            min_clearance_m=self._min_clearance_m,
            clearance_cost_weight=self._clearance_cost_weight,
            start_escape_max_ring=int(parameters.get("motion_planner/start_escape_max_ring", 8)),
        )
        print("Dynamem create_obstacle_map: AStar ready", flush=True)
        print("Dynamem create_obstacle_map: configuring graph memory…", flush=True)
        # Frontier / explore memory: mark goals blocked after waypoint timeout so
        # multi-goal A* skips stuck frontiers instead of re-picking them.
        self._habitat_blocked_goals: set[tuple[float, float]] = set()
        self._habitat_recent_goals: list[tuple[float, float]] = []

        cfg = self.embodied_agent
        if cfg.open_vocab_scene_graph.enabled and not self.manipulation_only:
            print("Dynamem create_obstacle_map: building open-vocabulary scene graph…", flush=True)
            from emet.mapping.scene_graph.processor import SceneGraphProcessor

            sg_name = cfg.open_vocab_scene_graph.config_name
            if self.cpu_only and sg_name == "default_scene_graph":
                sg_name = "cpu_scene_graph"
            dev = cfg.open_vocab_scene_graph.device
            if dev is None:
                dev = "cpu" if self.cpu_only else None
            self._open_vocab_sg_processor = SceneGraphProcessor(config_name=sg_name, device=dev)
            self.voxel_map.set_scene_graph_processor(self._open_vocab_sg_processor)
            print("Dynamem create_obstacle_map: open-vocabulary scene graph ready", flush=True)

        if cfg.graph_eqa_memory.enabled and not self.manipulation_only:
            print("Dynamem create_obstacle_map: building GraphEQA memory…", flush=True)
            self.graph_memory = GraphEQAMemory(
                parameters=parameters,
                log_dir=os.path.join(self.log, "graph_eqa_log"),
                defer_llm_clients=True,
            )
            gcfg = cfg.graph_eqa_memory
            self._graph_eqa_use_instance_graph = gcfg.use_instance_graph
            self._graph_eqa_use_sensor_perception = gcfg.use_sensor_perception
            if gcfg.graph_instance_dedup_xy_m is not None:
                self._graph_dedup_xy_m = float(gcfg.graph_instance_dedup_xy_m)
            elif isinstance(parameters, dict):
                self._graph_dedup_xy_m = float(
                    parameters.get("graph_instance_dedup_xy_m", DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M)
                )
            else:
                self._graph_dedup_xy_m = float(
                    parameters.get("graph_instance_dedup_xy_m", DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M)
                )
            from emet.memory.graph_eqa.graph_object_fusion.setup import attach_graph_object_fusion

            self._graph_object_fusion = attach_graph_object_fusion(
                self.graph_memory,
                parameters,
                fref=gcfg.graph_object_fusion,
            )
            self._calibration_writer = None
            dev_sg = self.device if self.device in ("cuda", "mps") else "cuda"
            self.sensor_builder = SensorGraphBuilder(
                perception_client=None,
                use_voxel_fallback=True,
                device=dev_sg,
                cpu_only=self.cpu_only,
                parameters=parameters,
            )
            print("Dynamem create_obstacle_map: GraphEQA memory ready", flush=True)
        print("Dynamem create_obstacle_map: complete", flush=True)

    def setup_custom_blueprint(self):
        """
        This function define rerun blueprint of DynaMem module.
        """
        if getattr(self.rerun_visualizer, "enabled", True) is False:
            return
        from emet.visualization.rerun import spatial3d_view_world

        main = rrb.Horizontal(
            spatial3d_view_world(),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="robot_monologue"),
                rrb.Spatial2DView(name="relevant image", origin="/observation_similar_to_text"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="world/head_camera/rgb"),
                rrb.Spatial2DView(name="ee_rgb", origin="world/ee_camera/rgb"),
                rrb.Spatial2DView(name="map_topdown", origin="world/map_snapshot/topdown"),
            ),
            rrb.Vertical(
                rrb.TextDocumentView(name="Scene Graph", origin="world/scene_graph/summary"),
            ),
            column_shares=[3, 1, 1, 1],
        )
        collapse = getattr(self.rerun_visualizer, "collapse_panels", True)
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=collapse,
        )
        rr.send_blueprint(my_blueprint)

    def _graph_dedup_skips(self, label: str, xyz: np.ndarray) -> bool:
        """Skip adding a graph node if we already have the same label near this XY (GraphEQA v1 merge)."""
        if self.graph_memory is None or self._graph_dedup_xy_m <= 0:
            return False
        lb = label.strip().lower()
        for n in self.graph_memory.get_nodes():
            if not n.labels:
                continue
            nl = (n.labels[0] or "").strip().lower()
            if nl != lb:
                continue
            if float(np.linalg.norm(n.xyz[:2] - xyz[:2])) < self._graph_dedup_xy_m:
                return True
        return False

    def _lazy_da3_estimator(self):
        if self._depth_source not in ("da3", "auto"):
            return None
        if self._da3_estimator is None:
            self._da3_estimator = create_da3_estimator_from_parameters(self.parameters, device=self.device)
        return self._da3_estimator

    def _lazy_lingbot_estimator(self) -> LingBotDepthEstimator | None:
        if self._depth_source != "lingbot":
            return None
        if self._lingbot_estimator is None:
            self._lingbot_estimator = LingBotDepthEstimator(
                create_lingbot_estimator_from_parameters(self.parameters),
                use_lingbot_pose=self._lingbot_use_pose,
            )
        return self._lingbot_estimator

    def _resolve_depth_map(
        self,
        rgb: np.ndarray,
        sensor_depth: np.ndarray | None,
        camera_K: np.ndarray | None,
        camera_pose: np.ndarray | None,
        rgb_right: np.ndarray | None = None,
        camera_K_right: np.ndarray | None = None,
        camera_pose_right: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Pick sensor depth, DA3, or auto per ``depth_source``.

        When ``debug_perfect_sensor_depth`` (YAML) or ``EMET_DYNAMEM_PERFECT_DEPTH=1`` is set, always prefer
        simulator / hardware **sensor** depth from the observation whenever it is present, so DA3 noise
        cannot mask extrinsic or frame bugs during calibration runs.

        Sets ``self._depth_map_from_da3_infer`` so :meth:`update` only applies ``da3_ignore_sky_fraction_top``
        to DA3-produced maps (never to raw sensor depth).
        """
        self._depth_map_from_da3_infer = False
        mode = str(self._depth_source).lower()
        if getattr(self, "_debug_perfect_sensor_depth", False):
            if sensor_depth is not None:
                sd = np.asarray(sensor_depth, dtype=np.float32)
                if sd.size > 0 and bool(np.any(np.isfinite(sd) & (sd > 1e-6))):
                    logger.info("debug_perfect_sensor_depth: using observation sensor depth (skipping DA3).")
                    return sd
            logger.warning(
                "debug_perfect_sensor_depth enabled but observation has no usable sensor depth; "
                "falling back to depth_source=%r.",
                mode,
            )
        if mode == "sensor":
            return sensor_depth
        if mode == "lingbot":
            est_lb = self._lazy_lingbot_estimator()
            if est_lb is None:
                raise RuntimeError("depth_source=lingbot but LingBot estimator failed to initialize.")
            self._depth_map_from_da3_infer = True
            depth_lb = est_lb.infer(
                rgb,
                camera_K=camera_K,
                camera_pose=camera_pose,
                force=(self.obs_count == 1),
            )
            if self._lingbot_use_pose and est_lb.last_camera_pose is not None:
                self._lingbot_last_pose = est_lb.last_camera_pose
            return depth_lb
        # Auto: prefer sensor without constructing DA3 (heavy); matches resolve_depth_map logic.
        if mode == "auto" and sensor_depth_usable(sensor_depth):
            return np.asarray(sensor_depth, dtype=np.float32)
        est = self._lazy_da3_estimator()
        self._depth_map_from_da3_infer = True
        return resolve_depth_map(
            self._depth_source,
            est,
            rgb,
            sensor_depth,
            camera_K,
            camera_pose,
            rgb_right,
            camera_K_right,
            camera_pose_right,
            da3_use_stereo=self._da3_use_stereo,
        )

    def _rerun_live_status_markdown(self) -> str:
        """Short markdown for the Rerun text panel: mapping progress and graph state."""
        lines: list[str] = [f"- **Observation step:** {self.obs_count}"]
        vm = getattr(self, "voxel_map", None)
        if vm is not None:
            try:
                obstacles, explored = vm.get_2d_map()
                if hasattr(obstacles, "cpu"):
                    obstacles = obstacles.cpu().numpy()
                if hasattr(explored, "cpu"):
                    explored = explored.cpu().numpy()
                obs_cells = int(obstacles.sum()) if obstacles is not None and obstacles.size else 0
                exp_cells = int(explored.sum()) if explored is not None and explored.size else 0
                lines.append(f"- **2D map:** {exp_cells} explored cells, {obs_cells} obstacle cells")
            except Exception:
                lines.append("- **2D map:** (unavailable)")
            pcd = getattr(vm, "voxel_pcd", None)
            pts = getattr(pcd, "_points", None) if pcd is not None else None
            if pts is not None:
                n = int(pts.shape[0]) if hasattr(pts, "shape") else 0
                lines.append(f"- **Voxel point cloud:** {n} points")
        gm = getattr(self, "graph_memory", None)
        if gm is not None:
            try:
                lines.append(f"- **Graph memory nodes:** {len(gm.get_nodes())}")
            except Exception:
                lines.append("- **Graph memory:** (unavailable)")
        return "\n".join(lines)

    def _rerun_refresh_monologue_panel(self) -> None:
        """Push ``robot_monologue`` = stored plan/EQA text plus live mapping status."""
        if not getattr(self.rerun_visualizer, "enabled", True):
            return
        base = (self._rerun_monologue_base or "").strip()
        if not base:
            base = "*No navigation plan or EQA answer in this session step yet — building the map from depth.*"
        live = self._rerun_live_status_markdown()
        doc = f"{base}\n\n---\n\n## Live status\n{live}"
        self.rerun_visualizer.log_text("robot_monologue", doc)

    def update(self):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""

        obs = self.robot.get_observation()
        if obs is None:
            logger.warning("get_observation() returned None; skipping voxel update")
            self.robot.set_mapping_depth_for_rerun(None)
            return
        self.obs_count += 1
        rgb, sensor_depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
        run_infer_full = self._da3_infer_every_n <= 1 or (self.obs_count - 1) % self._da3_infer_every_n == 0
        if self._depth_source == "lingbot":
            run_infer_full = self._lingbot_infer_every_n <= 1 or (self.obs_count - 1) % self._lingbot_infer_every_n == 0
        depth: np.ndarray | None
        if (
            not run_infer_full
            and self._depth_source in ("da3", "auto")
            and not getattr(self, "_debug_perfect_sensor_depth", False)
            and self._da3_last_depth is not None
            and self._da3_last_depth.shape[:2] == rgb.shape[:2]
        ):
            depth = np.asarray(self._da3_last_depth, dtype=np.float32).copy()
            self._depth_map_from_da3_infer = True
            if self._depth_source == "auto" and sensor_depth is not None and np.asarray(sensor_depth).size > 0:
                depth = np.asarray(sensor_depth, dtype=np.float32)
                self._depth_map_from_da3_infer = False
        elif (
            not run_infer_full
            and self._depth_source == "lingbot"
            and self._lingbot_last_depth is not None
            and self._lingbot_last_depth.shape[:2] == rgb.shape[:2]
        ):
            depth = np.asarray(self._lingbot_last_depth, dtype=np.float32).copy()
            self._depth_map_from_da3_infer = True
        else:
            depth = self._resolve_depth_map(
                rgb,
                sensor_depth,
                K,
                camera_pose,
                rgb_right=getattr(obs, "head_rgb_right", None),
                camera_K_right=getattr(obs, "head_camera_K_right", None),
                camera_pose_right=getattr(obs, "head_camera_pose_right", None),
            )
            if depth is not None and getattr(self, "_depth_map_from_da3_infer", False):
                if self._depth_source == "lingbot":
                    self._lingbot_last_depth = np.asarray(depth, dtype=np.float32).copy()
                else:
                    self._da3_last_depth = np.asarray(depth, dtype=np.float32).copy()
        if self._depth_source == "lingbot" and getattr(self, "_lingbot_last_pose", None) is not None:
            if self._lingbot_use_pose:
                camera_pose = self._lingbot_last_pose
        if depth is None:
            logger.error(f"No depth map available (depth_source={self._depth_source!r}); skipping voxel update.")
            self.robot.set_mapping_depth_for_rerun(None)
            return
        if getattr(self, "_depth_map_from_da3_infer", False):
            depth = np.asarray(depth, dtype=np.float32)
            sky = float(self.parameters.get("da3_ignore_sky_fraction_top", 0.0) or 0.0)
            if sky > 0.0:
                depth = apply_da3_sky_row_mask(depth, sky)
            speckle_k = int(self.parameters.get("filters/depth_speckle_open_kernel", 0) or 0)
            if speckle_k > 0:
                depth = apply_depth_speckle_filter(
                    depth,
                    open_kernel=speckle_k,
                    open_iterations=int(self.parameters.get("filters/depth_speckle_open_iterations", 1) or 1),
                    min_depth=float(self.parameters.get("min_depth", 0.25)),
                    max_depth=float(self.parameters.get("max_depth", 2.5)),
                )
        self.robot.set_mapping_depth_for_rerun(depth)
        base_xyt = None
        if obs.gps is not None and obs.compass is not None:
            g = np.asarray(obs.gps, dtype=np.float64).reshape(-1)
            c = np.asarray(obs.compass, dtype=np.float64).ravel()
            if g.size >= 2 and c.size >= 1:
                local_xyt = np.array([float(g[0]), float(g[1]), float(c[0])], dtype=np.float64)
                base_xyt = nav_xyt_to_world_xyt(local_xyt, getattr(obs, "emet_session", None))
        if _env_truthy("EMET_DYNAMEM_MAP_DEBUG"):
            sess = getattr(obs, "emet_session", None) or {}
            org = sess.get("navigation_origin_xyt")
            cam_t = None
            if camera_pose is not None:
                cp = np.asarray(camera_pose, dtype=np.float64)
                if cp.shape == (4, 4):
                    cam_t = np.round(cp[:3, 3], 4).tolist()
            logger.info(
                "dynamem_map_debug step=%s depth_source=%s da3_infer=%s perfect_depth=%s "
                "nav_origin_xyt=%s base_xyt=%s camera_t=%s",
                self.obs_count,
                self._depth_source,
                bool(getattr(self, "_depth_map_from_da3_infer", False)),
                bool(getattr(self, "_debug_perfect_sensor_depth", False)),
                None if org is None else np.asarray(org, dtype=np.float64).round(4).tolist(),
                None if base_xyt is None else np.asarray(base_xyt, dtype=np.float64).round(4).tolist(),
                cam_t,
            )
        if getattr(obs, "emet_session", None) is not None:
            org = getattr(obs, "emet_session", {}).get("navigation_origin_xyt")
            if org is not None:
                self._cached_navigation_origin_xyt = np.asarray(org, dtype=np.float64).reshape(-1)[:3].copy()

        self.voxel_map.process_rgbd_images(rgb, depth, K, camera_pose, base_xyt=base_xyt)
        robot_xy = None
        if obs.gps is not None and obs.compass is not None:
            g = np.asarray(obs.gps, dtype=np.float64).reshape(-1)
            cc = np.asarray(obs.compass, dtype=np.float64).ravel()
            if g.size >= 2 and cc.size >= 1:
                wxyt = nav_xyt_to_world_xyt(
                    np.array([float(g[0]), float(g[1]), float(cc[0])], dtype=np.float64),
                    getattr(obs, "emet_session", None),
                )
                robot_xy = (float(wxyt[0]), float(wxyt[1]))
        if getattr(self.rerun_visualizer, "enabled", True):
            self.rerun_visualizer.log_topdown_map_snapshot(self.voxel_map, robot_base_xy=robot_xy)
        if self.voxel_map.voxel_pcd._points is not None:
            self.rerun_visualizer.update_voxel_map(space=self.space, robot_base_xy=robot_xy)
        if self.voxel_map.semantic_memory._points is not None:
            self.rerun_visualizer.log_custom_pointcloud(
                "world/semantic_memory/pointcloud",
                self.voxel_map.semantic_memory._points.detach().cpu(),
                self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
                0.03,
            )
        if self.use_scene_graph and self.voxel_map.use_instance_memory and self.graph_memory is None:
            instances = self.get_voxel_map().get_instances()
            if instances:
                self._update_scene_graph()
                self.rerun_visualizer.update_scene_graph(
                    self.scene_graph,
                    self.semantic_sensor,
                    detection_model=getattr(self, "detection_model", None),
                    graph_memory=self.graph_memory,
                )

        has_hm3d_labeler = getattr(self.robot, "hm3d_semantic_labeler", None) is not None
        if self.graph_memory is not None and (
            self.sensor_builder is not None or self._graph_eqa_use_instance_graph or has_hm3d_labeler
        ):
            if getattr(self, "_skip_graph_perception_updates", False):
                from emet.memory.graph_eqa.dynamem_graph_hooks import (
                    update_graph_memory_ground_truth_from_observation,
                )

                update_graph_memory_ground_truth_from_observation(
                    graph_memory=self.graph_memory,
                    robot=self.robot,
                    obs=obs,
                    frame_step=self.obs_count,
                )
            else:
                from emet.memory.graph_eqa.dynamem_graph_hooks import update_graph_memory_from_dynamem_observation

                update_graph_memory_from_dynamem_observation(
                    graph_memory=self.graph_memory,
                    robot=self.robot,
                    voxel_map=self.voxel_map,
                    detection_model=self.detection_model,
                    sensor_builder=self.sensor_builder,
                    use_instance_graph=self._graph_eqa_use_instance_graph,
                    use_sensor_perception=self._graph_eqa_use_sensor_perception,
                    dedup_skips=self._graph_dedup_skips,
                    obs=obs,
                    frame_step=self.obs_count,
                    graph_object_fusion=getattr(self, "_graph_object_fusion", None),
                    calibration_writer=getattr(self, "_calibration_writer", None),
                )

        if self.graph_memory is not None:
            self._sync_graph_frontier_nodes()

        # Visualize open-vocab scene graph if attached (dynagraph uses graph_memory instead).
        ovsg = self.voxel_map.get_scene_graph()
        if ovsg is not None and ovsg.num_objects > 0 and self.graph_memory is None:
            self.rerun_visualizer.update_open_vocab_scene_graph(ovsg)

        self._rerun_refresh_monologue_panel()
        self._run_on_step_callbacks()

    def _run_on_step_callbacks(self) -> None:
        for cb in getattr(self, "_on_step_callbacks", ()) or ():
            try:
                cb(self)
            except Exception as exc:
                logger.warning(f"on_step callback failed: {exc}")

    def _update_scene_graph(self) -> None:
        """Update the scene graph with the latest instances from the voxel map."""
        if self.scene_graph is None:
            self.scene_graph = SceneGraph(self.parameters, self.get_voxel_map().get_instances())
        else:
            self.scene_graph.update(self.get_voxel_map().get_instances())
        self.scene_graph.get_relationships(debug=False)

    def dump_memory_to_text(
        self,
        include_bounds: bool = True,
        class_names: dict[int, str] | None = None,
    ) -> str:
        """Return instance memory as human-readable text (for logging or CLI dump)."""
        if not self.voxel_map.use_instance_memory:
            return "Instance memory is disabled."
        instances = self.get_voxel_map().get_instances()
        if class_names is None and self.semantic_sensor is not None and self.semantic_sensor.is_semantic():
            class_names = {}
            for inst in instances:
                cid = inst.get_category_id()
                if cid is not None and cid not in class_names:
                    name = self.semantic_sensor.get_class_name_for_id(cid)
                    if name is not None:
                        class_names[cid] = name
        return instances_to_text(instances, class_names=class_names, include_bounds=include_bounds)

    def describe_head_camera_scene_text(
        self,
        *,
        graph_memory: Any | None = None,
        memory_backend: Any | None = None,
        graph_memory_backend: Any | None = None,
    ) -> str:
        """User-facing answer for ``describe_scene`` (current view — not a motion skill).

        Priority for "what can you see":
        1. Caption the **current** head RGB (VLM when loaded).
        2. Ground / enrich with graph/map labels already known.
        3. Optional curated detector fallback if configured.
        Does **not** look around or explore — use ``scan_environment`` / ``explore`` for that.
        """
        mem_kw = {
            "graph_memory": graph_memory,
            "memory_backend": memory_backend,
            "graph_memory_backend": graph_memory_backend,
        }
        rgb, depth = self._describe_scene_capture_rgb()
        if isinstance(rgb, str):
            return rgb  # error string from capture

        det_cfg = self._detection_cfg()
        if env_agent_model_debug():
            print(
                "[model debug] describe_scene: caption current view + ground with graph/memory; "
                "no auto look-around "
                f"(detector_fallback={bool(det_cfg.get('describe_use_detector_fallback', False))})",
                flush=True,
            )

        text = self._describe_scene_try_sources(rgb, depth, det_cfg, **mem_kw)
        if text:
            return text

        return (
            "I don't have a captioner or mapped object labels for this view yet. "
            "I'm sending a photo of what is in front of me — ask me to look around or explore "
            "if you want me to map more of the room."
        )

    def _describe_scene_capture_rgb(self) -> tuple[np.ndarray, np.ndarray | None] | tuple[str, None]:
        """Return (rgb, depth) or (error_message, None)."""
        if self.robot is None or not hasattr(self.robot, "get_observation"):
            return "No robot view available.", None
        obs = self.robot.get_observation()
        if obs is None or getattr(obs, "rgb", None) is None:
            return "No current image.", None
        rgb = np.asarray(obs.rgb)
        if rgb.dtype != np.uint8:
            if rgb.size and float(np.nanmax(rgb)) <= 1.0 + 1e-6:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return "Head camera image has an unexpected shape.", None

        from emet.llms.vl_image import downsample_rgb_hwc, eqa_vl_image_kwargs

        eqa_cfg: dict[str, Any] = {}
        if isinstance(self.parameters, dict):
            eqa_cfg = self.parameters.get("eqa", {}) or {}
        elif self.parameters is not None and hasattr(self.parameters, "get"):
            raw = self.parameters.get("eqa", {}) or {}
            eqa_cfg = raw if isinstance(raw, dict) else {}
        if not isinstance(eqa_cfg, dict):
            eqa_cfg = {}
        img_kw = eqa_vl_image_kwargs(eqa_cfg)
        rgb = downsample_rgb_hwc(rgb, max_side=img_kw["image_max_side"], max_pixels=img_kw["image_max_pixels"])

        if env_agent_camera_debug():
            from emet.agent.camera_debug import print_camera_frame_diagnostics

            print_camera_frame_diagnostics("describe_scene (head RGB)", rgb, force=True)

        depth = getattr(obs, "depth", None)
        if depth is not None:
            depth = np.asarray(depth)
        return rgb, depth

    def _describe_scene_try_sources(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        det_cfg: dict[str, Any],
        *,
        graph_memory: Any | None = None,
        memory_backend: Any | None = None,
        graph_memory_backend: Any | None = None,
    ) -> str | None:
        """Caption the current view first; ground with graph/map; optional detector fallback."""
        # Live caption is the primary answer for "what can you see".
        vlm_text = self._describe_scene_vlm(rgb)
        mem_text = self._describe_scene_from_memory(
            graph_memory=graph_memory,
            memory_backend=memory_backend,
            graph_memory_backend=graph_memory_backend,
        )
        parts = [p for p in (vlm_text, mem_text) if p]
        if parts:
            return " ".join(parts)

        if bool(det_cfg.get("describe_use_detector_fallback", False)):
            dm = self.detection_model
            try:
                if isinstance(dm, YoloEPerception):
                    return self._describe_scene_yoloe(rgb, depth, dm)
                if isinstance(dm, OwlPerception):
                    thr = float(dm.confidence_threshold) if dm.confidence_threshold is not None else 0.2
                    thr = max(0.12, min(thr, 0.35))
                    return self._describe_scene_owl(rgb, dm, thr)
            except Exception as e:
                if env_agent_model_debug():
                    print(f"[model debug] describe_scene detector fallback failed: {e}", flush=True)
                return None
        return None

    def _detection_cfg(self) -> dict[str, Any]:
        if isinstance(self.parameters, dict):
            raw = self.parameters.get("detection", {}) or {}
        elif self.parameters is not None and hasattr(self.parameters, "get"):
            raw = self.parameters.get("detection", {}) or {}
        else:
            raw = {}
        return raw if isinstance(raw, dict) else {}

    def _describe_scene_vlm(self, rgb: np.ndarray) -> str | None:
        """Caption current RGB with DynaMem image_description / EQA VLM when present."""
        vm = self.get_voxel_map() if hasattr(self, "get_voxel_map") else None
        if vm is None:
            return None
        client = getattr(vm, "image_description_client", None) or getattr(vm, "eqa_client", None)
        if client is None:
            return None
        from emet.llms.vllm_factory import dynamem_vllm_call

        pil = Image.fromarray(rgb)
        prompt = (
            "Describe what is visible in this robot head-camera image in one short sentence. "
            "Name only clearly visible objects and surfaces. If the view is mostly empty floor/wall "
            "or the robot's own body, say that. Do not invent objects that are not clearly visible."
        )
        try:
            with paused_robot_streams(self.robot):
                out = dynamem_vllm_call(
                    client,
                    [pil, prompt],
                    system_prompt="",
                    max_new_tokens=64,
                )
        except Exception as e:
            if env_agent_model_debug():
                print(f"[model debug] describe_scene VLM caption failed: {e}", flush=True)
            return None
        text = (out or "").strip()
        if not text:
            return None
        if env_agent_model_debug():
            print(f"[model debug] describe_scene: VLM caption ({type(client).__name__})", flush=True)
        return f"From my head camera: {text}"

    def _describe_scene_from_memory(
        self,
        *,
        graph_memory: Any | None = None,
        memory_backend: Any | None = None,
        graph_memory_backend: Any | None = None,
    ) -> str | None:
        """Summarize known object labels from graph / memory backends (not live detector)."""
        labels: list[str] = []
        for backend in (graph_memory_backend, memory_backend):
            if backend is not None and hasattr(backend, "list_objects"):
                try:
                    labels = [str(x) for x in (backend.list_objects() or []) if str(x).strip()]
                except Exception:
                    labels = []
                if labels:
                    break
        if not labels and graph_memory is not None and hasattr(graph_memory, "get_nodes"):
            for n in graph_memory.get_nodes():
                if getattr(n, "is_viewpoint", False) or getattr(n, "is_frontier", False):
                    continue
                for lab in getattr(n, "labels", None) or []:
                    s = str(lab).strip()
                    if s:
                        labels.append(s)
            labels = list(dict.fromkeys(labels))
        if not labels and hasattr(self, "get_voxel_map"):
            vm = self.get_voxel_map()
            get_inst = getattr(vm, "get_instances", None) if vm is not None else None
            if callable(get_inst):
                try:
                    for inst in get_inst() or []:
                        # Prefer string category if present on instance
                        cat = getattr(inst, "category_name", None) or getattr(inst, "name", None)
                        if cat:
                            labels.append(str(cat))
                except Exception:
                    pass
                labels = list(dict.fromkeys(labels))
        if not labels:
            return None
        shown = labels[:20]
        extra = f" (+{len(labels) - len(shown)} more)" if len(labels) > len(shown) else ""
        if env_agent_model_debug():
            print(
                f"[model debug] describe_scene: memory/graph labels n={len(labels)}",
                flush=True,
            )
        return (
            f"From my map/scene graph ({len(labels)} object labels) I also know about: "
            + ", ".join(shown)
            + extra
            + "."
        )

    @staticmethod
    def _normalize_scene_rgb_u8(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr)
        if out.dtype != np.uint8:
            if out.size and float(np.nanmax(out)) <= 1.0 + 1e-6:
                out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def pick_interesting_scene_image(
        self,
        *,
        graph_memory: Any | None = None,
        live_rgb: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, str | None]:
        """Prefer a usable graph-object crop over live head RGB for Discord / chat photos.

        Returns ``(image_hwc_uint8, label_or_None)``. Label is set only for a real named
        object crop that passes the RGB usability gate; blank/white crops fall back to live RGB.
        """
        from emet.agent.camera_debug import rgb_frame_is_usable

        def _live() -> tuple[np.ndarray | None, str | None]:
            if live_rgb is None:
                return None, None
            arr = np.asarray(live_rgb)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return None, None
            out = self._normalize_scene_rgb_u8(arr.copy())
            return out, None

        gm = graph_memory if graph_memory is not None else getattr(self, "graph_memory", None)
        if gm is not None and hasattr(gm, "get_nodes") and hasattr(gm, "get_observations"):
            from emet.visualization.rerun import dynagraph_node_rgb_crop, node_has_detection_crop

            obs_rgb = {int(o.obs_id): np.asarray(o.rgb) for o in gm.get_observations()}
            # (named_bonus, support, area, label, arr)
            candidates: list[tuple[int, int, int, str, np.ndarray]] = []
            for n in gm.get_nodes():
                if getattr(n, "is_viewpoint", False) or getattr(n, "is_frontier", False):
                    continue
                if not node_has_detection_crop(n, obs_rgb):
                    continue
                crop = dynagraph_node_rgb_crop(n, obs_rgb)
                if crop is None or getattr(crop, "size", 0) == 0:
                    continue
                arr = self._normalize_scene_rgb_u8(np.asarray(crop))
                if arr.ndim != 3 or arr.shape[2] != 3:
                    continue
                if not rgb_frame_is_usable(arr):
                    continue
                raw_labels = [str(x).strip() for x in (getattr(n, "labels", None) or []) if str(x).strip()]
                label = raw_labels[0] if raw_labels else ""
                if not label or label.lower() in ("object", "unknown", "none"):
                    continue  # generic / empty — do not claim "closer look at object"
                score = int(getattr(n, "support_count", 1) or 1)
                area = int(arr.shape[0]) * int(arr.shape[1])
                candidates.append((1, score, area, label, arr))
            if candidates:
                candidates.sort(key=lambda t: (-t[0], -t[1], -t[2]))
                _nb, _score, _area, label, arr = candidates[0]
                return arr.copy(), label

        if hasattr(self, "get_voxel_map"):
            try:
                vm = self.get_voxel_map()
            except Exception:
                vm = None
            get_inst = getattr(vm, "get_instances", None) if vm is not None else None
            if callable(get_inst):
                best: tuple[int, str, np.ndarray] | None = None
                try:
                    for inst in get_inst() or []:
                        view = getattr(inst, "get_best_view", lambda: None)()
                        crop_t = getattr(view, "cropped_image", None) if view is not None else None
                        if crop_t is None:
                            continue
                        from emet.mapping.instance.instance import _cropped_image_to_caption_input

                        arr = _cropped_image_to_caption_input(crop_t)
                        if arr is None or arr.size == 0:
                            continue
                        arr = self._normalize_scene_rgb_u8(np.asarray(arr))
                        if not rgb_frame_is_usable(arr):
                            continue
                        cat = getattr(inst, "category_name", None) or getattr(inst, "name", None) or ""
                        cat_s = str(cat).strip()
                        if not cat_s or cat_s.lower() in ("object", "unknown", "none"):
                            continue
                        area = int(arr.shape[0]) * int(arr.shape[1])
                        if best is None or area > best[0]:
                            best = (area, cat_s, arr)
                except Exception:
                    best = None
                if best is not None:
                    return best[2].copy(), best[1]

        return _live()

    def _describe_scene_yoloe(self, rgb: np.ndarray, depth: np.ndarray | None, dm: YoloEPerception) -> str:
        """User-facing caption from YoloE — not the low-conf ScanNet dump used for mapping."""
        det_cfg: dict[str, Any] = {}
        if isinstance(self.parameters, dict):
            det_cfg = self.parameters.get("detection", {}) or {}
        elif self.parameters is not None and hasattr(self.parameters, "get"):
            raw = self.parameters.get("detection", {}) or {}
            det_cfg = raw if isinstance(raw, dict) else {}

        thr = float(det_cfg.get("describe_confidence_threshold", _DEFAULT_DESCRIBE_CONFIDENCE))
        thr = max(0.15, min(thr, 0.85))
        max_labels = int(
            det_cfg.get("describe_max_labels", _DEFAULT_DESCRIBE_MAX_LABELS) or _DEFAULT_DESCRIBE_MAX_LABELS
        )
        max_labels = max(1, min(max_labels, 30))
        use_curated = bool(det_cfg.get("describe_use_curated_vocab", True))

        old_vocab = None
        label_vocab: list[str]
        if use_curated:
            old_vocab = list(dm.class_list)
            label_vocab = list(_DESCRIBE_SCENE_YOLOE_LABELS)
            dm.class_list = label_vocab
        else:
            label_vocab = list(dm.class_list)
        try:
            # Pause ZMQ decode during GPU detect (same contention as chat LLM load).
            with paused_robot_streams(self.robot):
                _sem, _inst, task = dm.predict(
                    rgb,
                    depth=depth,
                    draw_instance_predictions=False,
                    confidence_threshold=thr,
                )
        finally:
            if old_vocab is not None:
                dm.class_list = old_vocab

        ic = task.get("instance_classes")
        scores = task.get("instance_scores")
        if ic is None or len(ic) == 0:
            return (
                "This view looks empty or unclear to me. "
                "Ask me to look around for a wider scan, or I can send a photo of what I see."
            )

        best: dict[str, float] = {}
        idxs = np.atleast_1d(np.asarray(ic)).astype(int).ravel()
        scs = (
            np.atleast_1d(np.asarray(scores, dtype=np.float64)).ravel()
            if scores is not None
            else np.ones(len(idxs), dtype=np.float64)
        )
        if len(scs) != len(idxs):
            scs = np.ones(len(idxs), dtype=np.float64)
        for idx, sc in zip(idxs, scs, strict=True):
            if float(sc) < thr:
                continue
            i = int(idx)
            if 0 <= i < len(label_vocab):
                name = label_vocab[i]
                prev = best.get(name)
                if prev is None or float(sc) > prev:
                    best[name] = float(sc)
        if not best:
            return (
                "This view looks empty or unclear to me. "
                "Ask me to look around for a wider scan, or I can send a photo of what I see."
            )
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:max_labels]
        names = [name for name, _sc in ranked]
        if names and set(names).issubset(_DESCRIBE_SCENE_STRUCTURE_LABELS):
            return (
                "Mostly empty from here — mainly "
                + ", ".join(names)
                + ". Ask me to look around if you want a wider view."
            )
        summary = ", ".join(names)
        return f"From my head camera I can make out: {summary}."

    def _describe_scene_owl(self, rgb: np.ndarray, dm: OwlPerception, confidence_threshold: float) -> str:
        texts = list(_DESCRIBE_SCENE_OWL_QUERIES)
        res = dm.predict(rgb, texts, confidence_threshold=confidence_threshold)
        labels = res["labels"]
        if labels.numel() == 0:
            return "This view looks empty or unclear to me. Ask me to look around, or I can send a photo."
        scores = res["scores"]
        best_by_label: dict[int, float] = {}
        for lab, sc in zip(labels.cpu().tolist(), scores.cpu().tolist(), strict=True):
            li = int(lab)
            sf = float(sc)
            if li not in best_by_label or sf > best_by_label[li]:
                best_by_label[li] = sf
        picked = [texts[i] for i in sorted(best_by_label)]
        summary = ", ".join(picked)
        return f"From my head camera, open-vocabulary detection suggests: {summary}."

    def _head_to_sweep(self, pan: float, tilt: float) -> None:
        """Move head for a look-around pan; return once close enough or briefly settled.

        Real Stretch head Dynamixels are fast; soft-wait is only to avoid blocking on joint
        tolerance. Sim MJCF head gains were raised (was kp=10 crawl) so pans should be snappy.
        """
        head_to = getattr(self.robot, "head_to", None)
        if not callable(head_to):
            return
        # Non-blocking; reliable=False avoids extra resends while we soft-wait.
        head_to(float(pan), float(tilt), blocking=False, reliable=False)
        get_js = getattr(self.robot, "get_joint_state", None)
        if not callable(get_js):
            time.sleep(DYNAMEM_HEAD_SWEEP_MAX_WAIT_S * 0.5)
            return
        try:
            from emet.motion.kinematics import HelloStretchIdx
        except Exception:
            time.sleep(DYNAMEM_HEAD_SWEEP_MAX_WAIT_S * 0.5)
            return

        t0 = time.time()
        stopped_since: float | None = None
        last_pan: float | None = None
        last_tilt: float | None = None
        while time.time() - t0 < DYNAMEM_HEAD_SWEEP_MAX_WAIT_S:
            try:
                joints, vels, _ = get_js()
            except Exception:
                joints, vels = None, None
            now = time.time()
            elapsed = now - t0
            if joints is None or len(joints) <= HelloStretchIdx.HEAD_TILT:
                time.sleep(0.04)
                continue

            cur_pan = float(joints[HelloStretchIdx.HEAD_PAN])
            cur_tilt = float(joints[HelloStretchIdx.HEAD_TILT])
            pan_err = abs(cur_pan - float(pan))
            tilt_err = abs(cur_tilt - float(tilt))
            near_goal = pan_err < DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD and tilt_err < DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD
            # Good enough for a sweep frame — do not wait out residual crawl.
            if near_goal and elapsed >= DYNAMEM_HEAD_SWEEP_MIN_MOVE_S * 0.5:
                break

            speed = 0.0
            if vels is not None and len(vels) > HelloStretchIdx.HEAD_TILT:
                speed = abs(float(vels[HelloStretchIdx.HEAD_PAN])) + abs(float(vels[HelloStretchIdx.HEAD_TILT]))
            pos_delta = 0.0
            if last_pan is not None and last_tilt is not None:
                pos_delta = abs(cur_pan - last_pan) + abs(cur_tilt - last_tilt)
            last_pan, last_tilt = cur_pan, cur_tilt

            # Loose: slow creep counts as stopped so we do not burn max wait every pan.
            moving = speed > DYNAMEM_HEAD_SWEEP_SPEED_TOL or pos_delta > DYNAMEM_HEAD_SWEEP_POS_DELTA_TOL
            if not moving:
                if stopped_since is None:
                    stopped_since = now
                if (now - stopped_since) >= DYNAMEM_HEAD_SWEEP_STOPPED_HOLD_S and (
                    elapsed >= DYNAMEM_HEAD_SWEEP_MIN_MOVE_S
                ):
                    break
            else:
                stopped_since = None
            time.sleep(0.04)

    def look_around(self):
        """
        Let the robot look around to check its surroudings.
        Rotating the robot head to compensate for the narrow field of view of realsense head camera
        """
        self.announce_action("Look around: sweeping head")
        tilt = float(motion_constants.look_front[1])
        # Four pans for Realsense FOV coverage (left → right-ish). Soft-wait exits on settle.
        # Explore-loop / smoke: two extremes ~halves wall time (~100s → ~50s per excursion).
        if getattr(self, "_fast_explore_lookaround", False):
            pans = [0.6, -1.8]
        else:
            pans = [0.6, -0.2, -1.0, -1.8]
        n = len(pans)
        t_sweep = time.time()
        for i, pan in enumerate(pans):
            self.announce_motion_progress(f"Look around: head pan {i + 1}/{n} (pan={pan:+.1f} rad, tilt={tilt:+.2f})")
            self._head_to_sweep(pan, tilt)
            time.sleep(DYNAMEM_HEAD_SWEEP_FRAME_SETTLE_S)
            self.update()
        self.announce_motion_progress(f"Look around: head sweep done ({time.time() - t_sweep:.1f}s)")
        # Return to look_front without a long blocking wait.
        self._head_to_sweep(float(motion_constants.look_front[0]), tilt)
        time.sleep(DYNAMEM_HEAD_SETTLE_S)

    def _find_phase_nav_timeout(self, default: float = 10.0) -> float:
        raw = self.parameters.get("find_phase_nav_step_timeout_s")
        if raw is None:
            return default
        return float(raw)

    def rotate_in_place(self):
        self.announce_action("Looking around: rotating in place")
        nav_timeout = self._find_phase_nav_timeout()
        if self.save_rerun:
            if not os.path.exists(self.log):
                os.makedirs(self.log)
            rr.save(self.log + "/" + "data_" + str(self.rerun_iter) + ".rrd")
        self.robot.move_to_nav_posture()
        self.announce_motion_progress("Looking around: nav posture + look_front")
        self.robot.look_front(blocking=True, timeout=nav_timeout)
        time.sleep(DYNAMEM_HEAD_SETTLE_S)
        wait_obs = getattr(self.robot, "wait_for_obs", None)
        if callable(wait_obs):
            wait_obs(timeout=nav_timeout)
        n_steps = 8
        logger.info("rotate_in_place: %d× relative +45° yaw (no XY translation)", n_steps)
        for step_i in range(n_steps):
            self.announce_motion_progress(f"Looking around: scan step {step_i + 1}/{n_steps}")
            self.robot.move_base_to(
                [0.0, 0.0, np.pi / 4.0],
                relative=True,
                blocking=True,
                timeout=nav_timeout,
            )
            if not self._realtime_updates:
                self.update()
            # Discord: mid + done only (avoid 8 spam messages); terminal already has every step.
            if step_i in (3, 7):
                self.announce_action(f"Looking around: scan step {step_i + 1}/{n_steps}")
        self.announce_motion_progress("Looking around: rotate-in-place done")
        self.rerun_iter += 1
        self._maybe_emit_navgrid_ascii(context="rotate_in_place")

    def rotate_base_degrees(self, degrees: float) -> float:
        """Relative in-place yaw (degrees). Positive = left/CCW. Returns commanded degrees."""
        deg = float(np.clip(float(degrees), -360.0, 360.0))
        if abs(deg) < 1e-3:
            return 0.0
        self.announce_action(f"Rotating {deg:+.0f}°")
        # Scale wait with angle (180° Spin ~5s); floor above find-phase default so large yaws finish.
        nav_timeout = max(float(self._find_phase_nav_timeout()), abs(deg) / 45.0 * 5.0 + 8.0)
        if hasattr(self.robot, "move_to_nav_posture"):
            self.robot.move_to_nav_posture()
        self.robot.move_base_to(
            [0.0, 0.0, float(np.deg2rad(deg))],
            relative=True,
            blocking=True,
            timeout=nav_timeout,
        )
        if not getattr(self, "_realtime_updates", False):
            try:
                self.update()
            except Exception:
                pass
        return deg

    def _seed_local_radius_explored(self, vm) -> bool:
        """Stamp ``local_radius`` explored disk at the current base (Stretch-style turn-around hack).

        Returns True if the map reports any explored cells afterward.
        """
        if vm is None or not hasattr(vm, "_update_visited"):
            return False
        try:
            xyt = np.asarray(self.robot.get_base_pose(), dtype=np.float64).reshape(-1)
        except Exception:
            return False
        if xyt.size < 2:
            return False
        try:
            import torch

            pose = torch.as_tensor(xyt[:3], dtype=torch.float32)
            device = getattr(vm, "map_2d_device", None)
            if device is not None:
                pose = pose.to(device)
            vm._update_visited(pose)
            # Invalidate 2D cache so the next get_2d_map includes _visited.
            if hasattr(vm, "_map2d"):
                vm._map2d = None
        except Exception:
            return False
        try:
            obstacles, explored = vm.get_2d_map()
        except Exception:
            return False
        if explored is None:
            return False
        exp_np = explored.cpu().numpy() if hasattr(explored, "cpu") else np.asarray(explored)
        return int(np.count_nonzero(exp_np)) > 0

    def clip_forward_distance_m(
        self,
        meters: float,
        *,
        step_m: float = 0.05,
        clearance_m: float = 0.05,
        require_map: bool = True,
    ) -> float:
        """Shorten a forward request using the 2D obstacle map.

        Always consults the voxel map before driving — including small nudges (0.1 m).
        When *require_map* is True (default), paths must stay on explored cells. If the map
        has no explored cells yet, stamps the configured ``local_radius`` disk at the base
        (same Stretch-style turn-around seed) and retries — never drives into unknown space
        beyond that disk. Stops *clearance_m* before the first occupied cell.
        """
        requested = float(np.clip(float(meters), 0.0, 1.5))
        if requested < 1e-3:
            return 0.0
        vm = self.get_voxel_map() if hasattr(self, "get_voxel_map") else None
        if vm is None:
            return 0.0 if require_map else requested

        def _load_maps():
            try:
                obstacles, explored = vm.get_2d_map()
            except Exception:
                return None, None
            if obstacles is None:
                return None, None
            obs_np = obstacles.cpu().numpy() if hasattr(obstacles, "cpu") else np.asarray(obstacles)
            exp_np = None
            if explored is not None:
                exp_np = explored.cpu().numpy() if hasattr(explored, "cpu") else np.asarray(explored)
            return obs_np, exp_np

        obs_np, exp_np = _load_maps()
        empty_cloud = bool(hasattr(vm, "is_empty") and vm.is_empty())
        n_obs = int(np.count_nonzero(obs_np)) if obs_np is not None else 0
        n_exp = int(np.count_nonzero(exp_np)) if exp_np is not None else 0
        if require_map and (obs_np is None or (empty_cloud and n_exp == 0) or (n_obs == 0 and n_exp == 0)):
            if self._seed_local_radius_explored(vm):
                obs_np, exp_np = _load_maps()
                n_obs = int(np.count_nonzero(obs_np)) if obs_np is not None else 0
                n_exp = int(np.count_nonzero(exp_np)) if exp_np is not None else 0
            if obs_np is None or (n_obs == 0 and n_exp == 0):
                return 0.0
        elif obs_np is None:
            return 0.0 if require_map else requested

        try:
            xyt = np.asarray(self.robot.get_base_pose(), dtype=np.float64).reshape(-1)
        except Exception:
            return 0.0 if require_map else requested
        if xyt.size < 3:
            return 0.0 if require_map else requested
        x0, y0, th = float(xyt[0]), float(xyt[1]), float(xyt[2])
        c, s = float(np.cos(th)), float(np.sin(th))
        traveled = 0.0
        step = max(0.02, float(step_m))
        clear = max(0.0, float(clearance_m))
        while traveled + step <= requested + 1e-9:
            probe = traveled + step
            xy = np.array([x0 + probe * c, y0 + probe * s], dtype=np.float64)
            try:
                grid = vm.xy_to_grid_coords(xy)
            except Exception:
                break
            if grid is None:
                break
            if hasattr(grid, "detach"):
                grid = grid.detach().cpu().numpy()
            gi, gj = int(grid[0]), int(grid[1])
            if gi < 0 or gj < 0 or gi >= obs_np.shape[0] or gj >= obs_np.shape[1]:
                break
            if bool(obs_np[gi, gj]):
                return max(0.0, traveled - clear)
            if require_map and exp_np is not None and not bool(exp_np[gi, gj]):
                # Do not leave the explored (incl. local_radius) disk into unknown space.
                return traveled
            traveled = probe
        return requested

    def move_forward_meters(self, meters: float) -> float:
        """Drive forward along current heading; clips for obstacles. Returns distance commanded."""
        requested = float(np.clip(float(meters), 0.0, 1.5))
        dist = self.clip_forward_distance_m(requested)
        if dist < 0.02:
            self.announce_action("Cannot move forward — need explored free space (scan?) or obstacle too close")
            return 0.0
        if dist + 1e-3 < requested:
            self.announce_action(f"Moving forward {dist:.2f} m (map-clipped from {requested:.2f} m)")
        else:
            self.announce_action(f"Moving forward {dist:.2f} m (map clear)")
        nav_timeout = self._find_phase_nav_timeout()
        if hasattr(self.robot, "move_to_nav_posture"):
            self.robot.move_to_nav_posture()
        # Relative body-frame: +x forward.
        self.robot.move_base_to(
            [float(dist), 0.0, 0.0],
            relative=True,
            blocking=True,
            timeout=nav_timeout,
        )
        if not getattr(self, "_realtime_updates", False):
            try:
                self.update()
            except Exception:
                pass
        return dist

    def _maybe_emit_navgrid_ascii(self, *, context: str = "") -> None:
        from emet.mapping.debug_navgrid_ascii import (
            build_navgrid_from_voxel_map,
            maybe_print_navgrid_ascii,
            navgrid_context_allowed,
        )

        if not navgrid_context_allowed(context):
            return
        try:
            robot_xy = self.world_base_xy()
        except Exception:
            robot_xy = None
        try:
            text = build_navgrid_from_voxel_map(
                self.voxel_map,
                graph_memory=getattr(self, "graph_memory", None),
                robot_xy=robot_xy,
            )
            if context:
                text = f"[navgrid:{context}]\n{text}"
            maybe_print_navgrid_ascii(text)
        except Exception as exc:
            logger.warning(f"Navgrid ASCII render skipped: {exc}")

    def _filter_unsafe_nav_traj(
        self,
        traj: list,
        *,
        start_xyt: np.ndarray | list[float] | None = None,
    ) -> tuple[list, str | None, float | None]:
        """Drop low-clearance / unexplored waypoints before confirm/exec.

        Returns:
            (filtered_traj, reject_reason, min_clearance_m). reject_reason is set when the
            executable chunk would be empty after filtering (same as planner failure).
        """
        if not traj:
            return [], "no_plan", None
        planner = getattr(self, "planner", None)
        if planner is None:
            return list(traj), None, None
        if getattr(planner, "_clearance_m", None) is None:
            try:
                planner.reset()
            except Exception:
                pass

        min_c = float(getattr(self, "_min_clearance_m", getattr(planner, "min_clearance_m", 0.0)) or 0.0)
        # Preserve trailing [nan, object_xyz] marker if present.
        object_tail: list = []
        body = list(traj)
        if len(body) >= 2:
            mid = np.asarray(body[-2], dtype=np.float64).reshape(-1)
            if mid.size >= 2 and np.isnan(mid[:2]).all():
                object_tail = body[-2:]
                body = body[:-2]

        start_xy = None
        if start_xyt is not None:
            s = np.asarray(start_xyt, dtype=np.float64).reshape(-1)
            if s.size >= 2 and np.isfinite(s[:2]).all():
                start_xy = (float(s[0]), float(s[1]))

        kept: list = []
        reject: str | None = None
        clearances: list[float] = []
        prev_xy: tuple[float, float] | None = None
        prev_is_start = False
        for raw in body:
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if arr.size < 2 or not np.isfinite(arr[:2]).all():
                continue
            xy = (float(arr[0]), float(arr[1]))
            # Always keep the first waypoint when it matches start (robot may sit in tight clearance).
            is_start = start_xy is not None and abs(xy[0] - start_xy[0]) < 1e-3 and abs(xy[1] - start_xy[1]) < 1e-3
            if not is_start and not planner.is_explored_xy(xy):
                reject = "rejected_unexplored"
                break
            c = float(planner.clearance_at_xy(xy))
            clearances.append(c)
            if not is_start and min_c > 0 and c < min_c:
                reject = "rejected_low_clearance"
                break
            # Mid-segment samples: reject chords that scrape low-clearance cells
            # between two individually-safe waypoints (post-simplify hazard).
            # The first segment may leave a tight start cell for the planner's
            # nearest clearance-safe escape cell.
            if prev_xy is not None and not is_start and not prev_is_start and hasattr(planner, "to_pt"):
                try:
                    if not planner.is_in_line_of_sight(planner.to_pt(prev_xy), planner.to_pt(xy)):
                        reject = "rejected_low_clearance_segment"
                        break
                except Exception:
                    pass
            kept.append(raw if isinstance(raw, list) else arr.tolist())
            prev_xy = xy
            prev_is_start = is_start

        min_along = float(min(clearances)) if clearances else None
        if not kept:
            return [], reject or "rejected_low_clearance", min_along
        if reject is not None and len(kept) <= 1 and start_xy is not None:
            # Only start survived → nothing useful to execute.
            return [], reject, min_along
        if object_tail:
            kept.extend(object_tail)
        return kept, None, min_along

    def _mark_nav_goal_blocked(self, *, reason: str = "aborted_waypoint_timeout") -> None:
        """Remember the last nav goal so explore multi-goal A* skips it next time."""
        blocked = getattr(self, "_habitat_blocked_goals", None)
        if blocked is None:
            self._habitat_blocked_goals = set()
            blocked = self._habitat_blocked_goals
        recent = getattr(self, "_habitat_recent_goals", None)
        if recent is None:
            self._habitat_recent_goals = []
            recent = self._habitat_recent_goals

        meta = dict(getattr(self, "_last_nav_plan", None) or {})
        candidates: list[tuple[float, float]] = []
        for key in ("goal_xyt", "object_xyz", "effective_goal_xy"):
            raw = meta.get(key)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                candidates.append((float(arr[0]), float(arr[1])))
        traj = meta.get("traj") or []
        for p in reversed(list(traj)):
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                candidates.append((float(arr[0]), float(arr[1])))
                break
        for xy in candidates:
            key = goal_key_xy(xy)
            blocked.add(key)
            recent.append(key)
        del recent[:-16]
        self._record_nav_plan_fields(outcome=reason, blocked_after_abort=True)
        logger.warning(f"Nav abort ({reason}): marked {len(candidates)} goal key(s) blocked for replan/explore skip")

    def _record_nav_plan_fields(self, **fields: Any) -> None:
        meta = dict(getattr(self, "_last_nav_plan", None) or {})
        meta.update(fields)
        self._last_nav_plan = meta

    def execute_action(
        self,
        text: str,
    ) -> tuple[bool | None, np.ndarray | None]:
        """
        This function is used to navigate the robot give text query.
        It will call the process_text function to get the trajectory for the robot to follow.
        It will then execute the trajectory using the execute_trajectory function.
        If text is empty, it will just explore the environment.

        Args:
            text: The text query for the robot to navigate to / explore.

        Returns:
            The first element is a boolean indicating whether the navigation is finished. If it is None, it means the navigation has some problem.
            The second element is the location of the target object, useful used to tell the robot how to orient itself and prepare pregrasp pose for manipulation.
                If it is None, it means the navigation has some problem.
        """
        if not self._realtime_updates:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        self.robot.switch_to_navigation_mode()

        start = self._planning_base_xyt(self.robot.get_base_pose())
        res = self.process_text(text, start)
        if len(res) == 0 and text != "" and text is not None:
            res = self.process_text("", start)

        if len(res) > 0:
            plan_meta = getattr(self, "_last_nav_plan", None) or {}
            announce = plan_meta.get("announce") or "Navigating…"
            if not str(announce).lower().startswith("navigat"):
                announce = f"Navigating… {announce}"
            # Confirm before posture/exec so operators can reject wall-hugging plans.
            object_xyz = None
            if len(res) >= 2 and np.isnan(np.asarray(res[-2], dtype=np.float64)).all():
                object_xyz = res[-1]
            from emet.controller.nav_confirm import confirm_navigation_plan

            if not confirm_navigation_plan(self, res, meta=plan_meta, object_xyz=object_xyz):
                self._record_nav_plan_fields(outcome="user_cancelled", confirmed=False)
                return None, None
            self._record_nav_plan_fields(confirmed=True, outcome="executing")
            self.announce_action(announce)
            n_exec = sum(1 for p in res if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all())
            logger.info(
                "Navigation plan OK; executing %d waypoints (localize=%s mode=%s path≈%.2fm chunked=%s)",
                n_exec,
                plan_meta.get("localize_source", "?"),
                plan_meta.get("mode", "?"),
                float(plan_meta.get("path_m") or 0.0),
                bool(plan_meta.get("chunked")),
            )
            nav_timeout = self._find_phase_nav_timeout()
            wait_obs = getattr(self.robot, "wait_for_obs", None)
            if wait_obs is not None:
                wait_obs(timeout=nav_timeout)
            if self._navigation_origin_xyt() is None:
                logger.warning(
                    "navigation_origin_xyt missing from emet_session; sim nav may use wrong frame "
                    "(restart sim server and ensure first observation arrived)."
                )
            # process_text ends with robot.say(...); re-sync nav posture + forward gaze before base moves.
            self.robot.move_to_nav_posture()
            self.robot.look_front(blocking=True)
            time.sleep(DYNAMEM_HEAD_SETTLE_S)
            # This means that the robot has already finished all of its trajectories and should stop to manipulate the object.
            # We will append a nan and point coordinates of the target object on the trajectory to denote that the robot is reaching the target point
            if len(res) >= 2 and np.isnan(res[-2]).all():
                if len(res) > 2:
                    exec_ok = self.robot.execute_trajectory(
                        res[:-2],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                        per_waypoint_timeout=nav_timeout,
                        final_timeout=max(nav_timeout, 30.0),
                        blocking=True,
                        world_frame=True,
                    )
                    if exec_ok is False:
                        self._record_nav_plan_fields(outcome="aborted_waypoint_timeout")
                        self._mark_nav_goal_blocked(reason="aborted_waypoint_timeout")
                        logger.warning("Navigation aborted: waypoint timeout during execute_trajectory")
                        return None, None

                self.robot.look_front()
                self.update()
                self._record_nav_plan_fields(outcome="ok")
                return True, res[-1]
            # The robot has not reached the object. Next it should look around and continue navigation
            else:
                exec_ok = self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    per_waypoint_timeout=nav_timeout,
                    final_timeout=max(nav_timeout, 30.0),
                    blocking=True,
                    world_frame=True,
                )
                if exec_ok is False:
                    self._record_nav_plan_fields(outcome="aborted_waypoint_timeout")
                    self._mark_nav_goal_blocked(reason="aborted_waypoint_timeout")
                    logger.warning("Navigation aborted: waypoint timeout during execute_trajectory")
                    return None, None
                self.robot.look_front()
                self.update()
                self._record_nav_plan_fields(outcome="ok_chunk")
                return False, None
        else:
            logger.warning("No plan from process_text; try again.")
            return None, None

    def run_exploration(self):
        """
        Go through exploration when the robot has not received any text query from the user.
        We use the voxel_grid map created by our collector to sample free space, and then use A* planner to get there.
        """

        self.announce_action("Exploring…")
        # "" means the robot has not received any text query from the user and should conduct exploration just to better know the environment
        status, _ = self.execute_action("")
        if status is None:
            self.announce_action("Exploring… no valid frontier right now")
            logger.warning("Exploration failed (no valid plan or frontier).")
            return False
        self._maybe_emit_navgrid_ascii(context="explore")
        return True

    def process_text(self, text, start_pose):
        """
        Process the text query and return the trajectory for the robot to follow.
        """

        logger.debug("process_text: %r", text)

        clear_nav = getattr(self.rerun_visualizer, "clear_nav_plan", None)
        if callable(clear_nav):
            clear_nav()
        else:
            self.rerun_visualizer.clear_identity("world/object")
            self.rerun_visualizer.clear_identity("world/xyt_goal")
            self.rerun_visualizer.clear_identity("world/robot_start_pose")
            self.rerun_visualizer.clear_identity("world/direction")
        self.rerun_visualizer.clear_identity("robot_monologue")
        self.rerun_visualizer.clear_identity("/observation_similar_to_text")
        self._last_nav_plan = None

        debug_text = ""
        mode = "navigation"
        localize_source = ""
        obs = None
        localized_point = None
        waypoints = None

        if text is not None and text != "" and self.space.traj is not None:
            logger.debug("Reusing saved trajectory target: %s", self.space.traj)
            traj_target_point = self.space.traj[-1]
            if hasattr(self.encoder, "feature_matching_threshold") and self.voxel_map.verify_point(
                text,
                traj_target_point,
                similarity_threshold=self.encoder.feature_matching_threshold,
            ):
                localized_point = traj_target_point
                localize_source = "saved_traj+verify"
                debug_text += "## Reusing prior plan target (SigLIP neighborhood OK).\n"
            elif hasattr(self.encoder, "feature_matching_threshold") and _finite_xyz_traj_target(traj_target_point):
                # Short queries ("red object") often fail SigLIP neighborhood re-check; still navigate to last grounding.
                localized_point = traj_target_point
                localize_source = "saved_traj"
                debug_text += "## Reusing prior plan target; semantic re-check was not decisive.\n"

        logger.debug("Target verification done (localized_point=%s)", localized_point is not None)

        if text is not None and text != "" and localized_point is None:
            graph_point = self._localize_point_from_graph_memory(text)
            if graph_point is not None:
                localized_point = graph_point
                localize_source = "graph"
                debug_text += "## Localized target from graph memory.\n"
                mode = "navigation"
                logger.info("Localized %r from graph memory at %s", text, np.asarray(graph_point).reshape(-1)[:3])

        if text is not None and text != "" and localized_point is None:
            det = getattr(self.voxel_map, "detection_model", None)
            if det is not None or self.encoder is not None:
                try:
                    (
                        localized_point,
                        loc_debug,
                        obs,
                        pointcloud,
                    ) = self.voxel_map.localize_text(text, debug=True, return_debug=True)
                    if localized_point is not None:
                        localize_source = "voxel"
                        debug_text += "## Localized target from voxel semantic memory.\n"
                    if loc_debug:
                        debug_text += str(loc_debug)
                    logger.info("Localized %r from voxel map: %s", text, localized_point is not None)
                except Exception as exc:
                    logger.debug("voxel localize_text failed for %r: %s", text, exc)

        # Do Frontier based exploration (optionally biased by the active EQA question).
        if text is None or text == "" or localized_point is None:
            debug_text += "## No object localization; falling back to frontier exploration.\n"
            frontier_text = self._exploration_text(text)
            explore_pt = pick_uncovered_explore_target(
                self,
                question=frontier_text or None,
                blocked=getattr(self, "_habitat_blocked_goals", None),
                recent_goals=getattr(self, "_habitat_recent_goals", None),
            )
            if explore_pt is not None:
                localized_point = explore_pt
                localize_source = "frontier_uncovered"
                debug_text += "## Selected blocked-aware explore frontier.\n"
                mode = "exploration"
            else:
                graph_frontier = self._best_frontier_point_from_graph(frontier_text)
                if graph_frontier is not None:
                    localized_point = graph_frontier
                    localize_source = "frontier_graph"
                    debug_text += "## Selected frontier target from graph memory.\n"
                    mode = "exploration"
                else:
                    localized_point = self.space.sample_frontier(self.planner, start_pose, frontier_text)
                    localize_source = "frontier_space" if localized_point is not None else ""
                    mode = "exploration"

        if obs is not None and mode == "navigation":
            obs = self.voxel_map.find_obs_id_for_text(text)
            if obs is not None:
                try:
                    idx = int(obs.item()) if hasattr(obs, "item") else int(obs)
                    if 0 < idx <= len(self.voxel_map.observations):
                        rgb = self.voxel_map.observations[idx - 1].rgb
                        self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb)
                except (TypeError, ValueError, IndexError):
                    pass

        if localized_point is None:
            logger.warning("process_text: no localized point for query %r", text)
            return []

        # TODO: Do we really need this line?
        if len(localized_point) == 2:
            localized_point = np.array([localized_point[0], localized_point[1], 0])

        _lp = np.asarray(
            localized_point.detach().cpu().numpy() if isinstance(localized_point, torch.Tensor) else localized_point,
            dtype=np.float64,
        ).reshape(-1)
        ox, oy = float(_lp[0]), float(_lp[1])
        oz = float(_lp[2]) if _lp.size > 2 else 1.5
        if not np.isfinite(oz) or abs(oz) < 1e-9:
            oz = 1.5

        waypoints = None
        n_planned = 0
        res = None
        point = None

        # Exploration: top-K frontiers → one multi-goal A* (skip sealed / unreachable).
        # Object nav stays single-goal.
        if mode == "exploration" and isinstance(self.planner, AStar):
            from emet.motion.frontier_goals import collect_explore_frontier_candidates

            frontier_text = self._exploration_text(text)
            cands = collect_explore_frontier_candidates(
                self,
                question=frontier_text or None,
                k=8,
                blocked=getattr(self, "_habitat_blocked_goals", None),
                recent_goals=getattr(self, "_habitat_recent_goals", None),
                seeds=[localized_point],
            )
            object_xys: list[np.ndarray] = []
            nav_goals: list[np.ndarray] = []
            for cand in cands:
                g = self.space.sample_navigation(start_pose, self.planner, cand)
                if g is None:
                    continue
                object_xys.append(np.asarray(cand, dtype=np.float64).reshape(-1))
                nav_goals.append(np.asarray(g, dtype=np.float64).reshape(-1))

            if len(nav_goals) >= 2:
                res = self.planner.plan(start_pose, nav_goals[0], goals=nav_goals)
                gi = getattr(res, "goal_index", None) if res is not None else None
                if res is not None and res.success and gi is not None and 0 <= int(gi) < len(nav_goals):
                    gi_i = int(gi)
                    point = nav_goals[gi_i]
                    localized_point = object_xys[gi_i]
                    _lp = np.asarray(localized_point, dtype=np.float64).reshape(-1)
                    ox, oy = float(_lp[0]), float(_lp[1])
                    oz = float(_lp[2]) if _lp.size > 2 else 1.5
                    if not np.isfinite(oz) or abs(oz) < 1e-9:
                        oz = 1.5
                    localize_source = f"{localize_source or 'frontier'}_multi_goal"
                    logger.info(
                        "Multi-goal explore: %d candidates, chose index=%d xy=(%.2f, %.2f)",
                        len(nav_goals),
                        gi_i,
                        ox,
                        oy,
                    )
                elif res is not None and not res.success:
                    logger.warning("Multi-goal explore plan failed: %s", res.reason)
                    res = None
            elif len(nav_goals) == 1:
                point = nav_goals[0]
                localized_point = object_xys[0]
                _lp = np.asarray(localized_point, dtype=np.float64).reshape(-1)
                ox, oy = float(_lp[0]), float(_lp[1])
                res = self.planner.plan(start_pose, point)

        if point is None and res is None:
            point = self.space.sample_navigation(start_pose, self.planner, localized_point)

        logger.info(
            "Nav endpoint sample: localize=%s target_xy=(%.2f, %.2f) base_goal=%s",
            localize_source or "?",
            ox,
            oy,
            None if point is None else np.asarray(point).reshape(-1)[:3],
        )

        if res is None:
            if point is None:
                logger.warning("No navigation endpoint sampled (planner may fail).")
            else:
                res = self.planner.plan(start_pose, point)

        if res is not None and res.success:
            waypoints = [pt.state for pt in res.trajectory]
            n_planned = len(waypoints)
        elif res is not None:
            waypoints = None
            logger.warning("Planner failure: %s", res.reason)

        # If we are navigating to some object of interest, send (x, y, z) of
        # the object so that we can make sure the robot looks at the object after navigation
        traj = []
        chunked = False
        full_traj_for_viz = None
        if waypoints is not None:
            finished = len(waypoints) <= 8 and mode == "navigation"
            chunked = not finished
            full_traj_for_viz = self.planner.clean_path_for_xy(
                list(waypoints), start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0
            )
            if finished:
                self.space.traj = None
            else:
                self.space.traj = waypoints[8:] + [[np.nan, np.nan, np.nan], localized_point]
            if not finished:
                waypoints = waypoints[:8]
            traj = self.planner.clean_path_for_xy(
                waypoints, start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0
            )
            if finished:
                traj.append([np.nan, np.nan, np.nan])
                if isinstance(localized_point, torch.Tensor):
                    localized_point = localized_point.tolist()
                traj.append(localized_point)
            traj, reject_reason, min_clr = self._filter_unsafe_nav_traj(traj, start_xyt=start_pose)
            if reject_reason is not None or not traj:
                logger.warning(
                    "Nav plan rejected after safety filter: %s (min_clearance=%s)",
                    reject_reason,
                    min_clr,
                )
                self._last_nav_plan = {
                    "mode": mode,
                    "localize_source": localize_source,
                    "n_planned": n_planned,
                    "chunked": chunked,
                    "path_m": 0.0,
                    "min_clearance_m": min_clr,
                    "outcome": reject_reason or "rejected_low_clearance",
                    "announce": f"Plan rejected ({reject_reason or 'unsafe'})",
                    "traj": [],
                }
                return []
            logger.info(
                "Planned trajectory: %d exec / %d planned waypoints (finished_chunk=%s min_clearance=%.3f)",
                len([p for p in traj if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all()]),
                n_planned,
                finished,
                float(min_clr) if min_clr is not None else float("nan"),
            )

        # Talk about what you are doing, as the robot.
        if self.robot is not None:
            if text is not None and text != "":
                self.robot.say("I am looking for a " + text + ".")
            else:
                self.robot.say("I am exploring the environment.")

        if text is not None and text != "":
            debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
        else:
            debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
        debug_text += (
            f"\n### Plan: mode=`{mode}` localize=`{localize_source or 'n/a'}` "
            f"planned_wps={n_planned} chunked={chunked}\n"
        )
        debug_text = "# Robot's monologue: \n" + debug_text
        self._rerun_monologue_base = debug_text
        self._rerun_refresh_monologue_panel()

        log_plan = getattr(self.rerun_visualizer, "log_nav_plan", None)
        if callable(log_plan) and traj:
            self._last_nav_plan = log_plan(
                traj,
                full_traj=full_traj_for_viz,
                start_xyt=start_pose,
                goal_xyt=point,
                object_xyz=[ox, oy, oz],
                mode=mode,
                localize_source=localize_source,
                query=text or "",
                n_planned=n_planned or None,
                chunked=chunked,
            )
            # Attach clearance / safety fields for agent tools.
            try:
                clr = self.planner.clearance_at_xy(start_pose[:2])
                path_clrs = [
                    self.planner.clearance_at_xy(np.asarray(p).reshape(-1)[:2])
                    for p in traj
                    if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all()
                ]
                self._record_nav_plan_fields(
                    min_clearance_m=float(min(path_clrs)) if path_clrs else None,
                    base_clearance_m=float(clr),
                    min_clearance_required_m=float(getattr(self, "_min_clearance_m", 0.0)),
                    traj=list(traj),
                )
            except Exception:
                pass
        elif traj:
            # NullVisualizer / older stubs: keep minimal legacy arrows.
            origins = []
            vectors = []
            for idx in range(len(traj) - 1):
                a = np.asarray(traj[idx], dtype=np.float64).reshape(-1)
                b = np.asarray(traj[idx + 1], dtype=np.float64).reshape(-1)
                if a.size < 2 or b.size < 2 or not np.isfinite(a[:2]).all() or not np.isfinite(b[:2]).all():
                    continue
                origins.append([float(a[0]), float(a[1]), 1.5])
                vectors.append([float(b[0] - a[0]), float(b[1] - a[1]), 0.0])
            if origins:
                self.rerun_visualizer.log_arrow3D("world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1)
            path_clrs = [
                self.planner.clearance_at_xy(np.asarray(p).reshape(-1)[:2])
                for p in traj
                if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all()
            ]
            self._last_nav_plan = {
                "mode": mode,
                "localize_source": localize_source,
                "n_planned": n_planned,
                "chunked": chunked,
                "path_m": 0.0,
                "min_clearance_m": float(min(path_clrs)) if path_clrs else None,
                "min_clearance_required_m": float(getattr(self, "_min_clearance_m", 0.0)),
                "announce": f"Navigating via {localize_source or mode}: {n_planned} wps",
                "traj": list(traj),
            }

        return traj

    def navigate(self, text, max_step=10):
        """
        The robot calls this function to navigate to the object.
        It will call execute_action function until it is ready for manipulation
        """
        # Do not call rr.init here during normal live viewing: RerunVisualizer already called
        # rr.init + rr.serve; a second init clears the recording and the ZMQ Rerun thread appears empty.
        if self.save_rerun:
            rr.init("Stretch_robot", recording_id=uuid4(), spawn=has_display())
            if not os.path.exists(self.log):
                os.makedirs(self.log)
            rr.save(self.log + "/" + "data_" + str(self.rerun_iter) + ".rrd")
        finished = False
        step = 0
        end_point = None
        while not finished and step < max_step:
            logger.debug("navigate step %s/%s", step, max_step)
            step += 1
            finished, end_point = self.execute_action(text)
            if finished is None:
                logger.warning("Navigation failed (blocked or no progress).")
                return None
        return end_point

    def place(
        self,
        text,
        local=True,
        init_tilt=INIT_HEAD_TILT,
        base_node="camera_depth_optical_frame",
    ):
        """
        An API for running placing. By calling this API, human will ask the robot to place whatever it holds
        onto objects specified by text queries A
        - hello_robot: a wrapper for home-robot StretchClient controller
        - socoket: we use this to communicate with workstation to get estimated gripper pose
        - text: queries specifying target object
        - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
        - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
        """
        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()
        self.manip_wrapper.move_to_position(head_pan=INIT_HEAD_PAN, head_tilt=init_tilt)

        if not local:
            rotation, translation = capture_and_process_image(
                mode="place",
                obj=text,
                socket=self.manip_socket,
                hello_robot=self.manip_wrapper,
            )
        else:
            if self.owl_sam_detector is None:
                # We can opt to use OWLv2 + SAMv2 for accurate object detection, but the placing receptacles are usually very easy to detect,
                # so we don't see the point of installing SAMv2 and using it
                # from emet.perception.detection.owl import OWLSAMProcessor

                # self.owl_sam_detector = OWLSAMProcessor(confidence_threshold=0.1)

                # A misnomer, this is actually YOLOE while its named as self.owl_sam_detector
                self.owl_sam_detector = YoloEPerception(confidence_threshold=0.05, size="l", device=self.device)
            rotation, translation = process_image_for_placing(
                obj=text,
                hello_robot=self.manip_wrapper,
                detection_model=self.owl_sam_detector,
                save_dir=self.log,
            )
        logger.debug("Place: rotation=%s translation=%s", rotation, translation)

        if rotation is None:
            return False

        # lift arm to the top before the robot extends the arm, prepare the pre-placing gripper pose
        self.manip_wrapper.move_to_position(lift_pos=1.05)
        self.manip_wrapper.move_to_position(wrist_yaw=0, wrist_pitch=0)

        # Placing the object
        move_to_point(self.manip_wrapper, translation, base_node, self.transform_node, move_mode=0)
        self.manip_wrapper.move_to_position(gripper_pos=1, blocking=True)

        # Lift the arm a little bit, and rotate the wrist roll of the robot in case the object attached on the gripper
        self.manip_wrapper.move_to_position(lift_pos=min(self.manip_wrapper.robot.get_six_joints()[1] + 0.3, 1.1))
        self.manip_wrapper.move_to_position(wrist_roll=2.5, blocking=True)
        self.manip_wrapper.move_to_position(wrist_roll=-2.5, blocking=True)

        # Wait for some time and shrink the arm back
        self.manip_wrapper.move_to_position(gripper_pos=1, lift_pos=1.05, arm_pos=0)
        self.manip_wrapper.move_to_position(wrist_pitch=-1.57)

        # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
        self.manip_wrapper.move_to_position(base_trans=-self.manip_wrapper.robot.get_six_joints()[0])
        return True

    def get_voxel_map(self):
        """Return the voxel map"""
        return self.voxel_map

    def manipulate(
        self,
        text,
        init_tilt=INIT_HEAD_TILT,
        base_node="camera_depth_optical_frame",
        skip_confirmation: bool = False,
    ):
        """
        An API for running manipulation. By calling this API, human will ask the robot to pick up objects
        specified by text queries A
        - hello_robot: a wrapper for home-robot StretchClient controller
        - socoket: we use this to communicate with workstation to get estimated gripper pose
        - text: queries specifying target object
        - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
        - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
        """

        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()

        gripper_pos = 1

        self.manip_wrapper.move_to_position(
            arm_pos=INIT_ARM_POS,
            head_pan=INIT_HEAD_PAN,
            head_tilt=init_tilt,
            gripper_pos=gripper_pos,
            lift_pos=INIT_LIFT_POS,
            wrist_pitch=INIT_WRIST_PITCH,
            wrist_roll=INIT_WRIST_ROLL,
            wrist_yaw=INIT_WRIST_YAW,
        )

        rotation, translation, depth, width = capture_and_process_image(
            mode="pick",
            obj=text,
            socket=self.manip_socket,
            hello_robot=self.manip_wrapper,
        )

        if rotation is None:
            return False

        if width < 0.05 and self.re == 3:
            gripper_width = 0.45
        elif width < 0.075 and self.re == 3:
            gripper_width = 0.6
        else:
            gripper_width = 1

        if not skip_confirmation and input("Do you want to do this manipulation? Y or N ") == "N":
            return False

        pickup(
            self.manip_wrapper,
            rotation,
            translation,
            base_node,
            self.transform_node,
            gripper_depth=depth,
            gripper_width=gripper_width,
        )

        # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
        self.manip_wrapper.move_to_position(base_trans=-self.manip_wrapper.robot.get_six_joints()[0])

        return True

    def _patch_images(self, images: list[Image.Image], patch_size=(480, 640), gap=5):
        """
        Patch a list of PIL Images into a numpy array, used for dicrod bot
        """
        # Resize all images to the same patch size
        images = [img.resize(patch_size) for img in images]

        # Calculate total width and height
        n_images = len(images)
        total_width = patch_size[0] * n_images + gap * (n_images - 1)
        total_height = patch_size[1]

        # Create a blank canvas
        canvas = Image.new("RGB", (total_width, total_height))

        # Paste images side-by-side
        for idx, img in enumerate(images):
            x = idx * (patch_size[0] + gap)
            canvas.paste(img, (x, 0))

        # Convert to numpy array
        return np.array(canvas)

    def run_eqa(self, question, max_planning_steps: int = 5):
        """
        API for calling EQA module
        """
        # See navigate(): avoid rr.init during live Rerun streaming (would reset the recording).
        if self.save_rerun:
            rr.init("Stretch_robot", recording_id=uuid4(), spawn=has_display())
            if not os.path.exists(self.log):
                os.makedirs(self.log)
            rr.save(self.log + "/" + "data_" + str(self.rerun_iter) + ".rrd")

        self.robot.switch_to_navigation_mode()

        discord_text, relevant_images = "", []

        # Early-stop: when exploration stalls (the scene graph gains no new nodes) yet the
        # model keeps returning the same answer, further planning steps re-ask with
        # identical inputs and cannot change the result — common when a question keyword
        # never becomes a node label (abstract/action words) or the robot is physically
        # stuck. Stop after ``stall_patience`` such steps. Productive exploration (a growing
        # graph) always continues, so this never cuts a run that is still gathering evidence.
        stall_patience = int(self.parameters.get("eqa_stall_patience", 4) or 0)
        prev_node_count = -1
        prev_answer = None
        stall = 0

        for _cnt_step in range(max_planning_steps):
            logger.info(
                "EQA planning step %d/%d for %r",
                _cnt_step + 1,
                max_planning_steps,
                question if isinstance(question, str) else str(question)[:80],
            )
            answer, discord_text, relevant_images, confidence = self.run_eqa_one_iter(question)
            if confidence:
                self.robot.say("The answer to " + question + " is " + answer)
                break

            if stall_patience > 0 and self.graph_memory is not None:
                # Never early-stop on a repeated Yes/No while question objects are still
                # uncovered — absence is not evidence; keep exploring frontiers.
                covers = getattr(self.graph_memory, "_graph_covers_relevant_objects", None)
                uncovered = bool(callable(covers) and not covers())
                if uncovered:
                    stall = 0
                    prev_node_count = len(self.graph_memory.get_nodes())
                    prev_answer = self.graph_memory.last_eqa_parsed[1]
                else:
                    node_count = len(self.graph_memory.get_nodes())
                    cur_answer = self.graph_memory.last_eqa_parsed[1]
                    if node_count <= prev_node_count and cur_answer and cur_answer == prev_answer:
                        stall += 1
                    else:
                        stall = 0
                    prev_node_count = node_count
                    prev_answer = cur_answer
                    if stall >= stall_patience:
                        logger.info(
                            "EQA early stop after %d/%d planning steps: exploration stalled (no new graph "
                            "nodes, stable answer %r) for %d steps; accepting the answer.",
                            _cnt_step + 1,
                            max_planning_steps,
                            cur_answer,
                            stall + 1,
                        )
                        break

        relevant_image = self._patch_images(relevant_images, patch_size=(270, 360))
        self.rerun_iter += 1

        return discord_text, relevant_image

    def run_eqa_one_iter(self, question, max_movement_step: int = 5):
        answer_output = None

        if not self._realtime_updates and not getattr(self, "_fast_explore_lookaround", False):
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()
        elif not self._realtime_updates:
            # Explore-loop (`_fast_explore_lookaround`) already swept / mapped; skip another ~60s look_around.
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        try:
            logger.info("EQA query_answer start for %r", question if isinstance(question, str) else str(question)[:80])
            t_qa0 = time.monotonic()
            (
                reasoning,
                answer,
                confidence,
                confidence_reasoning,
                target_point,
                relevant_images,
            ) = self.voxel_map.query_answer(question, self._planning_base_xyt(self.robot.get_base_pose()), self.planner)
            logger.info(
                "EQA query_answer done wall_s=%.1f confidence=%s answer=%r",
                time.monotonic() - t_qa0,
                confidence,
                answer,
            )
        except:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images = (
                "Exception happens in LLM querying!",
                "Unknown",
                False,
                "Exception happens in LLM querying!",
                self.space.sample_frontier(
                    self.planner, self._planning_base_xyt(self.robot.get_base_pose()), text=None
                ),
                [],
            )

        # Log the texts to rerun visualizer
        confidence_text = "I am confident with the answer" if confidence else "I am NOT confident with the answer"

        reasoning_output = (
            "\n#### Reasoning for the answer: " + reasoning
            if confidence
            else "\n#### Reasoning for the confidence: " + confidence_reasoning
        )

        answer_output = (
            "#### **Question:** "
            + question
            + "\n#### **Answer:** "
            + answer
            + "\n#### **Confidence:** "
            + confidence_text
            + reasoning_output
        )

        self._rerun_monologue_base = answer_output
        self._rerun_refresh_monologue_panel()
        if len(relevant_images) != 0:
            self.rerun_visualizer.log_custom_2d_image(
                "/observation_similar_to_text", self._patch_images(relevant_images)
            )

        # chat with user in the rerun
        if confidence:
            discord_text = answer + ". I believe this answer is correct because " + reasoning
        else:
            discord_text = "I am not confident to answer the question because " + confidence_reasoning

        discord_text += "\nI also provide relevant images here."

        if confidence:
            return answer, discord_text, relevant_images, confidence

        start_pose = self._planning_base_xyt(self.robot.get_base_pose())

        logger.debug("EQA navigate: target_point=%s", target_point)
        if target_point is None:
            # No usable navigation target (degenerate action parsed no image index): skip movement.
            return answer, discord_text, relevant_images, confidence

        # If we want to explore non obstacles (especially frontiers), remember where we currently want to face
        obstacles, _ = self.voxel_map.get_2d_map()
        target_grid = self.voxel_map.xy_to_grid_coords((target_point[0], target_point[1]))
        if not obstacles[int(target_grid[0]), int(target_grid[1])]:
            target_theta = self.space.sample_navigation(start_pose, self.planner, target_point)[-1]
            logger.debug("EQA navigate: target_theta=%s", target_theta)
        else:
            target_theta = None

        movement_step = 0
        while movement_step < max_movement_step:
            start_pose = self._planning_base_xyt(self.robot.get_base_pose())
            movement_step += 1
            self.update()
            finished = self.navigate_to_target_pose(target_point, start_pose, target_theta)
            if finished.finished:
                break

        return answer, discord_text, relevant_images, confidence

    def _log_nav_attempt(
        self,
        nav_res: NavAttemptResult,
        *,
        target_obs_id: int | None,
        goal_xy: np.ndarray,
    ) -> None:
        recorder = getattr(self, "_episode_diagnostics_recorder", None)
        if recorder is not None and hasattr(recorder, "append_nav_attempt"):
            row = {
                "target_obs_id": target_obs_id,
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                "effective_goal_xy": (
                    [float(nav_res.effective_goal_xy[0]), float(nav_res.effective_goal_xy[1])]
                    if getattr(nav_res, "effective_goal_xy", None)
                    else None
                ),
                "method": nav_res.method,
                "success": nav_res.success,
                "finished": nav_res.finished,
                "dist_m": nav_res.dist_m,
                "note": nav_res.note,
            }
            if getattr(nav_res, "path_xy", None):
                row["path_xy"] = nav_res.path_xy
            recorder.append_nav_attempt(row)
        if nav_res.finished or nav_res.success:
            eff = getattr(nav_res, "effective_goal_xy", None) or (
                float(goal_xy[0]),
                float(goal_xy[1]),
            )
            key = goal_key_xy(eff)
            recent = getattr(self, "_habitat_recent_goals", None)
            if recent is not None:
                recent.append(key)
                del recent[:-8]
        elif (
            str(nav_res.note or "").startswith("already_at_goal")
            or (not nav_res.finished and float(nav_res.dist_m) < 0.08)
            or (not nav_res.success and float(nav_res.dist_m) < 0.12)
        ):
            # Stuck / noop / no-progress: remember so uncovered explore does not re-pick.
            eff = getattr(nav_res, "effective_goal_xy", None) or (
                float(goal_xy[0]),
                float(goal_xy[1]),
            )
            key = goal_key_xy(eff)
            recent = getattr(self, "_habitat_recent_goals", None)
            if recent is not None:
                recent.append(key)
                del recent[:-8]
            blocked = getattr(self, "_habitat_blocked_goals", None)
            if blocked is not None:
                blocked.add(key)
                blocked.add(goal_key_xy(goal_xy))

    def navigate_to_target_pose(
        self,
        target_pose: torch.Tensor | np.ndarray | list | tuple | None,
        start_pose: torch.Tensor | np.ndarray | list | tuple | None,
        target_theta: float | None = None,
        *,
        target_obs_id: int | None = None,
    ):
        if target_pose is None:
            nav_res = NavAttemptResult(
                success=False,
                finished=False,
                dist_m=0.0,
                method="none",
                note="no_target",
                target_obs_id=target_obs_id,
            )
            self._last_nav_attempt = nav_res
            return NavOutcome.NO_TARGET

        res = None
        original_target_pose = target_pose
        tp = target_pose.detach().cpu().numpy() if hasattr(target_pose, "detach") else target_pose
        tp_arr = np.asarray(tp, dtype=np.float64).reshape(-1)
        goal_xy = np.array([float(tp_arr[0]), float(tp_arr[1])], dtype=np.float64)

        if habitat_perfect_nav_enabled(self.parameters) and is_habitat_robot_client(self.robot):
            nav_res = habitat_navmesh_navigate(
                self.robot,
                goal_xy,
                target_theta=target_theta,
            )
            nav_res.target_obs_id = target_obs_id
            self._last_nav_attempt = nav_res
            if nav_res.finished or nav_res.success:
                logger.info(f"EQA habitat navmesh: {nav_res.note} dist={nav_res.dist_m:.2f}m")
            else:
                logger.info(f"EQA habitat navmesh failed: {nav_res.note} (dist={nav_res.dist_m:.2f}m)")
            self._last_nav_plan = {
                "mode": "navigation",
                "localize_source": "eqa_target",
                "goal_xyt": [float(goal_xy[0]), float(goal_xy[1]), float(target_theta or 0.0)],
                "method": "habitat_navmesh",
                "note": nav_res.note,
            }
            self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
            # Stuck / noop / no-progress: same blocked-goal memory as voxel timeout abort
            # so uncover explore / multi-goal A* skip this frontier.
            stuck = (
                str(nav_res.note or "").startswith("already_at_goal")
                or (not nav_res.finished and float(nav_res.dist_m) < 0.08)
                or (not nav_res.success and float(nav_res.dist_m) < 0.12)
            )
            if stuck:
                self._mark_nav_goal_blocked(reason=f"habitat_navmesh_{nav_res.note or 'stuck'}")
            if nav_res.finished:
                return NavOutcome.REACHED
            if nav_res.success or float(getattr(nav_res, "dist_m", 0.0) or 0.0) >= 0.12:
                return NavOutcome.PROGRESS
            return NavOutcome.STUCK

        target_pose = self.space.sample_navigation(start_pose, self.planner, original_target_pose)

        # A* planning
        if target_pose is not None:
            res = self.planner.plan(start_pose, target_pose)

        # Parse A* results into traj
        if res is not None and res.success:
            waypoints = [pt.state for pt in res.trajectory]
        elif res is not None:
            waypoints = None
            logger.warning("navigate_to_target_pose planner failure: %s", res.reason)
        else:
            waypoints = None

        if waypoints is not None:
            self.rerun_visualizer.log_custom_pointcloud(
                "world/target_pose",
                [original_target_pose[0], original_target_pose[1], 1.5],
                torch.Tensor([1, 0, 0]),
                0.1,
            )

        finished = False
        n_planned = 0
        truncated = False
        full_traj_for_viz = None
        if waypoints is not None:
            n_planned = len(waypoints)
            truncated = len(waypoints) > 8
            full_traj_for_viz = self.planner.clean_path_for_xy(
                list(waypoints), start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0
            )
            if truncated:
                waypoints = waypoints[:8]
            traj = self.planner.clean_path_for_xy(
                waypoints, start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0
            )
            finished = not truncated
            if finished and target_theta is not None:
                traj[-1][2] = target_theta
            traj, reject_reason, min_clr = self._filter_unsafe_nav_traj(traj, start_xyt=start_pose)
            if reject_reason is not None or not traj:
                logger.warning(f"navigate_to_target_pose rejected after safety filter: {reject_reason}")
                self._last_nav_plan = {
                    "mode": "navigation",
                    "localize_source": "eqa_target",
                    "goal_xyt": list(np.asarray(target_pose, dtype=np.float64).reshape(-1)[:3])
                    if target_pose is not None
                    else [float(goal_xy[0]), float(goal_xy[1]), 0.0],
                    "object_xyz": list(np.asarray(original_target_pose, dtype=np.float64).reshape(-1)[:3]),
                    "n_planned": n_planned,
                    "chunked": truncated,
                    "min_clearance_m": min_clr,
                    "outcome": reject_reason or "rejected_low_clearance",
                }
                reason = reject_reason or "rejected_low_clearance"
                self._mark_nav_goal_blocked(reason=reason)
                nav_res = NavAttemptResult(
                    success=False,
                    finished=False,
                    dist_m=0.0,
                    method="voxel_astar",
                    note=reason,
                    target_obs_id=target_obs_id,
                )
                self._last_nav_attempt = nav_res
                self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
                return NavOutcome.SAFETY_REJECTED
            logger.info(
                "navigate_to_target_pose: %d exec / %d planned waypoints (finished=%s)",
                len(traj),
                n_planned,
                finished,
            )
        else:
            traj = None

        before_xy = np.asarray(start_pose, dtype=np.float64).reshape(-1)[:2].copy()
        # draw traj on rerun and execute it
        if traj is not None:
            log_plan = getattr(self.rerun_visualizer, "log_nav_plan", None)
            if callable(log_plan):
                self._last_nav_plan = log_plan(
                    traj,
                    full_traj=full_traj_for_viz,
                    start_xyt=start_pose,
                    goal_xyt=target_pose,
                    object_xyz=original_target_pose,
                    mode="navigation",
                    localize_source="eqa_target",
                    n_planned=n_planned or None,
                    chunked=truncated,
                )
                self._record_nav_plan_fields(traj=list(traj))
            else:
                origins = []
                vectors = []
                for idx in range(len(traj) - 1):
                    origins.append([traj[idx][0], traj[idx][1], 1.5])
                    vectors.append([traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0])
                self.rerun_visualizer.log_arrow3D("world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1)
                self.rerun_visualizer.log_custom_pointcloud(
                    "world/robot_start_pose",
                    [start_pose[0], start_pose[1], 1.5],
                    torch.Tensor([0, 0, 1]),
                    0.1,
                )

            from emet.controller.nav_confirm import confirm_navigation_plan

            if not confirm_navigation_plan(
                self,
                traj,
                meta=getattr(self, "_last_nav_plan", None) or {},
                object_xyz=original_target_pose,
            ):
                self._record_nav_plan_fields(outcome="user_cancelled", confirmed=False)
                nav_res = NavAttemptResult(
                    success=False,
                    finished=False,
                    dist_m=0.0,
                    method="voxel_astar",
                    note="user_rejected_plan",
                    target_obs_id=target_obs_id,
                )
                self._last_nav_attempt = nav_res
                self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
                return NavOutcome.USER_CANCELLED

            nav_timeout = self._find_phase_nav_timeout()
            exec_ok = self.robot.execute_trajectory(
                traj,
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
                per_waypoint_timeout=nav_timeout,
                final_timeout=max(nav_timeout, 30.0),
                blocking=True,
                world_frame=True,
            )
            if exec_ok is False:
                self._record_nav_plan_fields(outcome="aborted_waypoint_timeout")
                self._mark_nav_goal_blocked(reason="aborted_waypoint_timeout")
                nav_res = NavAttemptResult(
                    success=False,
                    finished=False,
                    dist_m=0.0,
                    method="voxel_astar",
                    note="aborted_waypoint_timeout",
                    target_obs_id=target_obs_id,
                )
                self._last_nav_attempt = nav_res
                self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
                return NavOutcome.ABORTED_TIMEOUT
            after_xy = np.asarray(self.robot.get_base_pose(), dtype=np.float64).reshape(-1)[:2]
            dist_m = float(np.hypot(after_xy[0] - before_xy[0], after_xy[1] - before_xy[1]))
            note = "ok" if finished else f"moved_{dist_m:.2f}m"
            nav_res = NavAttemptResult(
                success=dist_m >= 0.12 or finished,
                finished=finished,
                dist_m=dist_m,
                method="voxel_astar",
                note=note,
                target_obs_id=target_obs_id,
            )
        else:
            note = res.reason if res is not None else "sample_nav_failed"
            logger.info(f"EQA voxel nav failed: {note}")
            self._last_nav_plan = {
                "mode": "navigation",
                "localize_source": "eqa_target",
                "goal_xyt": [float(goal_xy[0]), float(goal_xy[1]), 0.0],
                "object_xyz": list(np.asarray(original_target_pose, dtype=np.float64).reshape(-1)[:3]),
                "outcome": str(note),
            }
            self._mark_nav_goal_blocked(reason=str(note))
            nav_res = NavAttemptResult(
                success=False,
                finished=False,
                dist_m=0.0,
                method="voxel_astar",
                note=note,
                target_obs_id=target_obs_id,
            )

        self._last_nav_attempt = nav_res
        self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
        if finished:
            return NavOutcome.REACHED
        if nav_res.success or float(getattr(nav_res, "dist_m", 0.0) or 0.0) >= 0.12:
            return NavOutcome.PROGRESS
        if nav_res.note == "sample_nav_failed" or str(nav_res.note or "").startswith("no_target"):
            return NavOutcome.NO_TARGET
        if nav_res.note and str(nav_res.note).startswith("plan"):
            return NavOutcome.PLAN_FAILED
        return NavOutcome.STUCK


# Backward compatibility alias
RobotAgent = DynamemController
