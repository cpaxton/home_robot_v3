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

import os
import time
from collections import Counter
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
from emet.motion.algo.a_star import AStar
from emet.perception.depth import create_da3_estimator_from_parameters, resolve_depth_map
from emet.perception.depth.lingbot_estimator import LingBotDepthEstimator, create_lingbot_estimator_from_parameters
from emet.perception.depth.da3_estimator import apply_da3_sky_row_mask, sensor_depth_usable
from emet.perception.depth.lingbot_estimator import LingBotDepthEstimator, create_lingbot_estimator_from_parameters
from emet.perception.detection.owl import OwlPerception
from emet.perception.detection.yoloe import YoloEPerception

# from emet.perception.encoders.mobile_clip_encoder import MaskMobileClipEncoder
from emet.perception.encoders.clip_encoder import MaskClipEncoder
from emet.perception.encoders.siglip_encoder import MaskSiglipEncoder
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

        self.tts = PiperTextToSpeech()

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
        self.manip_wrapper = ManipulationWrapper(self.robot, stretch_gripper_max=stretch_gripper_max, end_link=end_link)
        self.robot.move_to_nav_posture()

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
        if not keywords:
            node = frontier_nodes[0]
            return np.array([float(node.xyz[0]), float(node.xyz[1]), 1.0], dtype=float)
        best_node = None
        best_score = -1.0
        for node in frontier_nodes:
            labels = [str(lbl).strip().lower() for lbl in (node.labels or []) if str(lbl).strip()]
            score = keyword_overlap_score(labels, keywords)
            if score > best_score:
                best_score = score
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

            self.encoder = get_shared_mask_siglip_encoder(
                version="so400m", device=self.device, feature_matching_threshold=0.14
            )

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
            self.detection_model = YoloEPerception(
                confidence_threshold=det_conf,
                device=self.device,
                size="l",
            )
            semantic_memory_resolution = 0.05
            image_shape = (360, 270)
        else:
            self.detection_model = OwlPerception(
                version="owlv2-L-p14-ensemble", device=self.device, confidence_threshold=0.15
            )
            semantic_memory_resolution = 0.05
            image_shape = (480, 360)

        _eqa = parameters.get("eqa", {}) or {}
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
            detection=self.detection_model,
            encoder=self.encoder,
            image_shape=image_shape,
            log=self.log,
            mllm=self.mllm,
            run_eqa=self.eqa,
            device=self.device,
            eqa_backend=_eqa.get("backend", "qwen_vl"),
            eqa_vl_model_size=_eqa.get("vl_model_size", "3B"),
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
        print_vram_snapshot("after_create_obstacle_map_sparse_voxel_map")
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            rotation_step_size=parameters.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=parameters.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=parameters.get("motion_planner/frontier/dilate_obstacle_size", 0),
        )
        self.planner = AStar(self.space)

        cfg = self.embodied_agent
        if cfg.open_vocab_scene_graph.enabled and not self.manipulation_only:
            from emet.mapping.scene_graph.processor import SceneGraphProcessor

            sg_name = cfg.open_vocab_scene_graph.config_name
            if self.cpu_only and sg_name == "default_scene_graph":
                sg_name = "cpu_scene_graph"
            dev = cfg.open_vocab_scene_graph.device
            if dev is None:
                dev = "cpu" if self.cpu_only else None
            self._open_vocab_sg_processor = SceneGraphProcessor(config_name=sg_name, device=dev)
            self.voxel_map.set_scene_graph_processor(self._open_vocab_sg_processor)

        if cfg.graph_eqa_memory.enabled and not self.manipulation_only:
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
                parameters if isinstance(parameters, dict) else None,
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
            depth = np.asarray(self._da3_last_depth, dtype=np.float32, copy=True)
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
            depth = np.asarray(self._lingbot_last_depth, dtype=np.float32, copy=True)
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
            sky = float(self.parameters.get("da3_ignore_sky_fraction_top", 0.0) or 0.0)
            if sky > 0.0:
                depth = apply_da3_sky_row_mask(np.asarray(depth, dtype=np.float32), sky)
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
        if self.use_scene_graph and self.voxel_map.use_instance_memory:
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
            self.sensor_builder is not None
            or self._graph_eqa_use_instance_graph
            or has_hm3d_labeler
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

        # Visualize open-vocab scene graph if attached
        ovsg = self.voxel_map.get_scene_graph()
        if ovsg is not None and ovsg.num_objects > 0:
            self.rerun_visualizer.update_open_vocab_scene_graph(ovsg)

        self._rerun_refresh_monologue_panel()

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

    def describe_head_camera_scene_text(self) -> str:
        """Summarize the current head RGB using the controller's detector (YoloE or OWL).

        Used by the embodied agent ``describe_scene`` tool so the model can answer
        "what do you see" without sending an image.
        """
        if self.robot is None or not hasattr(self.robot, "get_observation"):
            return "No robot view available."
        obs = self.robot.get_observation()
        if obs is None or getattr(obs, "rgb", None) is None:
            return "No current image."
        rgb = np.asarray(obs.rgb)
        if rgb.dtype != np.uint8:
            if rgb.size and float(np.nanmax(rgb)) <= 1.0 + 1e-6:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return "Head camera image has an unexpected shape."

        if env_agent_camera_debug():
            from emet.agent.camera_debug import print_camera_frame_diagnostics

            print_camera_frame_diagnostics("describe_scene (head RGB, detector input)", rgb, force=True)

        depth = getattr(obs, "depth", None)
        if depth is not None:
            depth = np.asarray(depth)

        dm = self.detection_model
        if env_agent_model_debug():
            if dm is None:
                print(
                    "[model debug] describe_scene: no detection_model on controller (YoloE/OWL labels unavailable)",
                    flush=True,
                )
            else:
                print(
                    f"[model debug] describe_scene: detector={type(dm).__name__} on head RGB "
                    "(separate from the chat LLM; tool result is from this detector, not from the LLM's weights)",
                    flush=True,
                )

        if dm is None:
            return (
                "I have a live camera frame but object detection is disabled for this session "
                "(e.g. manipulation-only or voxel EQA without instances), so I cannot name objects. "
                "Use send_image to show the view, or explore and query_memory if you need the map."
            )

        try:
            if isinstance(dm, YoloEPerception):
                return self._describe_scene_yoloe(rgb, depth, dm)
            if isinstance(dm, OwlPerception):
                thr = float(dm.confidence_threshold) if dm.confidence_threshold is not None else 0.2
                thr = max(0.12, min(thr, 0.35))
                return self._describe_scene_owl(rgb, dm, thr)
        except Exception as e:
            return (
                f"I could not run detection on the camera frame ({type(e).__name__}: {e}). "
                "Try send_image if you need the raw view."
            )

        return "Unknown detector type for scene description; try send_image for a picture."

    def _describe_scene_yoloe(self, rgb: np.ndarray, depth: np.ndarray | None, dm: YoloEPerception) -> str:
        _sem, _inst, task = dm.predict(rgb, depth=depth, draw_instance_predictions=False)
        ic = task.get("instance_classes")
        if ic is None or len(ic) == 0:
            return (
                "From my head camera: nothing is segmented above the current detection threshold. "
                "The view may be empty, dark, or objects may not match the detector vocabulary. "
                "send_image can still show the raw frame."
            )
        class_list = dm.class_list
        names: list[str] = []
        for idx in np.atleast_1d(np.asarray(ic)).astype(int).ravel():
            if 0 <= int(idx) < len(class_list):
                names.append(class_list[int(idx)])
        if not names:
            return "From my head camera: detections did not map to class names. send_image can show the raw view."
        counts = Counter(names)
        parts = [f"{n} (×{c})" if c > 1 else n for n, c in counts.most_common()]
        summary = ", ".join(parts)
        return f"From my head camera I can make out: {summary}."

    def _describe_scene_owl(self, rgb: np.ndarray, dm: OwlPerception, confidence_threshold: float) -> str:
        texts = list(_DESCRIBE_SCENE_OWL_QUERIES)
        res = dm.predict(rgb, texts, confidence_threshold=confidence_threshold)
        labels = res["labels"]
        if labels.numel() == 0:
            return (
                "From my head camera: nothing matched the open-vocabulary checks above the confidence cutoff. "
                "You can still use send_image to see the frame."
            )
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

    def look_around(self):
        """
        Let the robot look around to check its surroudings.
        Rotating the robot head to compensate for the narrow field of view of realsense head camera
        """
        logger.info("Look around: sweeping head")
        for pan in [0.6, -0.2, -1.0, -1.8]:
            tilt = -0.6
            self.robot.head_to(pan, tilt, blocking=True)
            self.update()

    def _find_phase_nav_timeout(self, default: float = 10.0) -> float:
        raw = self.parameters.get("find_phase_nav_step_timeout_s")
        if raw is None:
            return default
        return float(raw)

    def rotate_in_place(self):
        logger.info("Rotate in place: scanning environment")
        nav_timeout = self._find_phase_nav_timeout()
        if self.save_rerun:
            if not os.path.exists(self.log):
                os.makedirs(self.log)
            rr.save(self.log + "/" + "data_" + str(self.rerun_iter) + ".rrd")
        self.robot.move_to_nav_posture()
        self.robot.look_front(blocking=True, timeout=nav_timeout)
        time.sleep(DYNAMEM_HEAD_SETTLE_S)
        wait_obs = getattr(self.robot, "wait_for_obs", None)
        if callable(wait_obs):
            wait_obs(timeout=nav_timeout)
        logger.info("rotate_in_place: 8× relative +45° yaw (no XY translation)")
        for _step_i in range(8):
            self.robot.move_base_to(
                [0.0, 0.0, np.pi / 4.0],
                relative=True,
                blocking=True,
                timeout=nav_timeout,
            )
            if not self._realtime_updates:
                self.update()
        self.rerun_iter += 1
        self._maybe_emit_navgrid_ascii(context="rotate_in_place")

    def _maybe_emit_navgrid_ascii(self, *, context: str = "") -> None:
        from emet.mapping.debug_navgrid_ascii import (
            build_navgrid_from_voxel_map,
            maybe_print_navgrid_ascii,
            navgrid_context_allowed,
        )

        if not navgrid_context_allowed(context):
            return
        try:
            xyt = self.robot.get_base_pose()
            robot_xy = (float(xyt[0]), float(xyt[1]))
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
            logger.info("Navigation plan OK; executing trajectory")
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
                    self.robot.execute_trajectory(
                        res[:-2],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                        blocking=True,
                        world_frame=True,
                    )

                self.robot.look_front()
                self.update()
                return True, res[-1]
            # The robot has not reached the object. Next it should look around and continue navigation
            else:
                self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    blocking=True,
                    world_frame=True,
                )
                self.robot.look_front()
                self.update()
                return False, None
        else:
            logger.warning("No plan from process_text; try again.")
            return None, None

    def run_exploration(self):
        """
        Go through exploration when the robot has not received any text query from the user.
        We use the voxel_grid map created by our collector to sample free space, and then use A* planner to get there.
        """

        # "" means the robot has not received any text query from the user and should conduct exploration just to better know the environment
        status, _ = self.execute_action("")
        if status is None:
            logger.warning("Exploration failed (no valid plan or frontier).")
            return False
        self._maybe_emit_navgrid_ascii(context="explore")
        return True

    def process_text(self, text, start_pose):
        """
        Process the text query and return the trajectory for the robot to follow.
        """

        logger.debug("process_text: %r", text)

        self.rerun_visualizer.clear_identity("world/object")
        self.rerun_visualizer.clear_identity("world/xyt_goal")
        self.rerun_visualizer.clear_identity("world/robot_start_pose")
        self.rerun_visualizer.clear_identity("world/direction")
        self.rerun_visualizer.clear_identity("robot_monologue")
        self.rerun_visualizer.clear_identity("/observation_similar_to_text")

        debug_text = ""
        mode = "navigation"
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
                debug_text += "## Last visual grounding results looks fine so directly use it.\n"
            elif hasattr(self.encoder, "feature_matching_threshold") and _finite_xyz_traj_target(traj_target_point):
                # Short queries ("red object") often fail SigLIP neighborhood re-check; still navigate to last grounding.
                localized_point = traj_target_point
                debug_text += "## Reusing saved trajectory target; semantic re-check was not decisive.\n"

        logger.debug("Target verification done (localized_point=%s)", localized_point is not None)

        if text is not None and text != "" and localized_point is None:
            graph_point = self._localize_point_from_graph_memory(text)
            if graph_point is not None:
                localized_point = graph_point
                debug_text += "## Localized target from graph memory.\n"
                mode = "navigation"
                logger.debug("Localized target from graph for query %r", text)

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
                    if loc_debug:
                        debug_text += str(loc_debug)
                    logger.debug("Localized target from voxel map for query %r", text)
                except Exception as exc:
                    logger.debug("voxel localize_text failed for %r: %s", text, exc)

        # Do Frontier based exploration (optionally biased by the active EQA question).
        if text is None or text == "" or localized_point is None:
            debug_text += "## Navigation fails, so robot starts exploring environments.\n"
            frontier_text = self._exploration_text(text)
            graph_frontier = self._best_frontier_point_from_graph(frontier_text)
            if graph_frontier is not None:
                localized_point = graph_frontier
                debug_text += "## Selected frontier target from graph memory.\n"
                mode = "exploration"
            else:
                localized_point = self.space.sample_frontier(self.planner, start_pose, frontier_text)
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
        self.rerun_visualizer.log_custom_pointcloud(
            "world/object",
            [ox, oy, oz],
            torch.Tensor([1, 0, 0]),
            0.12,
        )

        point = self.space.sample_navigation(start_pose, self.planner, localized_point)

        logger.debug("Navigation endpoint: %s", point)

        waypoints = None

        if point is None:
            res = None
            logger.warning("No navigation endpoint sampled (planner may fail).")
        else:
            res = self.planner.plan(start_pose, point)

        if res is not None and res.success:
            waypoints = [pt.state for pt in res.trajectory]
        elif res is not None:
            waypoints = None
            logger.warning("Planner failure: %s", res.reason)

        if point is not None:
            self.rerun_visualizer.update_nav_goal(np.asarray(point, dtype=np.float64))

        # If we are navigating to some object of interest, send (x, y, z) of
        # the object so that we can make sure the robot looks at the object after navigation
        traj = []
        if waypoints is not None:
            finished = len(waypoints) <= 8 and mode == "navigation"
            if finished:
                self.space.traj = None
            else:
                self.space.traj = waypoints[8:] + [[np.nan, np.nan, np.nan], localized_point]
            if not finished:
                waypoints = waypoints[:8]
            traj = self.planner.clean_path_for_xy(waypoints)
            if finished:
                traj.append([np.nan, np.nan, np.nan])
                if isinstance(localized_point, torch.Tensor):
                    localized_point = localized_point.tolist()
                traj.append(localized_point)
            logger.debug("Planned trajectory (%d waypoints): %s", len(traj), traj)

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
        debug_text = "# Robot's monologue: \n" + debug_text
        self._rerun_monologue_base = debug_text
        self._rerun_refresh_monologue_panel()

        if traj is not None:
            origins = []
            vectors = []
            for idx in range(len(traj)):
                if idx != len(traj) - 1:
                    origins.append([traj[idx][0], traj[idx][1], 1.5])
                    vectors.append([traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0])
            self.rerun_visualizer.log_arrow3D("world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1)
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 1.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

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

        if skip_confirmation or input("Do you want to do this manipulation? Y or N ") != "N":
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
            answer, discord_text, relevant_images, confidence = self.run_eqa_one_iter(question)
            if confidence:
                self.robot.say("The answer to " + question + " is " + answer)
                break

            if stall_patience > 0 and self.graph_memory is not None:
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

        if not self._realtime_updates:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        try:
            (
                reasoning,
                answer,
                confidence,
                confidence_reasoning,
                target_point,
                relevant_images,
            ) = self.voxel_map.query_answer(question, self._planning_base_xyt(self.robot.get_base_pose()), self.planner)
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
            if finished:
                break

        return answer, discord_text, relevant_images, confidence

    def navigate_to_target_pose(
        self,
        target_pose: torch.Tensor | np.ndarray | list | tuple | None,
        start_pose: torch.Tensor | np.ndarray | list | tuple | None,
        target_theta: float | None = None,
    ):
        res = None
        original_target_pose = target_pose
        if target_pose is not None:
            # target_pose originally represents the place where the object of interest is.
            # This line finds the pose where the robot should stop
            target_pose = self.space.sample_navigation(start_pose, self.planner, target_pose)

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

        finished = True
        if waypoints is not None:
            if not len(waypoints) <= 8:
                waypoints = waypoints[:8]
                finished = False
            traj = self.planner.clean_path_for_xy(waypoints)
            if finished and target_theta is not None:
                traj[-1][2] = target_theta
            logger.debug("navigate_to_target_pose trajectory (%d pts): %s", len(traj), traj)
        else:
            traj = None

        # draw traj on rerun and execute it
        if traj is not None:
            origins = []
            vectors = []
            for idx in range(len(traj)):
                if idx != len(traj) - 1:
                    origins.append([traj[idx][0], traj[idx][1], 1.5])
                    vectors.append([traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0])
            self.rerun_visualizer.log_arrow3D("world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1)
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 1.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

            self.robot.execute_trajectory(
                traj,
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
                blocking=True,
                world_frame=True,
            )

        return finished


# Backward compatibility alias
RobotAgent = DynamemController
