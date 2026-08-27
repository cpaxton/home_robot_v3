# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Encoder, detector, voxel map, navigation space, and ``get_voxel_map``."""

from __future__ import annotations

import os

from emet.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
from emet.mapping.voxel import (
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from emet.mapping.voxel.voxel import _instance_memory_kwargs_from_params
from emet.memory.graph_eqa import GraphEQAMemory, SensorGraphBuilder
from emet.memory.graph_eqa.ingest.instance_observations import DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M
from emet.motion.algo.a_star import AStar, default_min_clearance_m
from emet.perception.detection.owl import OwlPerception
from emet.perception.encoders.clip_encoder import MaskClipEncoder
from emet.utils.logger import Logger
from emet.utils.vram_debug import print_vram_snapshot

logger = Logger(__name__)


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
    # Instance graph is opt-in via use_instance_memory / use_instance_graph — honor it even on
    # manipulation_only nav stacks (Habitat HM-EQA, OVMM find, …). Only skip the detector when
    # the stack is genuinely label-only (no instances, no voxel list_objects EQA).
    if not self._use_instance_memory and (self.manipulation_only or self.eqa):
        # No object detection for pure manipulation or voxel EQA without instances.
        # GraphEQA sets eqa=False and use_instance_memory=True so YoloE runs for instance labels.
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
    self.planner.debug_start_escape = True
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
        from emet.memory.graph_eqa.graph_object_fusion.attach import attach_graph_object_fusion

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


def get_voxel_map(self):
    """Return the voxel map used for occupancy and open-vocab localize."""
    return self.voxel_map
