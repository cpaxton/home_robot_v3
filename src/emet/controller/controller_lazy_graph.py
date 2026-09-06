# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""LazyGraph: DynaMem find + graph commits on nav arrival (no streaming YoloE graph)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from emet.controller.controller_dynagraph import DynagraphController
from emet.memory.graph_eqa.lazy_graph_commit import commit_graph_from_arrival_obs
from emet.utils.logger import Logger
from emet.visualization.null_visualizer import visualizer_is_enabled

logger = Logger(__name__)


class LazyGraphController(DynagraphController):
    """
    Sibling to Dynagraph: same voxel map, merge/staleness, frontier sync, and EQA loop.

    Differences:
    - No per-frame instance/VLM streaming into ``GraphEQAMemory`` (see ``_lazy_graph_mode``).
    - Qwen label extract + ``add_observation`` only after successful nav arrival.
    - YoloE may still feed voxel find; detector class names never author graph labels.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("use_instance_graph", False)
        kwargs.setdefault("use_sensor_perception", True)
        super().__init__(*args, **kwargs)
        self._lazy_graph_mode = True
        self.query_driven_memory = bool(self.parameters.get("query_driven_memory", False))
        if self.query_driven_memory:
            from emet.memory.graph_eqa.agentic.config import agentic_verify_enabled

            if not agentic_verify_enabled(self):
                raise ValueError("query_driven_memory requires eqa.agentic_verify=true")

    @property
    def query_candidates(self):
        return self.graph_memory.query_candidates

    def propose_query_candidate(self, phrase, xyz, stats):
        """Preserve retrieval provenance without creating an object node."""
        source = stats.get("source_obs_id")
        if source is None:
            return None
        try:
            return self.query_candidates.propose(phrase, int(source), 0, xyz, retrieval_score=stats.get("max_cosine"))
        except ValueError as exc:
            logger.warning(f"Query candidate rejected: {exc}")
            return None

    def retrieve_query_candidate(self, phrase: str):
        """Keep low-confidence voxel matches in the search tier, not object memory."""
        query = " ".join(phrase.lower().split())
        rejected = {
            r.source_obs_id
            for r in self.query_candidates.records.values()
            if r.query == query and r.rejected_revision is not None
        }
        return self.voxel_map.retrieve_text_candidate(phrase, excluded_obs_ids=rejected)

    def ground_query_candidate(self, handle, *, after_observation: int):
        """Promote only from admitted, object-specific geometry in a new frame."""
        from emet.memory.graph_eqa.graph_object_fusion.attach import fusion_config_from_sources
        from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate, GraphObjectFusion
        from emet.memory.graph_eqa.ingest.instance_observations import (
            filter_detections_for_graph_admission,
            frame_instances_to_detections,
            frame_rgb_hwc_uint8,
            frame_world_xyz_hw3,
        )

        record = self.query_candidates.records[handle]
        self._grounded_query_target = None
        vm = self.voxel_map
        # Failed reacquisition must revoke even a previously grounded reference.
        record.grounded_revision = None
        record.invalidation_reason = "reacquisition pending"
        if len(vm.observations) <= max(after_observation, record.source_obs_id):
            return {"ok": False, "reason": "fresh observation required"}
        frame = vm.observations[-1]
        rgb = frame_rgb_hwc_uint8(frame)
        if rgb is None or frame.depth is None:
            return {"ok": False, "reason": "RGB-D required"}
        fusion = getattr(self, "_graph_object_fusion", None)
        if fusion is None:
            fusion = GraphObjectFusion(fusion_config_from_sources(parameters=self.parameters))
        if not fusion.config.use_instance_nodes or not fusion.config.enabled:
            return {"ok": False, "reason": "instance admission/fusion disabled"}
        # Lazy mapping intentionally does not run streaming instance detection.
        # Detect only this arrival view, using the same RGB/depth/world geometry.
        if getattr(frame, "instance", None) is None:
            detector = self.detection_model
            if detector is None:
                from emet.perception.detection.yoloe import get_shared_yoloe_perception

                detector = get_shared_yoloe_perception(
                    confidence_threshold=self.parameters.get("detection", {}).get("confidence_threshold", 0.05),
                    device=self.device,
                    size="l",
                )
            depth = frame.depth
            if hasattr(depth, "detach"):
                depth = depth.detach().cpu().numpy()
            _, masks, metadata = detector.predict(rgb, depth=depth, draw_instance_predictions=False)
            frame = SimpleNamespace(
                rgb=rgb,
                depth=depth,
                full_world_xyz=frame.full_world_xyz,
                instance=masks,
                instance_classes=metadata.get("instance_classes", []),
                instance_scores=metadata.get("instance_scores", []),
            )
        else:
            detector = self.detection_model
        detections = frame_instances_to_detections(
            frame, min_depth=vm.min_depth, max_depth=vm.max_depth, detection_model=detector
        )
        admitted, _ = filter_detections_for_graph_admission(detections, config=fusion.config)
        from emet.memory.query_grounding import cache_grounding_record, select_query_detections

        matching_ids, verification = select_query_detections(
            record.query,
            record.target_description,
            detections,
            rgb,
            client=getattr(self.graph_memory, "eqa_client", None),
        )
        import os
        from dataclasses import asdict
        from pathlib import Path

        cache_dir = (self.parameters.get("query_memory", {}) or {}).get("grounding_cache_dir")
        # The existing explicit episode artifact directory opts into diagnostic
        # capture. No new runtime-policy environment flag is needed.
        if not cache_dir and os.environ.get("EMET_EQA_EPISODE_DIR"):
            cache_dir = Path(os.environ["EMET_EQA_EPISODE_DIR"]) / "grounding"
        cache_path = cache_grounding_record(
            cache_dir,
            query=record.query,
            revision=len(vm.observations),
            source_obs_id=record.source_obs_id,
            detections=detections,
            matching_ids=matching_ids,
            verification=verification,
            rgb=rgb,
            depth=frame.depth,
            masks=frame.instance,
            metadata={
                "target_description": record.target_description,
                "retrieval_score": record.retrieval_score,
                "admission_config": asdict(fusion.config),
                "min_depth": vm.min_depth,
                "max_depth": vm.max_depth,
            },
        )
        matches = [d for d in detections if d["instance_id"] in matching_ids]
        if len(matches) != 1 or not any(d is matches[0] for d in admitted):
            self.query_candidates.reject(
                handle, observation_revision=len(vm.observations), reason="target absent or ambiguous"
            )
            return {
                "ok": False,
                "reason": "target absent or ambiguous",
                "cache_path": cache_path,
                "n_detections": len(detections),
                "matching_ids": matching_ids,
                "verification_source": verification["source"],
            }
        det = matches[0]
        candidate = GraphDetectionCandidate(
            label=det["label_short"],
            xyz=np.asarray(det["xyz"]),
            bbox_xyxy=det["bbox_xyxy"],
            bounds_3d=det["bounds_3d"],
            detection_score=det["detection_score"],
            mask_point_count=det["mask_point_count"],
            countable_instance=True,
        )
        obs_id = fusion.apply_detection(self.graph_memory, rgb, candidate)
        if obs_id is None:
            return {"ok": False, "reason": "instance budget exhausted"}
        nodes = [n for n in self.graph_memory.get_nodes() if n.obs_id == obs_id and n.countable_instance]
        if len(nodes) != 1:
            return {"ok": False, "reason": "instance identity unresolved"}
        # Graph node indices are renumbered by maintenance. The object's stable
        # observation ID is the query identity; never persist a node-list index.
        self.query_candidates.ground(handle, instance_id=obs_id, observation_revision=len(vm.observations))
        from emet.memory.grounded_target import GroundedTarget

        world = frame_world_xyz_hw3(frame).detach().cpu().numpy()
        mask = frame.instance
        depth = frame.depth
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()
        valid = (mask == det["instance_id"]) & (depth > vm.min_depth) & (depth < vm.max_depth)
        valid &= np.isfinite(world).all(axis=-1) & np.isfinite(depth)
        self._grounded_query_target = GroundedTarget(handle, obs_id, len(vm.observations), world[valid])
        return {"ok": True, "instance_id": obs_id, "obs_id": obs_id, "xyz": self._grounded_query_target.xyz.tolist()}

    def prepare_query_target(self, query: str):
        """Reacquire a unique query reference immediately before manipulation."""
        query = " ".join(query.lower().split())
        records = [r for r in self.query_candidates.records.values() if r.query == query]
        if not records:
            xyz, stats = self.retrieve_query_candidate(query)
            candidate = self.propose_query_candidate(query, xyz, stats) if xyz is not None else None
            records = [candidate] if candidate is not None else []
        if len(records) != 1:
            raise ValueError("Manipulation requires a unique query candidate")
        before = len(self.voxel_map.observations)
        self.update(full_perception=True)
        result = self.ground_query_candidate(records[0].handle, after_observation=before)
        if not result["ok"]:
            raise ValueError(result["reason"])
        records[0].require_grounding(len(self.voxel_map.observations))
        return self._grounded_query_target

    def execute_action(self, text: str) -> tuple[bool | None, np.ndarray | None]:
        status, object_xyz = super().execute_action(text)
        if status is True and self.graph_memory is not None and not self.query_driven_memory:
            obs = self.robot.get_observation()
            plan = getattr(self, "_last_nav_plan", None) or {}
            try:
                commit_graph_from_arrival_obs(
                    graph_memory=self.graph_memory,
                    robot=self.robot,
                    sensor_builder=self.sensor_builder,
                    obs=obs,
                    query_text=text or None,
                    localize_source=str(plan.get("localize_source") or ""),
                    object_xyz=object_xyz,
                    frame_step=self.obs_count,
                    parameters=self.parameters,
                )
                self.graph_memory.maintain(self.obs_count)
                if visualizer_is_enabled(self.rerun_visualizer):
                    self.rerun_visualizer.log_dynagraph_state(
                        self.graph_memory,
                        ground_truth_mode=self.ground_truth_mode,
                    )
            except Exception as exc:
                logger.warning(f"lazy_graph arrival commit failed: {exc}")
        return status, object_xyz

    def _commit_lazy_graph_arrival(
        self,
        *,
        action_obs_id: int | None = None,
        target_point: Any | None = None,
    ) -> None:
        """Qwen label-extract commit when the HM-EQA loop arrives at a nav target.

        The classic EQA loop navigates via ``run_eqa_one_iter`` →
        ``navigate_to_target_pose`` (never ``execute_action``), so without this hook
        a lazy-graph HM-EQA run would commit nothing and stay graphless. Fires on the
        ``finished.finished`` arrival point with the current observation.
        """
        if self.graph_memory is None or not self._lazy_graph_mode or self.query_driven_memory:
            return
        obs = self.robot.get_observation()
        if obs is None:
            return
        xyz: np.ndarray | None = None
        if target_point is not None:
            xyz = np.asarray(target_point, dtype=float).reshape(-1)[:3]
        try:
            commit_graph_from_arrival_obs(
                graph_memory=self.graph_memory,
                robot=self.robot,
                sensor_builder=self.sensor_builder,
                obs=obs,
                object_xyz=xyz,
                frame_step=self.obs_count,
                parameters=self.parameters,
            )
            self.graph_memory.maintain(self.obs_count)
        except Exception as exc:
            logger.warning(f"lazy_graph EQA arrival commit failed: {exc}")
