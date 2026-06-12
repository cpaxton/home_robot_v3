# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""Shared observation → GraphEQAMemory update (DynaMem agent + GraphEQAController)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from emet.memory.graph_eqa.calibration_export import CalibrationFrameWriter, detections_to_json_rows
from emet.memory.graph_eqa.graph_memory import labels_are_semantic_graph_hypothesis
from emet.memory.graph_eqa.graph_observation_pipeline import apply_instance_items_to_graph
from emet.memory.graph_eqa.instance_observations import (
    frame_instances_to_detections,
    frame_rgb_hwc_uint8,
    instance_items_from_instance_memory,
)
from emet.memory.graph_eqa.sensor_graph_builder import SensorGraphBuilder, short_labels_from_voxel_descriptions
from emet.memory.graph_eqa.viewer_frame import viewer_xyz_world_from_observation


def _nav_origin_xyt(obs: Any) -> list[float] | None:
    origin = getattr(obs, "navigation_origin_xyt", None)
    if origin is None:
        return None
    arr = np.asarray(origin, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _detection_to_candidate(d: dict[str, Any]) -> Any:
    from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate

    emb = d.get("embedding")
    if emb is not None:
        emb = np.asarray(emb, dtype=np.float32)
    return GraphDetectionCandidate(
        label=str(d.get("label_short", d.get("label", "object"))),
        xyz=np.asarray(d["xyz"], dtype=np.float64),
        bbox_xyxy=tuple(d["bbox_xyxy"]) if d.get("bbox_xyxy") is not None else None,
        bounds_3d=d.get("bounds_3d"),
        embedding=emb,
    )


def update_graph_memory_from_dynamem_observation(
    *,
    graph_memory: Any,
    robot: Any,
    voxel_map: Any,
    detection_model: Any,
    sensor_builder: SensorGraphBuilder,
    use_instance_graph: bool,
    use_sensor_perception: bool,
    dedup_skips: Callable[[str, np.ndarray], bool] | None,
    obs: Any,
    frame_step: int | None = None,
    graph_object_fusion: Any | None = None,
    calibration_writer: CalibrationFrameWriter | None = None,
    encoder: Any | None = None,
) -> None:
    """Append one observation to ``graph_memory`` (same logic as ``GraphEQAController.update`` tail).

    When ``use_instance_graph`` is true, YoloE / instance-mask detections are added as graph nodes
    first. When ``use_sensor_perception`` is also true, the sensor VLM may add further nodes for
    objects the detector missed (deduped by ``dedup_skips``).
    """
    rgb = obs.rgb
    if obs.camera_pose is None:
        return

    if frame_step is not None and hasattr(graph_memory, "set_graph_timestep"):
        graph_memory.set_graph_timestep(int(frame_step))

    viewer_xyz = viewer_xyz_world_from_observation(obs, robot=robot)

    vm = voxel_map
    instance_items: list[tuple[str, np.ndarray, tuple[int, int, int, int]]] = []
    raw_dets: list[dict[str, Any]] = []
    if use_instance_graph and getattr(vm, "observations", None) and len(vm.observations) > 0:
        frame = vm.observations[-1]
        raw_dets = frame_instances_to_detections(
            frame,
            min_depth=float(vm.min_depth),
            max_depth=float(vm.max_depth),
            detection_model=detection_model,
        )
        instance_items = [
            (
                d["label_short"],
                np.asarray(d["xyz"], dtype=np.float64),
                tuple(d["bbox_xyxy"]),
            )
            for d in raw_dets
        ]
        if not instance_items and getattr(frame, "instance", None) is not None and getattr(
            vm, "use_instance_memory", False
        ):
            instance_items = instance_items_from_instance_memory(vm, detection_model)

        if calibration_writer is not None and raw_dets:
            calibration_writer.append(
                step=int(frame_step or 0),
                detections=detections_to_json_rows(raw_dets),
                navigation_origin_xyt=_nav_origin_xyt(obs),
            )

        if instance_items or raw_dets:
            frame_rgb = frame_rgb_hwc_uint8(frame)
            crop_rgb = frame_rgb if frame_rgb is not None else np.asarray(rgb)

            cfg = getattr(graph_object_fusion, "config", None) if graph_object_fusion is not None else None
            use_fusion = cfg is not None and getattr(cfg, "enabled", False)

            if use_fusion and raw_dets:
                for d in raw_dets:
                    cand = _detection_to_candidate(d)
                    graph_object_fusion.apply_detection(
                        graph_memory,
                        crop_rgb,
                        cand,
                        viewer_xyz=viewer_xyz,
                    )
            elif instance_items:
                apply_instance_items_to_graph(
                    graph_memory,
                    crop_rgb,
                    instance_items,
                    dedup_skips=dedup_skips or (lambda _l, _x: False),
                    viewer_xyz=viewer_xyz,
                )

    labeler = getattr(robot, "hm3d_semantic_labeler", None)
    if labeler is not None and obs.semantic is not None:
        from emet.habitat.hm3d_semantics import hm3d_instance_items_from_obs

        items = hm3d_instance_items_from_obs(labeler, obs)
        if items:
            apply_instance_items_to_graph(
                graph_memory,
                rgb,
                items,
                dedup_skips=dedup_skips or (lambda _l, _x: False),
            )
            return
        hm3d_labels = labeler.labels_from_frame(obs.semantic, obs.depth)
        if hm3d_labels and sensor_builder is not None:
            xyz = sensor_builder.world_xyz_for_observation(obs)
            graph_memory.add_observation(rgb, xyz, hm3d_labels)
            return
        if hm3d_labels:
            return

    voxel_labels = None
    if getattr(vm, "image_descriptions", None) and len(vm.image_descriptions) > 0:
        voxel_labels = vm.image_descriptions[-1][0]

    if use_sensor_perception:
        labels, desc = sensor_builder.labels_and_description_from_observation(obs, voxel_labels=voxel_labels)
        xyz = sensor_builder.world_xyz_for_observation(obs)
    else:
        labels = short_labels_from_voxel_descriptions(voxel_labels) if voxel_labels else ["object"]
        desc = None
        xyz = np.array(obs.camera_pose[:3, 3], dtype=float)

    fusion_cfg = getattr(getattr(graph_object_fusion, "config", None), "enabled", False)
    if fusion_cfg:
        dedup = None
    else:
        dedup = dedup_skips

    if labels_are_semantic_graph_hypothesis(labels):
        for label in labels:
            if dedup and dedup(label, xyz):
                continue
            graph_memory.add_observation(rgb, xyz, [label], description=desc, viewer_xyz=viewer_xyz)
    elif not instance_items:
        graph_memory.record_navigation_sample(rgb, xyz, base_xyz=viewer_xyz)


def sync_graph_frontier_nodes(
    *,
    graph_memory: Any,
    voxel_map: Any,
    planner: Any,
    base_xyt: Any,
    question: str | None = None,
) -> int:
    """Mirror voxel frontiers into graph frontier nodes (optional question-guided hints)."""
    if graph_memory is None or not getattr(graph_memory, "frontier_nodes_enabled", False):
        return 0
    if not hasattr(graph_memory, "sync_frontier_nodes"):
        return 0
    keywords = None
    if question:
        from emet.memory.graph_eqa.frontier_nodes import exploration_keywords_from_text

        keywords = exploration_keywords_from_text(question)
    return int(
        graph_memory.sync_frontier_nodes(
            voxel_map,
            planner,
            base_xyt,
            question_keywords=keywords,
        )
    )


def _base_xyz_from_robot(robot: Any) -> np.ndarray | None:
    try:
        bp = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
        if bp.size >= 2:
            bz = float(bp[2]) if bp.size >= 3 else 0.0
            return np.array([float(bp[0]), float(bp[1]), bz], dtype=np.float64)
    except Exception:
        pass
    return None


def update_graph_memory_ground_truth_from_observation(
    *,
    graph_memory: Any,
    robot: Any,
    obs: Any,
    frame_step: int | None = None,
) -> None:
    """
    GT mode: record each camera viewpoint without creating new graph entity nodes.

    Sim ``sim_object_placements`` remain the authoritative object list; instance
    detections attach to those nodes separately in ``DynagraphController.update``.
    """
    if obs.camera_pose is None:
        return
    if frame_step is not None and hasattr(graph_memory, "set_graph_timestep"):
        graph_memory.set_graph_timestep(int(frame_step))
    xyz = np.array(obs.camera_pose[:3, 3], dtype=float)
    graph_memory.record_navigation_sample(
        obs.rgb,
        xyz,
        base_xyz=_base_xyz_from_robot(robot),
        link_viewpoint_node=False,
    )
