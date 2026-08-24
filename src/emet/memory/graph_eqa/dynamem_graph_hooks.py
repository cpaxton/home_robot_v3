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
from emet.memory.graph_eqa.graph_label_filter import (
    filter_graph_labels,
    is_graph_label_allowed,
    resolve_graph_scene_profile,
)
from emet.memory.graph_eqa.graph_memory import labels_are_semantic_graph_hypothesis
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate
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


def _default_bounds_3d_for_xyz(xyz: np.ndarray, *, half_extent_m: float = 0.18) -> dict[str, list[float]]:
    """Small axis-aligned box around a label-only detection (VLM path) for IoU dedup."""
    c = np.asarray(xyz, dtype=np.float64).reshape(3)
    h = float(half_extent_m)
    mn = c - h
    mx = c + h
    return {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "center": c.tolist(),
        "size": (mx - mn).tolist(),
    }


def _detection_to_candidate(d: dict[str, Any]) -> GraphDetectionCandidate:
    emb = d.get("embedding")
    if emb is not None:
        emb = np.asarray(emb, dtype=np.float32)
    xyz = np.asarray(d["xyz"], dtype=np.float64)
    bounds = d.get("bounds_3d")
    if bounds is None and xyz is not None:
        bounds = _default_bounds_3d_for_xyz(xyz)
    return GraphDetectionCandidate(
        label=str(d.get("label_short", d.get("label", "object"))),
        xyz=xyz,
        bbox_xyxy=tuple(d["bbox_xyxy"]) if d.get("bbox_xyxy") is not None else None,
        bounds_3d=bounds,
        embedding=emb,
        identity_key=d.get("identity_key"),
        countable_instance=bool(d.get("countable_instance", d.get("instance_id") is not None)),
    )


def _instance_item_fields(item: Any) -> tuple[Any, Any, Any, Any]:
    """Normalize legacy tuples and identity-carrying instance records."""
    if hasattr(item, "label") and hasattr(item, "xyz"):
        return (
            item.label,
            item.xyz,
            getattr(item, "bbox_xyxy", None),
            getattr(item, "identity_key", None),
        )
    if isinstance(item, dict):
        return (
            item.get("label", "object"),
            item["xyz"],
            item.get("bbox_xyxy"),
            item.get("identity_key"),
        )
    if len(item) >= 4:
        return item[0], item[1], item[2], item[3]
    if len(item) >= 3:
        return item[0], item[1], item[2], None
    return item[0], item[1], None, None


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
    if hasattr(graph_memory, "set_capture_context"):
        session_id = ""
        session_fn = getattr(robot, "get_emet_session", None)
        if callable(session_fn):
            try:
                session = session_fn() or {}
                if isinstance(session, dict):
                    session_id = str(
                        session.get("session_id") or session.get("run_id") or session.get("environment_name") or ""
                    )
            except Exception:
                session_id = ""
        graph_memory.set_capture_context(
            camera_pose_world=obs.camera_pose,
            base_pose_world=viewer_xyz,
            session_id=session_id or None,
        )
    scene_profile = resolve_graph_scene_profile(
        robot=robot,
        parameters=getattr(graph_memory, "parameters", None),
    )

    vm = voxel_map
    instance_items: list[Any] = []
    raw_dets: list[dict[str, Any]] = []
    visible_labels: list[str] = []
    if use_instance_graph and getattr(vm, "observations", None) and len(vm.observations) > 0:
        frame = vm.observations[-1]
        raw_dets = frame_instances_to_detections(
            frame,
            min_depth=float(vm.min_depth),
            max_depth=float(vm.max_depth),
            detection_model=detection_model,
        )
        raw_dets = [
            d
            for d in raw_dets
            if is_graph_label_allowed(
                str(d.get("label_short", d.get("label", ""))),
                scene_profile=scene_profile,
            )
        ]
        instance_items = [
            (
                d["label_short"],
                np.asarray(d["xyz"], dtype=np.float64),
                tuple(d["bbox_xyxy"]),
                None,
            )
            for d in raw_dets
        ]
        visible_labels.extend(str(item[0]) for item in instance_items)
        if (
            not instance_items
            and getattr(frame, "instance", None) is not None
            and getattr(vm, "use_instance_memory", False)
        ):
            instance_items = [
                it
                for it in instance_items_from_instance_memory(vm, detection_model)
                if is_graph_label_allowed(str(it[0]), scene_profile=scene_profile)
            ]

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
                    scene_profile=scene_profile,
                )

    labeler = getattr(robot, "hm3d_semantic_labeler", None)
    if labeler is not None and obs.semantic is not None:
        from emet.habitat.hm3d_semantics import hm3d_instance_items_from_obs

        items = hm3d_instance_items_from_obs(labeler, obs, with_instance_ids=True)
        if items:
            visible_labels.extend(str(_instance_item_fields(item)[0]) for item in items)
            fusion_enabled = graph_object_fusion is not None and getattr(
                getattr(graph_object_fusion, "config", None),
                "enabled",
                False,
            )
            if fusion_enabled:
                crop_rgb = np.asarray(rgb)
                for item in items:
                    label, xyz_item, bbox, identity_key = _instance_item_fields(item)
                    cand = _detection_to_candidate(
                        {
                            "label": str(label),
                            "xyz": np.asarray(xyz_item, dtype=np.float64),
                            "bbox_xyxy": bbox,
                            "identity_key": identity_key,
                            "countable_instance": True,
                        }
                    )
                    graph_object_fusion.apply_detection(
                        graph_memory,
                        crop_rgb,
                        cand,
                        viewer_xyz=viewer_xyz,
                    )
            else:
                apply_instance_items_to_graph(
                    graph_memory,
                    rgb,
                    items,
                    dedup_skips=dedup_skips or (lambda _l, _x: False),
                )
            if hasattr(graph_memory, "observe_visible_labels"):
                graph_memory.observe_visible_labels(
                    visible_labels,
                    viewer_xyz,
                    step=frame_step,
                )
            return
        hm3d_labels = labeler.labels_from_frame(obs.semantic, obs.depth)
        visible_labels.extend(str(label) for label in hm3d_labels)
        if hm3d_labels and sensor_builder is not None:
            xyz = sensor_builder.world_xyz_for_observation(obs)
            graph_memory.add_observation(rgb, xyz, hm3d_labels)
            if hasattr(graph_memory, "observe_visible_labels"):
                graph_memory.observe_visible_labels(
                    visible_labels,
                    viewer_xyz,
                    step=frame_step,
                )
            return
        if hm3d_labels:
            if hasattr(graph_memory, "observe_visible_labels"):
                graph_memory.observe_visible_labels(
                    visible_labels,
                    viewer_xyz,
                    step=frame_step,
                )
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

    labels = filter_graph_labels(labels, scene_profile=scene_profile)
    visible_labels.extend(str(label) for label in labels)

    fusion_enabled = graph_object_fusion is not None and getattr(
        getattr(graph_object_fusion, "config", None),
        "enabled",
        False,
    )

    if labels_are_semantic_graph_hypothesis(labels):
        if fusion_enabled:
            crop_rgb = np.asarray(rgb)
            xyz_a = np.asarray(xyz, dtype=np.float64)
            for label in labels:
                cand = _detection_to_candidate({"label": str(label), "xyz": xyz_a})
                graph_object_fusion.apply_detection(
                    graph_memory,
                    crop_rgb,
                    cand,
                    viewer_xyz=viewer_xyz,
                )
        else:
            for label in labels:
                if dedup_skips and dedup_skips(label, xyz):
                    continue
                graph_memory.add_observation(rgb, xyz, [label], description=desc, viewer_xyz=viewer_xyz)
    elif not instance_items:
        graph_memory.record_navigation_sample(rgb, xyz, base_xyz=viewer_xyz)

    if fusion_enabled and graph_object_fusion is not None:
        graph_object_fusion.consolidate_high_iou_nodes(graph_memory)
    if hasattr(graph_memory, "observe_visible_labels"):
        graph_memory.observe_visible_labels(
            visible_labels,
            viewer_xyz,
            step=frame_step,
        )


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
