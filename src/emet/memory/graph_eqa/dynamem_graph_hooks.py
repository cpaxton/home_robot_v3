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
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Shared observation → GraphEQAMemory update (DynaMem agent + GraphEQAController)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_memory import labels_are_semantic_graph_hypothesis
from emet.memory.graph_eqa.graph_observation_pipeline import apply_instance_items_to_graph
from emet.memory.graph_eqa.instance_observations import (
    frame_instances_to_labels_xyz,
    frame_rgb_hwc_uint8,
    instance_items_from_instance_memory,
)
from emet.memory.graph_eqa.sensor_graph_builder import SensorGraphBuilder, short_labels_from_voxel_descriptions
from emet.memory.graph_eqa.viewer_frame import viewer_xyz_world_from_observation


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
    if use_instance_graph and getattr(vm, "observations", None) and len(vm.observations) > 0:
        frame = vm.observations[-1]
        instance_items = frame_instances_to_labels_xyz(
            frame,
            min_depth=float(vm.min_depth),
            max_depth=float(vm.max_depth),
            detection_model=detection_model,
        )
        if not instance_items and getattr(frame, "instance", None) is not None and getattr(
            vm, "use_instance_memory", False
        ):
            instance_items = instance_items_from_instance_memory(vm, detection_model)
        if instance_items:
            frame_rgb = frame_rgb_hwc_uint8(frame)
            crop_rgb = frame_rgb if frame_rgb is not None else np.asarray(rgb)
            apply_instance_items_to_graph(
                graph_memory,
                crop_rgb,
                instance_items,
                dedup_skips=dedup_skips or (lambda _l, _x: False),
                viewer_xyz=viewer_xyz,
            )

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

    if labels_are_semantic_graph_hypothesis(labels):
        for label in labels:
            if dedup_skips and dedup_skips(label, xyz):
                continue
            graph_memory.add_observation(rgb, xyz, [label], description=desc, viewer_xyz=viewer_xyz)
    elif not instance_items:
        graph_memory.record_navigation_sample(rgb, xyz, base_xyz=viewer_xyz)
