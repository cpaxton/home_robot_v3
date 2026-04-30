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
from emet.memory.graph_eqa.instance_observations import frame_instances_to_labels_xyz
from emet.memory.graph_eqa.sensor_graph_builder import SensorGraphBuilder, short_labels_from_voxel_descriptions


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
) -> None:
    """Append one observation to ``graph_memory`` (same logic as ``GraphEQAController.update`` tail)."""
    rgb = obs.rgb
    if obs.camera_pose is None:
        return

    vm = voxel_map
    if use_instance_graph and getattr(vm, "observations", None) and len(vm.observations) > 0:
        frame = vm.observations[-1]
        items = frame_instances_to_labels_xyz(
            frame,
            min_depth=float(vm.min_depth),
            max_depth=float(vm.max_depth),
            detection_model=detection_model,
        )
        if items:
            apply_instance_items_to_graph(
                graph_memory,
                rgb,
                items,
                dedup_skips=dedup_skips or (lambda _l, _x: False),
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

    base_xyz: np.ndarray | None = None
    try:
        bp = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
        if bp.size >= 2:
            bz = float(bp[2]) if bp.size >= 3 else 0.0
            base_xyz = np.array([float(bp[0]), float(bp[1]), bz], dtype=np.float64)
    except Exception:
        pass

    if labels_are_semantic_graph_hypothesis(labels):
        graph_memory.add_observation(rgb, xyz, labels, description=desc)
    else:
        graph_memory.record_navigation_sample(rgb, xyz, base_xyz=base_xyz)
