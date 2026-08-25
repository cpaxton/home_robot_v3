# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""LazyGraph: Qwen object commits on nav arrival; viewpoints during passive mapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_label_filter import filter_graph_labels, resolve_graph_scene_profile
from emet.memory.graph_eqa.viewer_frame import viewer_xyz_world_from_observation
from emet.utils.logger import Logger

logger = Logger(__name__)

LABEL_SOURCE_QWEN_ARRIVAL = "qwen_arrival"


def record_lazy_graph_viewpoint(
    *,
    graph_memory: Any,
    robot: Any,
    obs: Any,
    frame_step: int | None = None,
) -> None:
    """Lightweight explored-station stamp (no detector / streaming VLM labels)."""
    if graph_memory is None or obs is None or obs.camera_pose is None:
        return
    if frame_step is not None and hasattr(graph_memory, "set_graph_timestep"):
        graph_memory.set_graph_timestep(int(frame_step))
    viewer_xyz = viewer_xyz_world_from_observation(obs, robot=robot)
    xyz = np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)
    if hasattr(graph_memory, "record_navigation_sample"):
        graph_memory.record_navigation_sample(
            obs.rgb,
            xyz,
            base_xyz=viewer_xyz,
            link_viewpoint_node=True,
        )


def commit_graph_from_arrival_obs(
    *,
    graph_memory: Any,
    robot: Any,
    sensor_builder: Any,
    obs: Any,
    query_text: str | None = None,
    localize_source: str = "",
    object_xyz: np.ndarray | None = None,
    frame_step: int | None = None,
    parameters: Any | None = None,
) -> int | None:
    """Run Qwen label extract on an arrival frame and add/merge graph object nodes.

    Detector / YoloE strings are never used as graph labels (``voxel_labels=None``).

    Returns:
        obs_id from ``add_observation``, or None if commit was skipped.
    """
    if graph_memory is None or obs is None or obs.camera_pose is None:
        return None
    if sensor_builder is None:
        logger.warning("lazy_graph commit skipped: no sensor_builder (Qwen extract unavailable)")
        return None

    if frame_step is not None and hasattr(graph_memory, "set_graph_timestep"):
        graph_memory.set_graph_timestep(int(frame_step))

    viewer_xyz = viewer_xyz_world_from_observation(obs, robot=robot)
    scene_profile = resolve_graph_scene_profile(
        robot=robot,
        parameters=parameters or getattr(graph_memory, "parameters", None),
    )

    labels, desc = sensor_builder.labels_and_description_from_observation(obs, voxel_labels=None)
    labels = filter_graph_labels(labels, scene_profile=scene_profile)
    if not labels:
        labels = ["object"]

    if object_xyz is not None:
        xyz = np.asarray(object_xyz, dtype=np.float64).reshape(-1)[:3]
    else:
        xyz = sensor_builder.world_xyz_for_observation(obs)

    obs_id = int(
        graph_memory.add_observation(
            obs.rgb,
            xyz,
            labels,
            description=desc,
            viewer_xyz=viewer_xyz,
        )
    )
    logger.info(
        f"lazy_graph commit obs_id={obs_id} labels={labels} "
        f"localize_source={localize_source!r} query={(query_text or '')[:80]!r}"
    )
    return obs_id
