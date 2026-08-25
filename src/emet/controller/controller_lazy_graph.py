# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""LazyGraph: DynaMem find + graph commits on nav arrival (no streaming YoloE graph)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.controller.controller_dynagraph import DynagraphController
from emet.memory.graph_eqa.lazy_graph_commit import commit_graph_from_arrival_obs
from emet.utils.logger import Logger

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

    def execute_action(self, text: str) -> tuple[bool | None, np.ndarray | None]:
        status, object_xyz = super().execute_action(text)
        if status is True and self.graph_memory is not None:
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
                if getattr(self.rerun_visualizer, "enabled", True):
                    self.rerun_visualizer.log_dynagraph_state(
                        self.graph_memory,
                        ground_truth_mode=self.ground_truth_mode,
                    )
            except Exception as exc:
                logger.warning(f"lazy_graph arrival commit failed: {exc}")
        return status, object_xyz
