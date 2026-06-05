# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Dynagraph: DynaMem voxel navigation + GraphEQA graph memory with merge and staleness."""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from emet.controller.controller_graph_eqa import GraphEQAController
from emet.memory.graph_eqa.sim_ground_truth_graph import (
    read_sim_object_placements,
    upsert_graph_memory_from_placements,
)


class DynagraphController(GraphEQAController):
    """
    Same stack as GraphEQA (voxel map + graph memory + EQA), with optional
    spatial merge and staleness pruning on ``GraphEQAMemory`` (see ``dynagraph_*`` keys in config).
    """

    def __init__(
        self,
        *args,
        ground_truth_mode: bool = False,
        visualize_ground_truth: bool = False,
        **kwargs,
    ):
        self.ground_truth_mode = ground_truth_mode
        # Separate GT Rerun layer only when comparing sensor graph vs sim reference.
        self.visualize_ground_truth = visualize_ground_truth and not ground_truth_mode
        self._gt_graph_loaded = False
        super().__init__(*args, **kwargs)
        self.setup_custom_blueprint()
        self._sync_ground_truth_from_session()
        if self.graph_memory is not None and getattr(self.rerun_visualizer, "enabled", True):
            if self.ground_truth_mode:
                self.rerun_visualizer.clear_identity("world/dynagraph/ground_truth/nodes")
            self.rerun_visualizer.log_dynagraph_state(self.graph_memory)

    def setup_custom_blueprint(self) -> None:
        if getattr(self.rerun_visualizer, "enabled", True) is False:
            return
        gt_column = None
        if self.visualize_ground_truth:
            gt_column = rrb.Vertical(
                rrb.Spatial3DView(name="Sim GT (reference)", origin="world/dynagraph/ground_truth"),
                rrb.TextDocumentView(name="Sim GT", origin="world/dynagraph/ground_truth/summary"),
            )
        graph_label = "Graph (ground truth)" if self.ground_truth_mode else "Dynagraph 3D"
        summary_label = "Graph (GT)" if self.ground_truth_mode else "Dynagraph graph"
        dynagraph_column = rrb.Vertical(
            rrb.Spatial3DView(name=graph_label, origin="world/dynagraph"),
            rrb.TextDocumentView(name=summary_label, origin="world/dynagraph/summary"),
        )
        right_columns: list = [dynagraph_column]
        if gt_column is not None:
            right_columns.append(gt_column)
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="robot_monologue"),
                rrb.Spatial2DView(name="relevant image", origin="/observation_similar_to_text"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="world/head_camera"),
                rrb.Spatial2DView(name="ee_rgb", origin="world/ee_camera"),
                rrb.Spatial2DView(name="map_topdown", origin="world/map_snapshot/topdown"),
            ),
            rrb.Vertical(*right_columns),
            column_shares=[3, 1, 1, 1],
        )
        collapse = getattr(self.rerun_visualizer, "collapse_panels", True)
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(main, rrb.TimePanel(state=True)), collapse_panels=collapse))

    def _sync_ground_truth_from_session(self) -> int:
        """Upsert GT graph nodes and log Rerun layers; returns number of GT bodies in session."""
        session = self.robot.get_emet_session()
        placements = read_sim_object_placements(session)
        if not placements:
            return 0
        if self.ground_truth_mode and self.graph_memory is not None and not self._gt_graph_loaded:
            if self.obs_count > 0:
                self.graph_memory.set_graph_timestep(self.obs_count)
            rgb = getattr(self, "_gt_rgb_cache", None)
            if rgb is None:
                obs = self.robot.get_observation()
                rgb = np.asarray(obs.rgb, dtype=np.uint8)
                self._gt_rgb_cache = rgb
            upsert_graph_memory_from_placements(
                self.graph_memory,
                rgb,
                placements,
            )
            self._gt_graph_loaded = True
        if self.visualize_ground_truth and getattr(self.rerun_visualizer, "enabled", True):
            self.rerun_visualizer.log_dynagraph_ground_truth(
                placements,
                graph_memory=self.graph_memory,
            )
        return len(placements)

    def refresh_ground_truth(self) -> int:
        """Populate GT graph + Rerun from ``emet_session``; returns GT body count (0 if missing)."""
        n = self._sync_ground_truth_from_session()
        if self.graph_memory is not None and getattr(self.rerun_visualizer, "enabled", True):
            self.rerun_visualizer.log_dynagraph_state(
                self.graph_memory,
                ground_truth_mode=self.ground_truth_mode,
            )
        return n

    def update(self) -> None:
        if self.graph_memory is not None and (self.ground_truth_mode or self.visualize_ground_truth):
            self._sync_ground_truth_from_session()
        super().update()
        if self.graph_memory is None:
            return
        self.graph_memory.maintain(self.obs_count)
        self.rerun_visualizer.log_dynagraph_state(
            self.graph_memory,
            ground_truth_mode=self.ground_truth_mode,
        )
        self._rerun_refresh_monologue_panel()
        if self.obs_count % 8 == 0:
            self._maybe_emit_navgrid_ascii(context="dynagraph")
