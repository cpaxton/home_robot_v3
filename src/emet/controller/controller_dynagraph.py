# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Dynagraph: DynaMem voxel navigation + GraphEQA graph memory with merge and staleness."""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from emet.controller.controller_graph_eqa import GraphEQAController


class DynagraphController(GraphEQAController):
    """
    Same stack as GraphEQA (voxel map + graph memory + EQA), with optional
    spatial merge and staleness pruning on ``GraphEQAMemory`` (see ``dynagraph_*`` keys in config).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_custom_blueprint()
        if self.graph_memory is not None and getattr(self.rerun_visualizer, "enabled", True):
            self.rerun_visualizer.log_dynagraph_state(self.graph_memory)

    def setup_custom_blueprint(self) -> None:
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
            column_shares=[3, 1, 1],
        )
        collapse = getattr(self.rerun_visualizer, "collapse_panels", True)
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(main, rrb.TimePanel(state=True)), collapse_panels=collapse))

    def update(self) -> None:
        super().update()
        if self.graph_memory is None:
            return
        self.graph_memory.maintain(self.obs_count)
        self.rerun_visualizer.log_dynagraph_state(self.graph_memory)
        self._rerun_refresh_monologue_panel()
        if self.obs_count % 8 == 0:
            self._maybe_emit_navgrid_ascii(context="dynagraph")
