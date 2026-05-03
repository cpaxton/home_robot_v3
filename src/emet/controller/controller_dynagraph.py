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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

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

    def setup_custom_blueprint(self) -> None:
        if getattr(self.rerun_visualizer, "enabled", True) is False:
            return
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
            rrb.Vertical(
                rrb.Spatial3DView(name="Dynagraph 3D", origin="world/dynagraph"),
                rrb.TextDocumentView(name="Dynagraph graph", origin="world/dynagraph/summary"),
            ),
            column_shares=[3, 1, 1, 1],
        )
        collapse = getattr(self.rerun_visualizer, "collapse_panels", True)
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(main, rrb.TimePanel(state=True)), collapse_panels=collapse))

    def update(self) -> None:
        super().update()
        if self.graph_memory is None:
            return
        self.graph_memory.maintain(self.obs_count)
        self.rerun_visualizer.log_dynagraph_state(self.graph_memory)
