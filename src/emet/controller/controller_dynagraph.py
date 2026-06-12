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
        self._skip_graph_perception_updates = ground_truth_mode
        # Force a (shared) SigLIP encoder so the voxel map stores VL features for open-vocab
        # grounding, even in Habitat's manipulation_only nav-only setup. Must be set BEFORE
        # super().__init__ builds the voxel map.
        params = kwargs.get("parameters")
        if params is None and len(args) >= 2:
            params = args[1]
        if params is not None:
            try:
                params["force_eqa_siglip_encoder"] = True
            except Exception:
                pass
        super().__init__(*args, **kwargs)
        # Dynagraph contribution over the GraphEQA baseline: SigLIP-grounded CONFIRMED_MEMORY
        # (open-vocab grounding independent of caption labels) + frontier-coverage override
        # that explores unobserved question objects instead of revisiting seen views.
        self._eqa_explore_when_uncovered = True
        if self.graph_memory is not None:
            self.graph_memory.memory_summary_enabled = True
            self.graph_memory.set_text_grounder(self._siglip_text_match)
            self.graph_memory.set_obs_id_grounder(self._siglip_obs_id_for_text)
        if ground_truth_mode and self.graph_memory is not None:
            # Keep full rotate/explore viewpoint history (voxel frames also logged on export).
            self.graph_memory.set_navigation_samples_max(max(self.graph_memory.navigation_samples_max, 8192))
        self.setup_custom_blueprint()
        self._sync_ground_truth_from_session()
        if self.graph_memory is not None and getattr(self.rerun_visualizer, "enabled", True):
            if self.ground_truth_mode:
                self.rerun_visualizer.clear_identity("world/dynagraph/ground_truth/nodes")
            self.rerun_visualizer.log_dynagraph_state(self.graph_memory)

    def _associate_instances_to_ground_truth(self) -> None:
        """Attach YoloE/instance detections to nearest sim GT nodes (RGB + optional det label)."""
        if not self.ground_truth_mode or self.graph_memory is None:
            return
        vm = getattr(self, "voxel_map", None)
        if vm is None or not getattr(vm, "use_instance_memory", False):
            return
        obs_list = getattr(vm, "observations", None)
        if not obs_list:
            return
        from emet.memory.graph_eqa.sim_ground_truth_graph import associate_instance_detections_to_ground_truth

        associate_instance_detections_to_ground_truth(
            self.graph_memory,
            obs_list[-1],
            rgb=np.asarray(obs_list[-1].rgb, dtype=np.uint8),
            voxel_map=vm,
            detection_model=getattr(self, "detection_model", None),
        )

    def _associate_gt_to_frame_instances(self) -> None:
        """Project GT AABBs to image and match YoloE instance masks (reverse association)."""
        if not self.ground_truth_mode:
            return
        vm = getattr(self, "voxel_map", None)
        if vm is None:
            return
        obs_list = getattr(vm, "observations", None)
        if not obs_list:
            return
        placements = read_sim_object_placements(self.robot.get_emet_session())
        if not placements:
            return
        from emet.memory.graph_eqa.sim_ground_truth_graph import (
            associate_ground_truth_to_frame_instances,
            associate_ground_truth_to_voxel_observation,
        )

        frame = obs_list[-1]
        assocs = associate_ground_truth_to_frame_instances(placements, frame)
        if not hasattr(frame, "info") or frame.info is None:
            frame.info = {}
        if assocs:
            frame.info["gt_associations"] = assocs
        voxel_hits = associate_ground_truth_to_voxel_observation(placements, frame)
        if voxel_hits:
            frame.info["gt_voxel_hits"] = voxel_hits

    def setup_custom_blueprint(self) -> None:
        if getattr(self.rerun_visualizer, "enabled", True) is False:
            return
        from emet.visualization.rerun import spatial3d_view_world

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
            rrb.Vertical(*right_columns),
            column_shares=[3, 1, 1, 1] if gt_column is not None else [3, 1, 1],
        )
        collapse = getattr(self.rerun_visualizer, "collapse_panels", True)
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(main, rrb.TimePanel(state=True)), collapse_panels=collapse))

    def _sync_ground_truth_from_session(self) -> int:
        """Upsert GT graph nodes and log Rerun layers; returns number of GT bodies in session."""
        # Sim-only: Habitat's robot client has no emet session (no GT placements).
        get_sess = getattr(self.robot, "get_emet_session", None)
        if get_sess is None:
            return 0
        session = get_sess()
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
        if self.ground_truth_mode:
            self._associate_instances_to_ground_truth()
            self._associate_gt_to_frame_instances()
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
