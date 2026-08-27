# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Depth resolve, voxel update, scene-graph / graph ingest, Rerun status."""

from __future__ import annotations

import os
import time

import numpy as np

from emet.controller.dynamem.constants import (
    _env_truthy,
)
from emet.mapping.scene_graph import SceneGraph
from emet.perception.depth import create_da3_estimator_from_parameters, resolve_depth_map
from emet.perception.depth.da3_estimator import apply_da3_sky_row_mask, apply_depth_speckle_filter, sensor_depth_usable
from emet.perception.depth.lingbot_estimator import LingBotDepthEstimator, create_lingbot_estimator_from_parameters
from emet.utils.geometry import nav_xyt_to_world_xyt
from emet.utils.logger import Logger

logger = Logger(__name__)


def _graph_dedup_skips(self, label: str, xyz: np.ndarray) -> bool:
    """Skip adding a graph node if we already have the same label near this XY (GraphEQA v1 merge)."""
    if self.graph_memory is None or self._graph_dedup_xy_m <= 0:
        return False
    lb = label.strip().lower()
    for n in self.graph_memory.get_nodes():
        if not n.labels:
            continue
        nl = (n.labels[0] or "").strip().lower()
        if nl != lb:
            continue
        if float(np.linalg.norm(n.xyz[:2] - xyz[:2])) < self._graph_dedup_xy_m:
            return True
    return False


def _lazy_da3_estimator(self):
    if self._depth_source not in ("da3", "auto"):
        return None
    if self._da3_estimator is None:
        self._da3_estimator = create_da3_estimator_from_parameters(self.parameters, device=self.device)
    return self._da3_estimator


def _lazy_lingbot_estimator(self) -> LingBotDepthEstimator | None:
    if self._depth_source != "lingbot":
        return None
    if self._lingbot_estimator is None:
        self._lingbot_estimator = LingBotDepthEstimator(
            create_lingbot_estimator_from_parameters(self.parameters),
            use_lingbot_pose=self._lingbot_use_pose,
        )
    return self._lingbot_estimator


def _resolve_depth_map(
    self,
    rgb: np.ndarray,
    sensor_depth: np.ndarray | None,
    camera_K: np.ndarray | None,
    camera_pose: np.ndarray | None,
    rgb_right: np.ndarray | None = None,
    camera_K_right: np.ndarray | None = None,
    camera_pose_right: np.ndarray | None = None,
) -> np.ndarray | None:
    """Pick sensor depth, DA3, or auto per ``depth_source``.

    When ``debug_perfect_sensor_depth`` (YAML) or ``EMET_DYNAMEM_PERFECT_DEPTH=1`` is set, always prefer
    simulator / hardware **sensor** depth from the observation whenever it is present, so DA3 noise
    cannot mask extrinsic or frame bugs during calibration runs.

    Sets ``self._depth_map_from_da3_infer`` so :meth:`update` only applies ``da3_ignore_sky_fraction_top``
    to DA3-produced maps (never to raw sensor depth).
    """
    self._depth_map_from_da3_infer = False
    mode = str(self._depth_source).lower()
    if getattr(self, "_debug_perfect_sensor_depth", False):
        if sensor_depth is not None:
            sd = np.asarray(sensor_depth, dtype=np.float32)
            if sd.size > 0 and bool(np.any(np.isfinite(sd) & (sd > 1e-6))):
                logger.info("debug_perfect_sensor_depth: using observation sensor depth (skipping DA3).")
                return sd
        logger.warning(
            "debug_perfect_sensor_depth enabled but observation has no usable sensor depth; "
            "falling back to depth_source=%r.",
            mode,
        )
    if mode == "sensor":
        return sensor_depth
    if mode == "lingbot":
        est_lb = self._lazy_lingbot_estimator()
        if est_lb is None:
            raise RuntimeError("depth_source=lingbot but LingBot estimator failed to initialize.")
        self._depth_map_from_da3_infer = True
        depth_lb = est_lb.infer(
            rgb,
            camera_K=camera_K,
            camera_pose=camera_pose,
            force=(self.obs_count == 1),
        )
        if self._lingbot_use_pose and est_lb.last_camera_pose is not None:
            self._lingbot_last_pose = est_lb.last_camera_pose
        return depth_lb
    # Auto: prefer sensor without constructing DA3 (heavy); matches resolve_depth_map logic.
    if mode == "auto" and sensor_depth_usable(sensor_depth):
        return np.asarray(sensor_depth, dtype=np.float32)
    est = self._lazy_da3_estimator()
    self._depth_map_from_da3_infer = True
    return resolve_depth_map(
        self._depth_source,
        est,
        rgb,
        sensor_depth,
        camera_K,
        camera_pose,
        rgb_right,
        camera_K_right,
        camera_pose_right,
        da3_use_stereo=self._da3_use_stereo,
    )


def _rerun_live_status_markdown(self) -> str:
    """Short markdown for the Rerun text panel: mapping progress and graph state."""
    lines: list[str] = [f"- **Observation step:** {self.obs_count}"]
    vm = getattr(self, "voxel_map", None)
    if vm is not None:
        try:
            obstacles, explored = vm.get_2d_map()
            if hasattr(obstacles, "cpu"):
                obstacles = obstacles.cpu().numpy()
            if hasattr(explored, "cpu"):
                explored = explored.cpu().numpy()
            obs_cells = int(obstacles.sum()) if obstacles is not None and obstacles.size else 0
            exp_cells = int(explored.sum()) if explored is not None and explored.size else 0
            lines.append(f"- **2D map:** {exp_cells} explored cells, {obs_cells} obstacle cells")
        except Exception:
            lines.append("- **2D map:** (unavailable)")
        pcd = getattr(vm, "voxel_pcd", None)
        pts = getattr(pcd, "_points", None) if pcd is not None else None
        if pts is not None:
            n = int(pts.shape[0]) if hasattr(pts, "shape") else 0
            lines.append(f"- **Voxel point cloud:** {n} points")
    gm = getattr(self, "graph_memory", None)
    if gm is not None:
        try:
            lines.append(f"- **Graph memory nodes:** {len(gm.get_nodes())}")
        except Exception:
            lines.append("- **Graph memory:** (unavailable)")
    return "\n".join(lines)


def _rerun_refresh_monologue_panel(self) -> None:
    """Push ``robot_monologue`` = stored plan/EQA text plus live mapping status."""
    if not getattr(self.rerun_visualizer, "enabled", True):
        return
    base = (self._rerun_monologue_base or "").strip()
    if not base:
        base = "*No navigation plan or EQA answer in this session step yet — building the map from depth.*"
    live = self._rerun_live_status_markdown()
    doc = f"{base}\n\n---\n\n## Live status\n{live}"
    self.rerun_visualizer.log_text("robot_monologue", doc)


def _run_full_perception(self) -> bool:
    """True when this update should run the expensive perception stack.

    Occupancy/clearance (what navigation needs) updates every frame; YoloE +
    SigLIP + instance memory + graph sync are throttled to every
    ``perception_every_n`` frames. ``perception_every_n=1`` = always (old
    behavior). The throttled path still produces a fresh depth/pointcloud, so
    A* has current obstacles; only object-level recall lags by one cadence.
    """
    return self._perception_every_n <= 1 or (self.obs_count - 1) % self._perception_every_n == 0


def update(self):
    """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""

    _t_update0 = time.time()
    obs = self.robot.get_observation()
    if obs is None:
        logger.warning("get_observation() returned None; skipping voxel update")
        self.robot.set_mapping_depth_for_rerun(None)
        return
    self.obs_count += 1
    rgb, sensor_depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
    run_infer_full = self._da3_infer_every_n <= 1 or (self.obs_count - 1) % self._da3_infer_every_n == 0
    if self._depth_source == "lingbot":
        run_infer_full = self._lingbot_infer_every_n <= 1 or (self.obs_count - 1) % self._lingbot_infer_every_n == 0
    depth: np.ndarray | None
    if (
        not run_infer_full
        and self._depth_source in ("da3", "auto")
        and not getattr(self, "_debug_perfect_sensor_depth", False)
        and self._da3_last_depth is not None
        and self._da3_last_depth.shape[:2] == rgb.shape[:2]
    ):
        depth = np.asarray(self._da3_last_depth, dtype=np.float32).copy()
        self._depth_map_from_da3_infer = True
        if self._depth_source == "auto" and sensor_depth is not None and np.asarray(sensor_depth).size > 0:
            depth = np.asarray(sensor_depth, dtype=np.float32)
            self._depth_map_from_da3_infer = False
    elif (
        not run_infer_full
        and self._depth_source == "lingbot"
        and self._lingbot_last_depth is not None
        and self._lingbot_last_depth.shape[:2] == rgb.shape[:2]
    ):
        depth = np.asarray(self._lingbot_last_depth, dtype=np.float32).copy()
        self._depth_map_from_da3_infer = True
    else:
        depth = self._resolve_depth_map(
            rgb,
            sensor_depth,
            K,
            camera_pose,
            rgb_right=getattr(obs, "head_rgb_right", None),
            camera_K_right=getattr(obs, "head_camera_K_right", None),
            camera_pose_right=getattr(obs, "head_camera_pose_right", None),
        )
        if depth is not None and getattr(self, "_depth_map_from_da3_infer", False):
            if self._depth_source == "lingbot":
                self._lingbot_last_depth = np.asarray(depth, dtype=np.float32).copy()
            else:
                self._da3_last_depth = np.asarray(depth, dtype=np.float32).copy()
    if self._depth_source == "lingbot" and getattr(self, "_lingbot_last_pose", None) is not None:
        if self._lingbot_use_pose:
            camera_pose = self._lingbot_last_pose
    if depth is None:
        logger.error(f"No depth map available (depth_source={self._depth_source!r}); skipping voxel update.")
        self.robot.set_mapping_depth_for_rerun(None)
        return
    if getattr(self, "_depth_map_from_da3_infer", False):
        depth = np.asarray(depth, dtype=np.float32)
        sky = float(self.parameters.get("da3_ignore_sky_fraction_top", 0.0) or 0.0)
        if sky > 0.0:
            depth = apply_da3_sky_row_mask(depth, sky)
        speckle_k = int(self.parameters.get("filters/depth_speckle_open_kernel", 0) or 0)
        if speckle_k > 0:
            depth = apply_depth_speckle_filter(
                depth,
                open_kernel=speckle_k,
                open_iterations=int(self.parameters.get("filters/depth_speckle_open_iterations", 1) or 1),
                min_depth=float(self.parameters.get("min_depth", 0.25)),
                max_depth=float(self.parameters.get("max_depth", 2.5)),
            )
    self.robot.set_mapping_depth_for_rerun(depth)
    base_xyt = None
    if obs.gps is not None and obs.compass is not None:
        g = np.asarray(obs.gps, dtype=np.float64).reshape(-1)
        c = np.asarray(obs.compass, dtype=np.float64).ravel()
        if g.size >= 2 and c.size >= 1:
            local_xyt = np.array([float(g[0]), float(g[1]), float(c[0])], dtype=np.float64)
            base_xyt = nav_xyt_to_world_xyt(local_xyt, getattr(obs, "emet_session", None))
    if _env_truthy("EMET_DYNAMEM_MAP_DEBUG"):
        sess = getattr(obs, "emet_session", None) or {}
        org = sess.get("navigation_origin_xyt")
        cam_t = None
        if camera_pose is not None:
            cp = np.asarray(camera_pose, dtype=np.float64)
            if cp.shape == (4, 4):
                cam_t = np.round(cp[:3, 3], 4).tolist()
        logger.info(
            "dynamem_map_debug step=%s depth_source=%s da3_infer=%s perfect_depth=%s "
            "nav_origin_xyt=%s base_xyt=%s camera_t=%s",
            self.obs_count,
            self._depth_source,
            bool(getattr(self, "_depth_map_from_da3_infer", False)),
            bool(getattr(self, "_debug_perfect_sensor_depth", False)),
            None if org is None else np.asarray(org, dtype=np.float64).round(4).tolist(),
            None if base_xyt is None else np.asarray(base_xyt, dtype=np.float64).round(4).tolist(),
            cam_t,
        )
    if getattr(obs, "emet_session", None) is not None:
        org = getattr(obs, "emet_session", {}).get("navigation_origin_xyt")
        if org is not None:
            self._cached_navigation_origin_xyt = np.asarray(org, dtype=np.float64).reshape(-1)[:3].copy()

    self.voxel_map.process_rgbd_images(
        rgb, depth, K, camera_pose, base_xyt=base_xyt, full_perception=self._run_full_perception()
    )
    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
        print(f"[update] process_rgbd={time.monotonic():.3f}", flush=True)
    robot_xy = None
    if obs.gps is not None and obs.compass is not None:
        g = np.asarray(obs.gps, dtype=np.float64).reshape(-1)
        cc = np.asarray(obs.compass, dtype=np.float64).ravel()
        if g.size >= 2 and cc.size >= 1:
            wxyt = nav_xyt_to_world_xyt(
                np.array([float(g[0]), float(g[1]), float(cc[0])], dtype=np.float64),
                getattr(obs, "emet_session", None),
            )
            robot_xy = (float(wxyt[0]), float(wxyt[1]))
    if getattr(self.rerun_visualizer, "enabled", True):
        self.rerun_visualizer.log_topdown_map_snapshot(self.voxel_map, robot_base_xy=robot_xy)
    if self.voxel_map.voxel_pcd._points is not None:
        self.rerun_visualizer.update_voxel_map(space=self.space, robot_base_xy=robot_xy)
    if self.voxel_map.semantic_memory._points is not None:
        self.rerun_visualizer.log_custom_pointcloud(
            "world/semantic_memory/pointcloud",
            self.voxel_map.semantic_memory._points.detach().cpu(),
            self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
            0.03,
        )
    if self.use_scene_graph and self.voxel_map.use_instance_memory and self.graph_memory is None:
        instances = self.get_voxel_map().get_instances()
        if instances:
            self._update_scene_graph()
            self.rerun_visualizer.update_scene_graph(
                self.scene_graph,
                self.semantic_sensor,
                detection_model=getattr(self, "detection_model", None),
                graph_memory=self.graph_memory,
            )

    has_hm3d_labeler = getattr(self.robot, "hm3d_semantic_labeler", None) is not None
    if self._run_full_perception() and self.graph_memory is not None and getattr(self, "_lazy_graph_mode", False):
        from emet.memory.graph_eqa.ingest.lazy_graph_commit import record_lazy_graph_viewpoint

        record_lazy_graph_viewpoint(
            graph_memory=self.graph_memory,
            robot=self.robot,
            obs=obs,
            frame_step=self.obs_count,
        )
    elif (
        self._run_full_perception()
        and self.graph_memory is not None
        and (self.sensor_builder is not None or self._graph_eqa_use_instance_graph or has_hm3d_labeler)
    ):
        if getattr(self, "_skip_graph_perception_updates", False):
            from emet.memory.graph_eqa.ingest.dynamem_graph_hooks import (
                update_graph_memory_ground_truth_from_observation,
            )

            update_graph_memory_ground_truth_from_observation(
                graph_memory=self.graph_memory,
                robot=self.robot,
                obs=obs,
                frame_step=self.obs_count,
            )
        else:
            from emet.memory.graph_eqa.ingest.dynamem_graph_hooks import (
                update_graph_memory_from_dynamem_observation,
            )

            update_graph_memory_from_dynamem_observation(
                graph_memory=self.graph_memory,
                robot=self.robot,
                voxel_map=self.voxel_map,
                detection_model=self.detection_model,
                sensor_builder=self.sensor_builder,
                use_instance_graph=self._graph_eqa_use_instance_graph,
                use_sensor_perception=self._graph_eqa_use_sensor_perception,
                dedup_skips=self._graph_dedup_skips,
                obs=obs,
                frame_step=self.obs_count,
                graph_object_fusion=getattr(self, "_graph_object_fusion", None),
                calibration_writer=getattr(self, "_calibration_writer", None),
            )
            if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                print(f"[update] graph_update={time.monotonic():.3f}", flush=True)

    if self.graph_memory is not None:
        self._sync_graph_frontier_nodes()

    # Visualize open-vocab scene graph if attached (dynagraph uses graph_memory instead).
    ovsg = self.voxel_map.get_scene_graph()
    if ovsg is not None and ovsg.num_objects > 0 and self.graph_memory is None:
        self.rerun_visualizer.update_open_vocab_scene_graph(ovsg)

    self._rerun_refresh_monologue_panel()
    self._run_on_step_callbacks()
    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
        print(f"[update] obs_count={self.obs_count} wall={time.time() - _t_update0:.3f}s", flush=True)


def _run_on_step_callbacks(self) -> None:
    for cb in getattr(self, "_on_step_callbacks", ()) or ():
        try:
            cb(self)
        except Exception as exc:
            logger.warning(f"on_step callback failed: {exc}")


def _update_scene_graph(self) -> None:
    """Update the scene graph with the latest instances from the voxel map."""
    if self.scene_graph is None:
        self.scene_graph = SceneGraph(self.parameters, self.get_voxel_map().get_instances())
    else:
        self.scene_graph.update(self.get_voxel_map().get_instances())
    self.scene_graph.get_relationships(debug=False)
