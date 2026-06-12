# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Memory backend adapters: DynaMem, GraphEQA, SVM.
# Save/load use the common directory format (memory/format.py) only.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from emet.memory.backend import CheckMemoryResult, LocalizeResult, MemoryBackend
from emet.memory.format import (
    SIM_GT_PLACEMENTS_FILENAME,
    FrameBlob,
    GraphBlob,
    GraphEdgeView,
    GraphNodeView,
    MemoryManifest,
    MemoryState,
    PointCloudBlob,
    is_memory_directory,
    load_memory,
    save_memory,
)


def _to_numpy(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _split_voxel_localize_result(result: Any) -> tuple[Any, str]:
    """Parse ``SparseVoxelMapDynamem.localize_text`` return (may be bare ``None`` or a tuple)."""
    if result is None:
        return None, ""
    if isinstance(result, (list, tuple)):
        target_point = result[0] if result else None
        debug_text = str(result[1]) if len(result) > 1 and result[1] is not None else ""
        return target_point, debug_text
    return result, ""


def frame_blobs_from_voxel_map(vm: Any) -> list[FrameBlob]:
    """Build ``FrameBlob`` list from ``voxel_map.observations`` + detection JSON cache rows."""
    from emet.memory.graph_eqa.graph_observation_pipeline import build_detections_json_rows

    det_model = getattr(vm, "detection_model", None)
    md = float(getattr(vm, "min_depth", 0.1))
    xd = float(getattr(vm, "max_depth", 4.0))
    frames: list[FrameBlob] = []
    for obs in getattr(vm, "observations", []) or []:
        cp = getattr(obs, "camera_pose", None)
        if cp is None:
            continue
        cp = _to_numpy(cp)
        rgb = _to_numpy(getattr(obs, "rgb", None))
        depth = _to_numpy(getattr(obs, "depth", None))
        camera_K = _to_numpy(getattr(obs, "camera_K", None))
        base_pose = _to_numpy(getattr(obs, "base_pose", None))
        feats = _to_numpy(getattr(obs, "feats", None))
        world_xyz = _to_numpy(getattr(obs, "full_world_xyz", None))
        inst = _to_numpy(getattr(obs, "instance", None))
        ic = _to_numpy(getattr(obs, "instance_classes", None))
        isc = _to_numpy(getattr(obs, "instance_scores", None))
        info = getattr(obs, "info", None)
        gt_associations = None
        if isinstance(info, dict):
            gt_associations = info.get("gt_associations")
        fb = FrameBlob(
            camera_pose=cp,
            base_pose=base_pose,
            camera_K=camera_K,
            rgb=rgb,
            depth=depth,
            feats=feats,
            world_xyz=world_xyz,
            instance=inst,
            instance_classes=ic,
            instance_scores=isc,
            gt_associations=gt_associations,
            info=info,
        )
        dets = build_detections_json_rows(fb, min_depth=md, max_depth=xd, detection_model=det_model)
        if dets:
            fb = replace(fb, detections=dets)
        frames.append(fb)
    return frames


def _restore_dynamem_from_state(voxel_map: Any, state: MemoryState) -> None:
    """Repopulate DynaMem voxel_map from MemoryState (semantic_memory from point_cloud)."""
    if state.point_cloud is None:
        return
    pc = state.point_cloud
    sm = voxel_map.semantic_memory
    sm._points = torch.from_numpy(np.asarray(pc.xyz, dtype=np.float32))
    if pc.rgb is not None:
        sm._rgb = torch.from_numpy(np.asarray(pc.rgb, dtype=np.float32))
    else:
        sm._rgb = torch.ones_like(sm._points)
    sm._weights = (
        torch.from_numpy(np.asarray(pc.weights, dtype=np.float32))
        if pc.weights is not None
        else torch.ones(sm._points.shape[0], 1, dtype=torch.float32)
    )
    sm._features = torch.from_numpy(np.asarray(pc.feats, dtype=np.float32)) if pc.feats is not None else sm._rgb
    sm._obs_counts = (
        torch.from_numpy(np.asarray(pc.obs_id, dtype=np.int64).ravel())
        if pc.obs_id is not None
        else torch.ones(sm._points.shape[0], dtype=torch.long)
    )
    sm._mins = sm._points.min(dim=0).values
    sm._maxs = sm._points.max(dim=0).values
    sm.obs_count = int(sm._obs_counts.max().item()) if sm._obs_counts.numel() else 0


class DynaMemBackend(MemoryBackend):
    """Adapter for SparseVoxelMapDynamem (voxel + semantic memory)."""

    def __init__(
        self,
        voxel_map: Any,
        confidence_threshold: float = 0.14,
    ):
        """Wrap a DynaMem voxel map (from controller.agent.voxel_map)."""
        self._voxel_map = voxel_map
        self._confidence_threshold = confidence_threshold

    def check_memory_for_object(self, text: str) -> CheckMemoryResult:
        result = self._voxel_map.localize_text(text, debug=False, return_debug=True)
        target_point, debug_text = _split_voxel_localize_result(result)
        if target_point is None:
            return CheckMemoryResult(
                confidence=0.0,
                location_xyz=None,
                extra_info={"debug_text": debug_text},
            )
        # Use alignment/similarity if available for confidence
        points, _, _, _ = self._voxel_map.semantic_memory.get_pointcloud()
        alignments = None
        if hasattr(self._voxel_map, "find_alignment_over_model") and points is not None:
            alignments = self._voxel_map.find_alignment_over_model(text)
        if alignments is not None:
            max_align = float(alignments.max().item())
            confidence = min(
                1.0, max(0.0, (max_align - self._confidence_threshold) / (1.0 - self._confidence_threshold))
            )
        else:
            confidence = 0.8 if target_point is not None else 0.0
        if hasattr(target_point, "cpu"):
            target_point = target_point.detach().cpu().numpy()
        if isinstance(target_point, np.ndarray) and target_point.size >= 3:
            xyz = np.array(target_point.flat[:3], dtype=float)
        elif isinstance(target_point, np.ndarray) and target_point.size == 2:
            xyz = np.array([float(target_point.flat[0]), float(target_point.flat[1]), 0.0])
        else:
            xyz = None
        return CheckMemoryResult(
            confidence=confidence,
            location_xyz=xyz,
            extra_info={"debug_text": debug_text},
        )

    def localize_text(self, text: str) -> LocalizeResult:
        result = self._voxel_map.localize_text(text, debug=False, return_debug=True)
        target_point, debug_text = _split_voxel_localize_result(result)
        if target_point is None:
            return LocalizeResult(
                point_xyz=None,
                success=False,
                extra_info={"debug_text": debug_text},
            )
        if hasattr(target_point, "cpu"):
            target_point = target_point.detach().cpu().numpy()
        if isinstance(target_point, np.ndarray) and target_point.size >= 3:
            xyz = np.array(target_point.flat[:3], dtype=float)
        elif isinstance(target_point, np.ndarray) and target_point.size == 2:
            xyz = np.array([float(target_point.flat[0]), float(target_point.flat[1]), 0.0])
        else:
            xyz = np.array(target_point, dtype=float)
        return LocalizeResult(
            point_xyz=xyz,
            success=True,
            extra_info={"debug_text": debug_text},
        )

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, Any]:
        """EQA-style query delegating to the underlying voxel map.

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        vm = self._voxel_map
        if not hasattr(vm, "query_answer"):
            raise NotImplementedError("This voxel map does not support query_answer")
        if not getattr(vm, "run_eqa", False):
            raise NotImplementedError(
                "DynaMem EQA is not enabled on this map (run_eqa=False). "
                "Use object-style questions with localize_text fallback, enable --eqa on the agent "
                "(eqa.backend in dynav_config.yaml defaults to qwen_vl), "
                "or use describe_scene / send_image for open-ended visual questions."
            )
        if not hasattr(vm, "image_description_client") or not hasattr(vm, "eqa_client"):
            raise NotImplementedError(
                "DynaMem EQA clients are not initialized (expected image_description_client and eqa_client)."
            )
        if vm.image_description_client is None or vm.eqa_client is None:
            raise NotImplementedError(
                "DynaMem EQA vision clients are not ready yet (image_description_client or eqa_client is None). "
                "With agent --eqa and --share-memory-vllm, wait until the agent LLM loads and binds the shared VL "
                "client, or use --no-share-memory-vllm to load the local EQA VLM at startup."
            )
        return vm.query_answer(question, xyt, planner)

    def save(self, path: str, extra_graph: GraphBlob | None = None) -> None:
        """Save to common directory format. Path must be a directory.

        If extra_graph is provided (e.g. from mapping scene graph), it is included in the saved state.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        vm = self._voxel_map
        point_cloud = None
        if hasattr(vm, "semantic_memory") and vm.semantic_memory is not None:
            pts, feats, weights, rgb = vm.semantic_memory.get_pointcloud()
            if pts is not None and pts.numel() > 0:
                obs_id = getattr(vm.semantic_memory, "_obs_counts", None)
                point_cloud = PointCloudBlob(
                    xyz=pts.cpu().numpy() if hasattr(pts, "cpu") else pts,
                    rgb=rgb.cpu().numpy() if rgb is not None and hasattr(rgb, "cpu") else rgb,
                    feats=feats.cpu().numpy() if feats is not None and hasattr(feats, "cpu") else feats,
                    weights=weights.cpu().numpy() if weights is not None and hasattr(weights, "cpu") else weights,
                    obs_id=obs_id.cpu().numpy() if obs_id is not None and hasattr(obs_id, "cpu") else obs_id,
                )

        frames = frame_blobs_from_voxel_map(vm)

        grid_origin = getattr(vm, "grid_origin", None)
        if grid_origin is not None and hasattr(grid_origin, "cpu"):
            grid_origin = grid_origin.cpu().numpy()
        grid_resolution = float(getattr(vm, "grid_resolution", 0.05))
        obstacles_2d = None
        explored_2d = None
        if hasattr(vm, "get_2d_map"):
            try:
                obstacles_2d, explored_2d = vm.get_2d_map()
                if hasattr(obstacles_2d, "cpu"):
                    obstacles_2d = obstacles_2d.cpu().numpy()
                if hasattr(explored_2d, "cpu"):
                    explored_2d = explored_2d.cpu().numpy()
            except Exception:
                pass

        state = MemoryState(
            point_cloud=point_cloud,
            frames=frames,
            grid_origin=grid_origin,
            grid_resolution=grid_resolution,
            obstacles_2d=obstacles_2d,
            explored_2d=explored_2d,
            graph=extra_graph,
            manifest=MemoryManifest(backend="dynamem"),
        )
        save_memory(state, str(dir_path))

    def load(self, path: str) -> None:
        """Load from common directory format."""
        path_obj = Path(path)
        if not path_obj.is_dir() or not is_memory_directory(path):
            raise FileNotFoundError(f"Not a memory directory: {path}")
        state = load_memory(path)
        _restore_dynamem_from_state(self._voxel_map, state)

    def supports_save_load(self) -> bool:
        return True


class GraphEQABackend(MemoryBackend):
    """Adapter for GraphEQAMemory. Uses graph for check/query; localize uses graph node positions."""

    def __init__(
        self,
        graph_memory: Any,
        voxel_map: Any | None = None,
    ):
        """Wrap GraphEQAMemory. Optionally provide voxel_map for richer localize (e.g. DynaMem voxel)."""
        self._graph = graph_memory
        self._voxel_map = voxel_map

    def check_memory_for_object(self, text: str) -> CheckMemoryResult:
        text_lower = text.lower().strip()
        nodes = self._graph.get_nodes()
        for node in nodes:
            if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                continue
            for label in node.labels:
                if text_lower in label.lower():
                    return CheckMemoryResult(
                        confidence=0.7,
                        location_xyz=np.array(node.xyz, dtype=float),
                        extra_info={"node_id": node.node_id},
                    )
        if self._voxel_map is not None and hasattr(self._voxel_map, "localize_text"):
            loc = self._voxel_map.localize_text(text, debug=False, return_debug=False)
            if loc is not None:
                if hasattr(loc, "cpu"):
                    loc = loc.detach().cpu().numpy()
                xyz = np.array(loc.flat[:3], dtype=float) if np.size(loc) >= 3 else None
                return CheckMemoryResult(
                    confidence=0.6,
                    location_xyz=xyz,
                    extra_info={},
                )
        return CheckMemoryResult(
            confidence=0.0,
            location_xyz=None,
            extra_info={},
        )

    def localize_text(self, text: str) -> LocalizeResult:
        text_lower = text.lower().strip()
        nodes = self._graph.get_nodes()
        for node in nodes:
            if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                continue
            for label in node.labels:
                if text_lower in label.lower():
                    return LocalizeResult(
                        point_xyz=np.array([node.xyz[0], node.xyz[1], 1.0], dtype=float),
                        success=True,
                        extra_info={"node_id": node.node_id},
                    )
        if self._voxel_map is not None and hasattr(self._voxel_map, "localize_text"):
            result = self._voxel_map.localize_text(text, debug=False, return_debug=True)
            target_point = result[0] if isinstance(result, (list, tuple)) else result
            if target_point is not None:
                if hasattr(target_point, "cpu"):
                    target_point = target_point.detach().cpu().numpy()
                xyz = np.array(target_point.flat[:3], dtype=float) if np.size(target_point) >= 3 else None
                if xyz is not None:
                    return LocalizeResult(point_xyz=xyz, success=True, extra_info={})
        return LocalizeResult(point_xyz=None, success=False, extra_info={})

    def list_objects(self) -> list[str]:
        labels = []
        for node in self._graph.get_nodes():
            labels.extend(node.labels)
        return list(dict.fromkeys(labels))

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, Any]:
        return self._graph.query_answer(question, xyt, planner)

    def save(
        self,
        path: str,
        *,
        ground_truth_mode: bool = False,
        sim_object_placements: dict[str, Any] | None = None,
    ) -> None:
        """Save to common directory format (graph + full voxel frame history when ``voxel_map`` is set)."""
        from emet.memory.graph_eqa.sim_ground_truth_graph import graph_node_export_fields

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        nodes = self._graph.get_nodes()
        edges = self._graph.get_edges()
        graph_blob = GraphBlob(
            nodes=[
                GraphNodeView(
                    node_id=n.node_id,
                    labels=list(n.labels),
                    xyz=list(np.ravel(n.xyz).tolist()),
                    obs_id=n.obs_id,
                    description=getattr(n, "description", None),
                    **graph_node_export_fields(n, sim_object_placements),
                )
                for n in nodes
            ],
            edges=[GraphEdgeView(id1=e[0], id2=e[1], relation=e[2]) for e in edges],
        )
        frames: list[FrameBlob]
        if self._voxel_map is not None:
            frames = frame_blobs_from_voxel_map(self._voxel_map)
        else:
            frames = []
            for obs in self._graph.get_observations():
                xyz = np.ravel(obs.xyz)
                if xyz.size < 3:
                    xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                pose = np.eye(4, dtype=np.float64)
                pose[:3, 3] = xyz[:3]
                info = None
                if obs.labels or getattr(obs, "description", None):
                    info = {"labels": list(obs.labels) if obs.labels else []}
                    if getattr(obs, "description", None):
                        info["description"] = obs.description
                frames.append(
                    FrameBlob(
                        camera_pose=pose,
                        base_pose=xyz[:3].tolist() if xyz.size >= 3 else None,
                        rgb=obs.rgb,
                        world_xyz=xyz.reshape(-1, 3)[0:1],
                        info=info,
                    )
                )

        has_sim_gt = bool(sim_object_placements)
        state = MemoryState(
            point_cloud=None,
            frames=frames,
            graph=graph_blob,
            manifest=MemoryManifest(
                backend="graph_eqa",
                has_point_cloud=False,
                ground_truth_mode=ground_truth_mode,
                has_sim_gt=has_sim_gt,
                sim_gt_placements_file=SIM_GT_PLACEMENTS_FILENAME if has_sim_gt else None,
            ),
        )
        save_memory(state, str(dir_path))

    def load(self, path: str) -> None:
        """Load from common directory format."""
        from emet.memory.graph_eqa.graph_memory import GraphNode, GraphObservation

        path_obj = Path(path)
        if not path_obj.is_dir() or not is_memory_directory(str(path)):
            raise FileNotFoundError(f"Not a memory directory: {path}")
        state = load_memory(path)
        if state.graph is None:
            return
        self._graph._nodes = [
            GraphNode(
                node_id=n.node_id,
                labels=list(n.labels),
                xyz=np.array(n.xyz, dtype=np.float64),
                obs_id=n.obs_id,
                description=getattr(n, "description", None),
            )
            for n in state.graph.nodes
        ]
        self._graph._edges = [(e.id1, e.id2, e.relation) for e in state.graph.edges]
        self._graph._observations = []
        for i, fr in enumerate(state.frames):
            xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            if fr.world_xyz is not None and fr.world_xyz.size >= 3:
                xyz = np.ravel(fr.world_xyz)[:3]
            labels = (fr.info or {}).get("labels", [])
            description = (fr.info or {}).get("description") if fr.info else None
            rgb = fr.rgb if fr.rgb is not None else np.zeros((1, 1, 3), dtype=np.uint8)
            self._graph._observations.append(
                GraphObservation(obs_id=i + 1, rgb=rgb, xyz=xyz, labels=labels, description=description)
            )
        self._graph._next_obs_id = max((n.obs_id for n in self._graph._nodes), default=0) + 1

    def supports_save_load(self) -> bool:
        return True

    def print_memory(self) -> str:
        """Return the 3D scene graph as an indented tree with objects and descriptions."""
        return self._graph.print_memory()


class SVMBackend(MemoryBackend):
    """Adapter for instance-memory (SVM) agent. Uses get_found_instances_by_class."""

    def __init__(self, agent: Any):
        """Wrap an agent that has get_found_instances_by_class (e.g. InstanceMemoryController)."""
        self._agent = agent

    def check_memory_for_object(self, text: str) -> CheckMemoryResult:
        matches = self._agent.get_found_instances_by_class(goal=text, threshold=0, debug=False)
        if not matches:
            return CheckMemoryResult(
                confidence=0.0,
                location_xyz=None,
                extra_info={"count": 0},
            )
        instance = matches[0][1]
        center = None
        if hasattr(instance, "get_center"):
            c = instance.get_center()
            if c is not None:
                center = c.cpu().numpy().flatten() if hasattr(c, "cpu") else np.array(c).flatten()
        if center is not None and center.size < 3:
            center = np.array([center[0], center[1], 0.0])
        confidence = min(1.0, 0.5 + 0.2 * len(matches))
        return CheckMemoryResult(
            confidence=confidence,
            location_xyz=center,
            extra_info={"count": len(matches)},
        )

    def localize_text(self, text: str) -> LocalizeResult:
        matches = self._agent.get_reachable_instances_by_class(goal=text, threshold=0, debug=False)
        if not matches:
            return LocalizeResult(point_xyz=None, success=False, extra_info={})
        instance = matches[0]
        center = None
        if hasattr(instance, "get_center"):
            c = instance.get_center()
            if c is not None:
                center = c.cpu().numpy().flatten() if hasattr(c, "cpu") else np.array(c).flatten()
        if center is not None and center.size < 3:
            center = np.array([center[0], center[1], 0.0])
        return LocalizeResult(
            point_xyz=center,
            success=center is not None,
            extra_info={},
        )

    def list_objects(self) -> list[str]:
        instances = getattr(self._agent.get_voxel_map(), "get_instances", lambda: [])()
        if not instances:
            return []
        if self._agent.semantic_sensor is None:
            return []
        names = []
        for inst in instances:
            oid = int(getattr(inst, "category_id", 0))
            if hasattr(oid, "item"):
                oid = int(oid.item())
            name = self._agent.semantic_sensor.get_class_name_for_id(oid)
            names.append(name)
        return list(dict.fromkeys(names))

    def save(self, path: str) -> None:
        """Save to common directory format (point cloud from voxel_map, optional frames)."""
        vm = self._agent.get_voxel_map()
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        point_cloud = None
        if hasattr(vm, "voxel_pcd") and vm.voxel_pcd is not None:
            try:
                points, _, _, rgb = vm.voxel_pcd.get_pointcloud()
                if points is not None and (
                    hasattr(points, "numel")
                    and points.numel() > 0
                    or getattr(points, "size", lambda: 0)
                    and points.size > 0
                ):
                    point_cloud = PointCloudBlob(
                        xyz=points.cpu().numpy() if hasattr(points, "cpu") else np.asarray(points),
                        rgb=rgb.cpu().numpy()
                        if rgb is not None and hasattr(rgb, "cpu")
                        else np.asarray(rgb)
                        if rgb is not None
                        else None,
                    )
            except Exception:
                pass

        grid_origin = getattr(vm, "grid_origin", None)
        if grid_origin is not None and hasattr(grid_origin, "cpu"):
            grid_origin = grid_origin.cpu().numpy()
        grid_resolution = float(getattr(vm, "grid_resolution", 0.05))
        obstacles_2d, explored_2d = None, None
        if hasattr(vm, "get_2d_map"):
            try:
                obstacles_2d, explored_2d = vm.get_2d_map()
                if hasattr(obstacles_2d, "cpu"):
                    obstacles_2d = obstacles_2d.cpu().numpy()
                if hasattr(explored_2d, "cpu"):
                    explored_2d = explored_2d.cpu().numpy()
            except Exception:
                pass

        state = MemoryState(
            point_cloud=point_cloud,
            frames=[],
            grid_origin=grid_origin,
            grid_resolution=grid_resolution,
            obstacles_2d=obstacles_2d,
            explored_2d=explored_2d,
            manifest=MemoryManifest(backend="svm"),
        )
        save_memory(state, str(dir_path))

    def load(self, path: str) -> None:
        """Load from common directory format. Restores point cloud if voxel_map has semantic_memory (DynaMem)."""
        path_obj = Path(path)
        if not path_obj.is_dir() or not is_memory_directory(path):
            raise FileNotFoundError(f"Not a memory directory: {path}")
        state = load_memory(path)
        real_vm = getattr(self._agent.get_voxel_map(), "_voxel_map", self._agent.get_voxel_map())
        if getattr(real_vm, "semantic_memory", None) is not None:
            _restore_dynamem_from_state(real_vm, state)

    def supports_save_load(self) -> bool:
        return True


class SceneGraphBackend(MemoryBackend):
    """Adapter for OpenVocabSceneGraph."""

    def __init__(self, scene_graph: Any, text_encoder: Any = None):
        self._sg = scene_graph
        self._text_encoder = text_encoder

    def check_memory_for_object(self, text: str) -> CheckMemoryResult:
        if self._text_encoder is None:
            node = self._sg.get_node_by_label(text)
            if node is not None and node.center is not None:
                return CheckMemoryResult(
                    confidence=0.8,
                    location_xyz=node.center,
                    extra_info={"node_id": node.node_id, "label": node.primary_label},
                )
            return CheckMemoryResult(confidence=0.0, location_xyz=None, extra_info={})

        confidence, location = self._sg.check_for_object(text, self._text_encoder)
        return CheckMemoryResult(
            confidence=confidence,
            location_xyz=location,
            extra_info={},
        )

    def localize_text(self, text: str) -> LocalizeResult:
        if self._text_encoder is None:
            node = self._sg.get_node_by_label(text)
            if node is not None and node.center is not None:
                return LocalizeResult(
                    point_xyz=node.center,
                    success=True,
                    extra_info={"node_id": node.node_id},
                )
            return LocalizeResult(point_xyz=None, success=False, extra_info={})

        center = self._sg.localize_text(text, self._text_encoder)
        if center is not None:
            return LocalizeResult(point_xyz=center, success=True, extra_info={})
        return LocalizeResult(point_xyz=None, success=False, extra_info={})

    def list_objects(self) -> list[str]:
        return self._sg.list_objects()

    def save(self, path: str) -> None:
        self._sg.save(path)

    def load(self, path: str) -> None:
        from emet.mapping.scene_graph.open_vocab_scene_graph import OpenVocabSceneGraph

        loaded = OpenVocabSceneGraph.load(path)
        self._sg.nodes = loaded.nodes
        self._sg.edges = loaded.edges
        self._sg._next_id = loaded._next_id

    def supports_save_load(self) -> bool:
        return True
