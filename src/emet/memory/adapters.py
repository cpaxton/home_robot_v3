# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Memory backend adapters: DynaMem, GraphEQA, SVM.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from emet.memory.backend import CheckMemoryResult, LocalizeResult, MemoryBackend


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
        result = self._voxel_map.localize_text(
            text, debug=False, return_debug=True
        )
        # localize_text returns (target_point, debug_text) or (target_point, debug_text, obs_id, point)
        target_point = result[0] if isinstance(result, (list, tuple)) else result
        debug_text = result[1] if len(result) > 1 else ""
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
            confidence = min(1.0, max(0.0, (max_align - self._confidence_threshold) / (1.0 - self._confidence_threshold)))
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
        target_point = result[0] if isinstance(result, (list, tuple)) else result
        debug_text = result[1] if len(result) > 1 else ""
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

    def save(self, path: str) -> None:
        self._voxel_map.write_to_pickle(path)

    def load(self, path: str) -> None:
        self._voxel_map.read_from_pickle(path)

    def supports_save_load(self) -> bool:
        return True


class GraphEQABackend(MemoryBackend):
    """Adapter for GraphEQAMemory. Uses graph for check/query; localize uses graph node positions."""

    def __init__(
        self,
        graph_memory: Any,
        voxel_map: Optional[Any] = None,
    ):
        """Wrap GraphEQAMemory. Optionally provide voxel_map for richer localize (e.g. DynaMem voxel)."""
        self._graph = graph_memory
        self._voxel_map = voxel_map

    def check_memory_for_object(self, text: str) -> CheckMemoryResult:
        text_lower = text.lower().strip()
        nodes = self._graph.get_nodes()
        for node in nodes:
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

    def list_objects(self) -> List[str]:
        labels = []
        for node in self._graph.get_nodes():
            labels.extend(node.labels)
        return list(dict.fromkeys(labels))

    def query_answer(
        self,
        question: str,
        xyt: Optional[Union[Any, np.ndarray, list]] = None,
        planner: Any = None,
    ) -> Tuple[str, str, bool, str, Optional[np.ndarray], Any]:
        return self._graph.query_answer(question, xyt, planner)

    def supports_save_load(self) -> bool:
        return False


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

    def list_objects(self) -> List[str]:
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
