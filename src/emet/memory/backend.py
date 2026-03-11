# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Unified memory backend interface for SVM, DynaMem, and GraphEQA.
# Tools and the agent use this interface so they do not depend on concrete backend types.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CheckMemoryResult:
    """Result of check_memory_for_object(text)."""

    confidence: float  # 0.0 to 1.0
    location_xyz: Optional[np.ndarray]  # (3,) world position if known
    extra_info: Dict[str, Any]  # backend-specific (e.g. debug_text, obs_id)


@dataclass
class LocalizeResult:
    """Result of localize_text(text) for navigation."""

    point_xyz: Optional[np.ndarray]  # (3,) or (2,) world position
    success: bool
    extra_info: Dict[str, Any]


class MemoryBackend(ABC):
    """Abstract interface for semantic memory used by agent tools.

    Implementations: DynaMem (voxel), GraphEQA (graph), SVM (instance memory).
    """

    @abstractmethod
    def check_memory_for_object(self, text: str) -> CheckMemoryResult:
        """Check whether the object/location is in memory and return confidence and optional location.

        Returns:
            CheckMemoryResult with confidence in [0, 1], optional location_xyz, and extra_info.
        """
        pass

    @abstractmethod
    def localize_text(self, text: str) -> LocalizeResult:
        """Localize the object/location in the map for navigation.

        Returns:
            LocalizeResult with point_xyz (for navigation) and success flag.
        """
        pass

    def list_objects(self) -> List[str]:
        """Optional: list known object/location labels. Default returns empty list."""
        return []

    def query_answer(
        self,
        question: str,
        xyt: Optional[Union[Any, np.ndarray, list]] = None,
        planner: Any = None,
    ) -> Tuple[str, str, bool, str, Optional[np.ndarray], Any]:
        """Optional: EQA-style query. Only backends that support EQA implement this.

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        raise NotImplementedError("This backend does not support query_answer")

    def save(self, path: str) -> None:
        """Optional: persist memory to path. Path must be a directory (common format:
        manifest.json, point_cloud.npz, frames, etc.).
        """
        raise NotImplementedError("This backend does not support save")

    def load(self, path: str) -> None:
        """Optional: load memory from path. Path must be a directory (common format)."""
        raise NotImplementedError("This backend does not support load")

    def supports_save_load(self) -> bool:
        """Whether save/load are implemented."""
        return False


def get_memory_backend(
    name: str,
    *,
    voxel_map: Any = None,
    graph_memory: Any = None,
    agent: Any = None,
    confidence_threshold: float = 0.14,
) -> MemoryBackend:
    """Factory: return a MemoryBackend for the given name and dependencies.

    Args:
        name: "dynamem" | "graph_eqa" | "svm"
        voxel_map: For dynamem: the SparseVoxelMapDynamem. For graph_eqa: optional voxel for localize.
        graph_memory: For graph_eqa: the GraphEQAMemory instance.
        agent: For svm: the agent with get_found_instances_by_class (InstanceMemoryController).
        confidence_threshold: For dynamem: minimum similarity for confidence.

    Returns:
        MemoryBackend implementation.
    """
    if name == "dynamem":
        if voxel_map is None:
            raise ValueError("get_memory_backend(name='dynamem') requires voxel_map")
        from emet.memory.adapters import DynaMemBackend
        return DynaMemBackend(voxel_map, confidence_threshold=confidence_threshold)
    if name == "graph_eqa":
        if graph_memory is None:
            raise ValueError("get_memory_backend(name='graph_eqa') requires graph_memory")
        from emet.memory.adapters import GraphEQABackend
        return GraphEQABackend(graph_memory, voxel_map=voxel_map)
    if name == "svm":
        if agent is None:
            raise ValueError("get_memory_backend(name='svm') requires agent")
        from emet.memory.adapters import SVMBackend
        return SVMBackend(agent)
    raise ValueError(f"Unknown memory backend: {name!r}. Use 'dynamem', 'graph_eqa', or 'svm'.")
