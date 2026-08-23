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
# Unified memory backend interface for SVM, DynaMem, and GraphEQA.
# Tools and the agent use this interface so they do not depend on concrete backend types.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CheckMemoryResult:
    """Result of check_memory_for_object(text)."""

    confidence: float  # 0.0 to 1.0
    location_xyz: np.ndarray | None  # (3,) world position if known
    extra_info: dict[str, Any]  # backend-specific (e.g. debug_text, obs_id)


@dataclass
class LocalizeResult:
    """Result of localize_text(text) for navigation."""

    point_xyz: np.ndarray | None  # (3,) or (2,) world position
    success: bool
    extra_info: dict[str, Any]


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

    def list_objects(self) -> list[str]:
        """Optional: list known object/location labels. Default returns empty list."""
        return []

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, Any]:
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

    def print_memory(self) -> str:
        """Optional: return memory as human-readable text (e.g. scene graph as tree)."""
        raise NotImplementedError("This backend does not support print_memory")


def get_memory_backend(
    name: str,
    *,
    voxel_map: Any = None,
    graph_memory: Any = None,
    scene_graph: Any = None,
    text_encoder: Any = None,
    agent: Any = None,
    confidence_threshold: float = 0.14,
) -> MemoryBackend:
    """Factory: return a MemoryBackend for the given name and dependencies.

    Args:
        name: "dynamem" | "static_graph" | "graph_eqa" (alias) | "svm" | "scene_graph"
        voxel_map: For dynamem: the SparseVoxelMapDynamem. For static_graph: optional voxel for localize.
        graph_memory: For static_graph: the GraphEQAMemory instance.
        scene_graph: For scene_graph: the OpenVocabSceneGraph instance.
        text_encoder: For scene_graph: encoder with encode_text().
        agent: For svm: the agent with get_found_instances_by_class (InstanceMemoryController).
        confidence_threshold: For dynamem: minimum similarity for confidence.

    Returns:
        MemoryBackend implementation.
    """
    from emet.eval.memory_backends import STATIC_GRAPH, is_static_graph_backend, normalize_benchmark_backend

    key = normalize_benchmark_backend(name, warn=False) if name else ""
    if key == "dynamem":
        if voxel_map is None:
            raise ValueError("get_memory_backend(name='dynamem') requires voxel_map")
        from emet.memory.adapters import DynaMemBackend

        return DynaMemBackend(voxel_map, confidence_threshold=confidence_threshold)
    if is_static_graph_backend(key) or key == STATIC_GRAPH:
        if graph_memory is None:
            raise ValueError(
                f"get_memory_backend(name={name!r}) requires graph_memory (canonical id: {STATIC_GRAPH!r})"
            )
        from emet.memory.adapters import GraphEQABackend

        return GraphEQABackend(graph_memory, voxel_map=voxel_map)
    if key == "svm":
        if agent is None:
            raise ValueError("get_memory_backend(name='svm') requires agent")
        from emet.memory.adapters import SVMBackend

        return SVMBackend(agent)
    if key == "scene_graph":
        if scene_graph is None:
            raise ValueError("get_memory_backend(name='scene_graph') requires scene_graph")
        from emet.memory.adapters import SceneGraphBackend

        return SceneGraphBackend(scene_graph, text_encoder=text_encoder)
    raise ValueError(
        f"Unknown memory backend: {name!r}. Use 'dynamem', '{STATIC_GRAPH}' (alias graph_eqa), 'svm', or 'scene_graph'."
    )
