# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Semantic / EQA memory models. Both Dynamem (voxel-based) and GraphEQA (graph-based) live here
# for consistency; Dynamem implementation remains in emet.mapping.voxel and is re-exported from
# emet.memory.dynamem.

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

__all__ = [
    "GraphEQAMemory",
    "SparseVoxelMapDynamem",
    "SparseVoxelMapNavigationSpaceDynamem",
]


def __getattr__(name: str):
    """Lazy re-export of Dynamem types so GraphEQA can be used without mapping.voxel."""
    if name in ("SparseVoxelMapDynamem", "SparseVoxelMapNavigationSpaceDynamem"):
        from emet.memory.dynamem import (
            SparseVoxelMapDynamem,
            SparseVoxelMapNavigationSpaceDynamem,
        )
        return SparseVoxelMapDynamem if name == "SparseVoxelMapDynamem" else SparseVoxelMapNavigationSpaceDynamem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
