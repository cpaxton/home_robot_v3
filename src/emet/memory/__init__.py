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
#
# Three memory models: sparse voxel map (emet.mapping.voxel), DynaMem (re-exported here),
# and Graph EQA (here). Dynamem implementation remains in emet.mapping.voxel.

from emet.memory.backend import MemoryBackend, get_memory_backend
from emet.memory.format import (
    MemoryState,
    UserMessageBlob,
    is_memory_directory,
    load_memory,
    save_memory,
)
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

__all__ = [
    "GraphEQAMemory",
    "SparseVoxelMapDynamem",
    "SparseVoxelMapNavigationSpaceDynamem",
    "MemoryState",
    "UserMessageBlob",
    "save_memory",
    "load_memory",
    "is_memory_directory",
    "MemoryBackend",
    "get_memory_backend",
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
