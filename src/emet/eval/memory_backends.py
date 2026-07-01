# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared memory-backend names across benchmark harnesses."""

from __future__ import annotations

from typing import Literal

# OVMM find-phase (Emet sim + Habitat proxy): includes oracle + GraphEQA-only row.
OVMM_MEMORY_BACKEND = Literal["dynamem", "graph_eqa", "dynagraph", "ground_truth"]
OVMM_MEMORY_BACKENDS: tuple[str, ...] = ("dynamem", "graph_eqa", "dynagraph", "ground_truth")

# SQA3D embodied QA: voxel-only vs Dynagraph (voxel + graph).
SQA3D_MEMORY_BACKEND = Literal["dynamem", "dynagraph"]
SQA3D_MEMORY_BACKENDS: tuple[str, ...] = ("dynamem", "dynagraph")
