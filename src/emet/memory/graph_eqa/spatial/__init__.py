# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Rooms, frontiers, spatial RAG, and place approaches.

``frontiers`` re-exports ``frontier_nodes`` and ``frontier_regions``.
"""

from emet.memory.graph_eqa.spatial.room_labels import (
    coerce_room_label,
    normalize_current_room,
    question_implies_indoor,
    sanitize_room_phrase,
)

__all__ = [
    "coerce_room_label",
    "normalize_current_room",
    "question_implies_indoor",
    "sanitize_room_phrase",
]
