# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Room partition backends. Only ``proximity`` is implemented; others are sweep stubs."""

from __future__ import annotations

import os
from typing import Any

BACKEND_PROXIMITY = "proximity"
BACKEND_OCCUPANCY_CC = "occupancy_cc"
BACKEND_PORTAL = "portal"

IMPLEMENTED_BACKENDS = frozenset({BACKEND_PROXIMITY})
KNOWN_BACKENDS = frozenset({BACKEND_PROXIMITY, BACKEND_OCCUPANCY_CC, BACKEND_PORTAL})
DEFAULT_BACKEND = BACKEND_PROXIMITY
DEFAULT_LINK_RADIUS_M = 2.0


def resolve_backend(raw: Any | None = None) -> str:
    """``proximity`` unless ``raw`` or ``EMET_EQA_ROOM_CLUSTERING_BACKEND`` says otherwise."""
    text = str(raw or "").strip().lower()
    if not text:
        text = os.environ.get("EMET_EQA_ROOM_CLUSTERING_BACKEND", "").strip().lower()
    if not text:
        return DEFAULT_BACKEND
    if text not in KNOWN_BACKENDS:
        raise ValueError(f"unknown room clustering backend {text!r}; want one of {sorted(KNOWN_BACKENDS)}")
    return text
