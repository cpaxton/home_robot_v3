# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Simulator-independent XY grid conventions shared by base and arm planning."""

from typing import Literal

import numpy as np

GridConvention = Literal["grid_params", "world_offset"]


def world_xy_to_grid(
    x: float,
    y: float,
    *,
    grid_origin: np.ndarray,
    resolution: float,
    convention: GridConvention = "grid_params",
) -> tuple[int, int]:
    """World XY → obstacle-grid indices."""
    go = np.asarray(grid_origin, dtype=np.float64).reshape(-1)
    res = float(resolution)
    if convention == "world_offset":
        gi = int(np.floor((float(x) - float(go[0])) / res))
        gj = int(np.floor((float(y) - float(go[1])) / res))
    else:
        gi = int(np.floor(float(x) / res + float(go[0])))
        gj = int(np.floor(float(y) / res + float(go[1])))
    return gi, gj
