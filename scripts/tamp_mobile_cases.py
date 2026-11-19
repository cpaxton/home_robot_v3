# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Validate declared simulator initial conditions before building a scene."""

import json
from pathlib import Path

import numpy as np


def load_case(path):
    case = {} if path is None else json.loads(Path(path).read_text())
    allowed = {
        "initial_base_x",
        "initial_object_xy",
        "initial_distractor_xy",
        "placement_region",
        "mobile",
        "approach_m",
        "departure_m",
    }
    if not isinstance(case, dict) or set(case) - allowed or type(case.get("mobile", True)) is not bool:
        raise ValueError("invalid_mobile_case_config")
    for name in ("initial_base_x", "approach_m", "departure_m"):
        if name in case and (
            isinstance(case[name], bool) or not isinstance(case[name], (int, float)) or not np.isfinite(case[name])
        ):
            raise ValueError("invalid_mobile_case_config")
    for name in ("initial_object_xy", "initial_distractor_xy"):
        if name in case:
            value = np.asarray(case[name], float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("invalid_mobile_case_config")
    region = np.asarray(case.get("placement_region", [[0.06, -0.73], [0.22, -0.57]]), float)
    if region.shape != (2, 2) or not np.isfinite(region).all() or np.any(region[0] >= region[1]):
        raise ValueError("invalid_mobile_case_config")
    return case
