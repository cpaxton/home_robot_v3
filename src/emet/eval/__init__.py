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

"""Evaluation harnesses (OVMM find-phase, Habitat adapters).

Keep this package init lazy. Importing ``emet.eval.harness`` (or
``emet eval affinity``) must not pull OVMM / MuJoCo — job wrappers run
affinity immediately after a previous sim process releases the GPU lock.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FindPhaseEpisode",
    "compute_find_phase_metrics",
    "load_find_phase_episodes",
    "score_find_object",
    "score_find_recep",
]

_LAZY_OVMM = {
    "FindPhaseEpisode": "emet.eval.ovmm_find_phase",
    "compute_find_phase_metrics": "emet.eval.ovmm_find_phase",
    "load_find_phase_episodes": "emet.eval.ovmm_find_phase",
    "score_find_object": "emet.eval.ovmm_find_phase",
    "score_find_recep": "emet.eval.ovmm_find_phase",
}


def __getattr__(name: str) -> Any:
    mod_name = _LAZY_OVMM.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(mod_name), name)
    globals()[name] = value
    return value
