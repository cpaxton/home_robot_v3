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
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.

"""Centralized env-based toggles for agent TTY / diagnostics."""

from __future__ import annotations

import os

_TRUE = frozenset({"1", "true", "yes", "on"})


def env_agent_model_debug() -> bool:
    """When set, print which models/clients are used (chat LLM, detectors, EQA VLM, etc.)."""
    v = os.environ.get("EMET_AGENT_MODEL_DEBUG", "").strip().lower()
    return v in _TRUE


def env_agent_camera_debug() -> bool:
    """When set, print per-frame stats for head cam / send_image / Discord (black-frame diagnosis)."""
    v = os.environ.get("EMET_AGENT_CAMERA_DEBUG", "").strip().lower()
    return v in _TRUE


def env_vram_debug() -> bool:
    """When set, print nvidia-smi + torch CUDA memory snapshots at major load milestones (see ``emet.utils.vram_debug``)."""
    v = os.environ.get("EMET_VRAM_DEBUG", "").strip().lower()
    return v in _TRUE
