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


def env_confirm_nav() -> bool:
    """Require y/n (terminal or Discord) before executing motion plans.

    Set ``EMET_CONFIRM_NAV=1`` (or use ``emet run agent --confirm-nav``). Recommended on
    the real robot so operators can reject wall-hugging A* paths.
    """
    v = os.environ.get("EMET_CONFIRM_NAV", "").strip().lower()
    return v in _TRUE


def env_agent_motion_status() -> bool:
    """Fine-grained motion progress on the terminal (head sweep / rotate steps).

    Default **on**. Set ``EMET_AGENT_MOTION_STATUS=0`` to silence step-by-step lines
    (coarse Discord announcements from ``announce_action`` still apply).
    """
    v = os.environ.get("EMET_AGENT_MOTION_STATUS", "").strip().lower()
    if not v:
        return True
    return v in _TRUE


def env_vram_debug() -> bool:
    """When set, print nvidia-smi + torch CUDA memory snapshots at major load milestones (see ``emet.utils.vram_debug``)."""
    v = os.environ.get("EMET_VRAM_DEBUG", "").strip().lower()
    return v in _TRUE


def env_base_rotate_only() -> bool:
    """Hardware safety: allow in-place yaw only (no XY drive).

    Set ``EMET_BASE_ROTATE_ONLY=1`` when the robot is plugged in / tethered so
    ``explore`` / ``move_forward`` / absolute nav goals cannot translate the base.
    Relative ``[0, 0, yaw]`` moves (``rotate_base``, ``rotate_in_place``,
    ``scan_environment``) still work.
    """
    v = os.environ.get("EMET_BASE_ROTATE_ONLY", "").strip().lower()
    return v in _TRUE
