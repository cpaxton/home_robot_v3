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

"""Read simulator GT lists attached to ZMQ observation dicts."""

from __future__ import annotations

from typing import Any

from emet.core.zmq_protocol import EMET_ZMQ_GT_OBJECTS_KEY


def read_gt_object_dicts_from_robot_client(robot: Any) -> list[dict[str, Any]]:
    """Return ``emet_gt_objects`` from the last full observation, if the client exposes ``_obs``."""
    lock = getattr(robot, "_obs_lock", None)
    if lock is None:
        obs = getattr(robot, "_obs", None)
        return _coerce_gt_list(obs)
    with lock:
        obs = getattr(robot, "_obs", None)
        return _coerce_gt_list(obs)


def _coerce_gt_list(obs: Any) -> list[dict[str, Any]]:
    if not isinstance(obs, dict):
        return []
    raw = obs.get(EMET_ZMQ_GT_OBJECTS_KEY)
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out
