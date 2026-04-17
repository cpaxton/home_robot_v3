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

"""Shared keys and helpers for MuJoCo ZMQ sim ↔ agent messages."""

from __future__ import annotations

# Published on observation and state dicts so the client can detect Stretch vs rby1, etc.
EMET_ZMQ_ROBOT_ID_KEY = "emet_robot_id"

# Normalized ids that count as Hello Stretch for StretchZmqClient.
_STRETCH_FAMILY = frozenset({"stretch", "hello_stretch", "hellostretch"})


def normalize_robot_id(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def robot_ids_match(a: str, b: str) -> bool:
    return normalize_robot_id(a) == normalize_robot_id(b)


def is_stretch_family(name: str) -> bool:
    return normalize_robot_id(name) in _STRETCH_FAMILY


def read_emet_robot_id(msg: dict | None) -> str | None:
    if msg is None:
        return None
    v = msg.get(EMET_ZMQ_ROBOT_ID_KEY)
    if v is None:
        return None
    return str(v)
