# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.
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

from typing import Any

# Published on observation and state dicts so the client can detect Stretch vs rby1, etc.
EMET_ZMQ_ROBOT_ID_KEY = "emet_robot_id"

# Frozen per-process metadata on every outbound ZMQ dict (obs / state / servo). See docs/zmq_session_metadata.md.
EMET_ZMQ_SESSION_KEY = "emet_session"
EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY = "schema_version"
CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION = 1

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


def read_emet_session(msg: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the ``emet_session`` dict from a ZMQ message, or ``None`` if missing or invalid."""
    if msg is None:
        return None
    raw = msg.get(EMET_ZMQ_SESSION_KEY)
    if not isinstance(raw, dict):
        return None
    return raw


def emet_session_has_current_schema(session: dict[str, Any] | None) -> bool:
    if session is None:
        return False
    v = session.get(EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY)
    try:
        return int(v) == CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION
    except (TypeError, ValueError):
        return False


def emet_session_cache_update(
    cached: dict[str, Any] | None,
    cached_step: int,
    msg: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, int]:
    """Merge server ``emet_session`` into a client-side cache, preferring the message with the higher ``step``."""
    if msg is None:
        return cached, cached_step
    sess = read_emet_session(msg)
    if sess is None or not emet_session_has_current_schema(sess):
        return cached, cached_step
    try:
        step = int(msg.get("step", -1))
    except (TypeError, ValueError):
        step = -1
    if cached is None or step >= cached_step:
        return dict(sess), step
    return cached, cached_step
