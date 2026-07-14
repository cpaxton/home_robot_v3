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

from typing import Any

# Published on observation and state dicts so the client can detect Stretch vs rby1, etc.
EMET_ZMQ_ROBOT_ID_KEY = "emet_robot_id"

# Frozen per-process metadata on every outbound ZMQ dict (obs / state / servo). See docs/zmq_session_metadata.md.
EMET_ZMQ_SESSION_KEY = "emet_session"
EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY = "schema_version"
CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION = 1

# Optional sim introspection: client sends ``{EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY: {...}}``.
EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY = "mujoco_ground_truth_dump"
# Sim OVMM full task / benchmarks: teleport a freejoint object body (pick/place proxy in MuJoCo).
EMET_ACTION_SIM_SET_BODY_POSE_KEY = "sim_set_body_pose"
# Dynamic-world benchmarks: set a named scene hinge/slide joint (e.g. Robocasa cabinet door).
EMET_ACTION_SIM_SET_JOINT_QPOS_KEY = "sim_set_joint_qpos"

# ZMQ recv actions that must bypass duplicate ``step`` filtering (see ``BaseZmqServer.spin_recv``).
EMET_ZMQ_META_ACTION_KEYS: frozenset[str] = frozenset(
    {
        EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY,
        EMET_ACTION_SIM_SET_BODY_POSE_KEY,
        EMET_ACTION_SIM_SET_JOINT_QPOS_KEY,
    }
)


def zmq_meta_action_should_bypass_duplicate_step(action: dict[str, Any]) -> bool:
    """True when *action* includes a meta command alongside ``step`` repeats."""
    if not action:
        return False
    return not EMET_ZMQ_META_ACTION_KEYS.isdisjoint(action.keys())


def build_mujoco_ground_truth_dump_action(
    step: int,
    path_on_sim_host: str,
    *,
    exclude_robot: bool = True,
    as_json: bool = False,
) -> dict[str, Any]:
    """Build a recv action dict instructing MuJoCo ZMQ servers to write a body pose snapshot."""

    payload: dict[str, Any] = {
        "path": str(path_on_sim_host),
        "exclude_robot": bool(exclude_robot),
        "json": bool(as_json),
    }
    return {"step": int(step), EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY: payload}


def build_sim_set_body_pose_action(
    step: int,
    body: str,
    pos: list[float] | tuple[float, float, float],
    *,
    quat: list[float] | tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Build recv action to teleport a sim freejoint body (OVMM pick/place proxy)."""
    payload: dict[str, Any] = {
        "body": str(body),
        "pos": [float(x) for x in pos[:3]],
    }
    if quat is not None:
        payload["quat"] = [float(x) for x in quat[:4]]
    return {"step": int(step), EMET_ACTION_SIM_SET_BODY_POSE_KEY: payload}


def build_sim_set_joint_qpos_action(step: int, joint: str, value: float) -> dict[str, Any]:
    """Build recv action to set a named scene hinge/slide joint qpos (doors, drawers)."""
    payload: dict[str, Any] = {"joint": str(joint), "value": float(value)}
    return {"step": int(step), EMET_ACTION_SIM_SET_JOINT_QPOS_KEY: payload}

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


def read_emet_robot_id_from_message_or_session(msg: dict[str, Any] | None) -> str | None:
    """Top-level ``emet_robot_id``, or the same key inside ``emet_session`` (newer servers)."""
    rid = read_emet_robot_id(msg)
    if rid:
        return rid
    sess = read_emet_session(msg)
    if sess is None:
        return None
    v = sess.get(EMET_ZMQ_ROBOT_ID_KEY)
    if v is None:
        return None
    return str(v)


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


def emet_session_manipulation_supported(session: dict[str, Any] | None) -> bool:
    """True when the ZMQ server advertises arm/gripper/posture control.

    Navigation-only sim backends (e.g. Habitat HM-EQA serve) set ``capabilities.manipulation=false``.
    When session is unknown, assume manipulation is available (real Stretch / full MuJoCo sim).
    """
    if session is None:
        return True
    caps = session.get("capabilities")
    if isinstance(caps, dict) and "manipulation" in caps:
        return bool(caps["manipulation"])
    if session.get("runtime_kind") == "habitat_hmeqa":
        return False
    return True
