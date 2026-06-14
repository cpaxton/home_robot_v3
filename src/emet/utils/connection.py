# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Load/save robot connection profiles for deploy, Mars bridge, and ZMQ clients.

Profiles live under ``~/.stretch/`` (shared with legacy memory paths):

``connection.json`` schema::

    {
      "active": "herman",          // name of the default profile (or null)
      "connections": {
        "herman": {
          "host": "192.168.1.42",  // required — robot IP or hostname
          "user": "root",          // SSH user (default "root" when saving)
          "password": "...",       // optional — omit to use SSH keys / EMET_ROBOT_PASSWORD
          "robot": "innate_mars",  // optional — emet robot id for ``--robot`` defaults
          "workspace": "~/innate-os/ros2_ws",  // optional — remote ROS2 workspace (Mars deploy)
          "emet_dir": "~/emet"     // optional — remote emet_core install root
        }
      }
    }

Saving or activating a profile also writes ``~/.stretch/robot_ip.txt`` with the host so
:func:`emet.utils.memory.lookup_address` and other legacy tools keep working.
"""

from __future__ import annotations

import json
import os
from typing import Any

_STRETCH_DIR = os.path.expanduser("~/.stretch")
_CONNECTION_FILE = os.path.join(_STRETCH_DIR, "connection.json")
_ROBOT_IP_FILE = os.path.join(_STRETCH_DIR, "robot_ip.txt")

_DEFAULT_CONFIG: dict[str, Any] = {"active": None, "connections": {}}


def _ensure_stretch_dir() -> None:
    os.makedirs(_STRETCH_DIR, exist_ok=True)


def _load_config() -> dict[str, Any]:
    _ensure_stretch_dir()
    if not os.path.exists(_CONNECTION_FILE):
        return dict(_DEFAULT_CONFIG)
    try:
        with open(_CONNECTION_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(_DEFAULT_CONFIG)
        data.setdefault("active", None)
        data.setdefault("connections", {})
        return data
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_CONFIG)


def _save_config(config: dict[str, Any]) -> None:
    _ensure_stretch_dir()
    with open(_CONNECTION_FILE, "w") as f:
        json.dump(config, f, indent=2)


def _strip(value: str) -> str:
    return value.strip()


def _sync_robot_ip_file(host: str) -> None:
    """Keep legacy ``robot_ip.txt`` aligned with the active profile host."""
    host = _strip(host)
    if not host:
        return
    _ensure_stretch_dir()
    with open(_ROBOT_IP_FILE, "w") as f:
        f.write(host)


def _build_connection_record(
    *,
    host: str,
    user: str,
    password: str | None = None,
    workspace: str | None = None,
    emet_dir: str | None = None,
    robot: str | None = None,
) -> dict[str, Any]:
    conn: dict[str, Any] = {"host": _strip(host), "user": _strip(user)}
    for key, value in (
        ("password", password),
        ("workspace", workspace),
        ("emet_dir", emet_dir),
        ("robot", robot),
    ):
        if value is not None:
            conn[key] = _strip(str(value))
    return conn


def get_active_connection() -> dict[str, Any] | None:
    """Return the active connection dict, or ``None`` if none is set."""
    config = _load_config()
    active_name = config.get("active")
    if not active_name:
        return None
    return config.get("connections", {}).get(active_name)


def get_connection(name: str | None = None) -> dict[str, Any] | None:
    """Return a saved profile by ``name``, or the active profile when ``name`` is ``None``."""
    config = _load_config()
    conns = config.get("connections", {})
    if name:
        return conns.get(name)
    active_name = config.get("active")
    if active_name:
        return conns.get(active_name)
    return None


def list_connections() -> list[tuple[str, bool]]:
    """Return ``[(profile_name, is_active), ...]`` sorted by name."""
    config = _load_config()
    active_name = config.get("active")
    conns = config.get("connections", {})
    return [(n, n == active_name) for n in sorted(conns)]


def save_connection(
    host: str,
    user: str = "root",
    password: str | None = None,
    name: str | None = None,
    set_active: bool = True,
    *,
    workspace: str | None = None,
    emet_dir: str | None = None,
    robot: str | None = None,
) -> str:
    """Create or update a connection profile on disk.

    Args:
        host: Robot IP or hostname (required). Also written to ``~/.stretch/robot_ip.txt``.
        user: SSH login user. Default ``"root"``.
        password: Optional SSH password stored in the profile. Omit for key-based auth;
            callers may still pass ``EMET_ROBOT_PASSWORD`` at runtime without persisting it.
        name: Profile name in ``connection.json``. Default: ``host`` (after strip).
        set_active: When ``True`` (default), mark this profile as ``active`` for commands that
            omit ``--robot-ip`` / ``--host`` (``emet deploy``, ``emet mars start``, etc.).
        workspace: Remote ROS2 workspace path (e.g. ``~/innate-os/ros2_ws`` for innate Mars).
            Used by :func:`emet.deploy.deploy` and :func:`emet.mars.resolve_mars_target`.
        emet_dir: Remote directory for ``emet_core`` + bridge env (default ``~/emet`` on read).
        robot: Emet robot id stored in the profile (e.g. ``innate_mars``) for CLI defaults.

    Returns:
        The profile name used (``name`` or stripped ``host``).

    Note:
        Optional fields are omitted from JSON when not provided, except ``host`` and ``user``.
        Updating an existing ``name`` replaces that entry in place.
    """
    config = _load_config()
    conn_name = _strip(name or host)
    conn = _build_connection_record(
        host=host,
        user=user,
        password=password,
        workspace=workspace,
        emet_dir=emet_dir,
        robot=robot,
    )
    config.setdefault("connections", {})[conn_name] = conn
    if set_active:
        config["active"] = conn_name
        _sync_robot_ip_file(conn["host"])
    _save_config(config)
    return conn_name


def set_active(name: str) -> bool:
    """Set ``active`` to an existing profile name. Returns ``False`` if ``name`` is unknown."""
    config = _load_config()
    conns = config.get("connections", {})
    if name not in conns:
        return False
    config["active"] = name
    _save_config(config)
    host = conns[name].get("host", "")
    _sync_robot_ip_file(str(host))
    return True


def delete_connection(name: str) -> bool:
    """Remove a profile. Clears ``active`` if it pointed at ``name``. Returns ``False`` if missing."""
    config = _load_config()
    conns = config.get("connections", {})
    if name not in conns:
        return False
    del conns[name]
    if config.get("active") == name:
        config["active"] = None
    _save_config(config)
    return True


def get_host_from_connection(name: str | None = None) -> str | None:
    """Return the ``host`` field for ``name`` or the active profile."""
    conn = get_connection(name)
    if conn is None:
        return None
    host = conn.get("host")
    return str(host) if host else None
