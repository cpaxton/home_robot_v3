# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Load/save robot connection profiles (host, user, optional password) for deploy and viewer."""

import json
import os
from typing import Any, Optional

# Reuse ~/.stretch from memory so connection and robot_ip live together
_STRETCH_DIR = os.path.expanduser("~/.stretch")
_CONNECTION_FILE = os.path.join(_STRETCH_DIR, "connection.json")


def _ensure_stretch_dir() -> None:
    if not os.path.exists(_STRETCH_DIR):
        os.makedirs(_STRETCH_DIR)


def _load_config() -> dict[str, Any]:
    _ensure_stretch_dir()
    if not os.path.exists(_CONNECTION_FILE):
        return {"active": None, "connections": {}}
    try:
        with open(_CONNECTION_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"active": None, "connections": {}}


def _save_config(config: dict[str, Any]) -> None:
    _ensure_stretch_dir()
    with open(_CONNECTION_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_active_connection() -> Optional[dict[str, Any]]:
    """Return the active connection dict (host, user, password if set) or None."""
    config = _load_config()
    active_name = config.get("active")
    if not active_name:
        return None
    conns = config.get("connections", {})
    return conns.get(active_name)


def get_connection(name: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return connection by name, or active connection if name is None."""
    config = _load_config()
    conns = config.get("connections", {})
    if name:
        return conns.get(name)
    active_name = config.get("active")
    if active_name:
        return conns.get(active_name)
    return None


def list_connections() -> list[tuple[str, bool]]:
    """Return list of (name, is_active) for all saved connections."""
    config = _load_config()
    active_name = config.get("active")
    conns = config.get("connections", {})
    return [(n, n == active_name) for n in sorted(conns)]


def save_connection(
    host: str,
    user: str = "root",
    password: Optional[str] = None,
    name: Optional[str] = None,
    set_active: bool = True,
) -> str:
    """Save a connection. Returns the name used (name or derived from host)."""
    _ensure_stretch_dir()
    config = _load_config()
    conn_name = name or host
    conn: dict[str, Any] = {"host": host.strip(), "user": user.strip()}
    if password is not None:
        conn["password"] = password
    if "connections" not in config:
        config["connections"] = {}
    config["connections"][conn_name] = conn
    if set_active:
        config["active"] = conn_name
    _save_config(config)
    # Keep legacy robot_ip.txt in sync so lookup_address() still works
    robot_ip_file = os.path.join(_STRETCH_DIR, "robot_ip.txt")
    with open(robot_ip_file, "w") as f:
        f.write(host.strip())
    return conn_name


def set_active(name: str) -> bool:
    """Set the active connection by name. Returns True if name exists."""
    config = _load_config()
    if name not in config.get("connections", {}):
        return False
    config["active"] = name
    host = config["connections"][name].get("host", "")
    _save_config(config)
    robot_ip_file = os.path.join(_STRETCH_DIR, "robot_ip.txt")
    if host:
        with open(robot_ip_file, "w") as f:
            f.write(host)
    return True


def delete_connection(name: str) -> bool:
    """Remove a connection by name. If it was active, clear active. Returns True if existed."""
    config = _load_config()
    conns = config.get("connections", {})
    if name not in conns:
        return False
    del conns[name]
    if config.get("active") == name:
        config["active"] = None
    _save_config(config)
    return True


def get_host_from_connection(name: Optional[str] = None) -> Optional[str]:
    """Convenience: return host string for the given or active connection."""
    conn = get_connection(name)
    if conn is None:
        return None
    return conn.get("host")
