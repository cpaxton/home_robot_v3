# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Resolve ZMQ host and robot id from CLI flags and saved connection profiles."""

from __future__ import annotations

from emet.utils.connection import get_connection, get_host_from_connection


def resolve_cli_host(robot_ip: str, connection_name: str | None, *, ip_from_default: bool) -> str:
    if not ip_from_default and robot_ip.strip():
        return robot_ip.strip()
    if connection_name:
        host = get_host_from_connection(connection_name)
        if host:
            return host.strip()
    if ip_from_default:
        host = get_host_from_connection()
        if host:
            return host.strip()
    return robot_ip.strip() or "127.0.0.1"


def resolve_cli_robot(robot: str, connection_name: str | None, *, robot_from_default: bool) -> str:
    if not robot_from_default:
        return robot.lower().replace("-", "_")
    conn = get_connection(connection_name) if connection_name else get_connection()
    if conn and conn.get("robot"):
        return str(conn["robot"]).lower().replace("-", "_")
    return robot.lower().replace("-", "_")
