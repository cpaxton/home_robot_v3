# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Resolve robot, host, and mapping parameters for ZMQ client apps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emet.app.robot_cli import discover_zmq_server_robot_id
from emet.config.loader import ResolvedEmetConfig
from emet.core.parameters import Parameters
from emet.robots import apply_robot_dynav_parameter_overrides
from emet.utils.connection import get_connection, get_host_from_connection

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass
class RuntimeContext:
    """Resolved robot connection + mapping parameters."""

    robot_id: str
    host: str
    config: ResolvedEmetConfig
    parameters: Parameters
    allow_missing_depth: bool
    config_path: str = ""
    robot_source: str = "default"
    host_source: str = "default"


def resolve_host(
    cli_host: str,
    *,
    host_from_default: bool,
    connection_name: str | None,
    config_connection: str | None = None,
) -> tuple[str, str]:
    """Return ``(host, source)`` — explicit CLI, connection profile, or fallback."""
    if not host_from_default and cli_host.strip():
        return cli_host.strip(), "cli"
    conn_name = connection_name or config_connection
    host = get_host_from_connection(conn_name)
    if host:
        return host.strip(), "connection"
    if host_from_default:
        active = get_host_from_connection()
        if active:
            return active.strip(), "connection_active"
    return (cli_host.strip() or "127.0.0.1"), "fallback"


def resolve_robot_id(
    cli_robot: str | None,
    *,
    robot_from_default: bool,
    config: ResolvedEmetConfig,
    connection_name: str | None,
    host: str,
    port_offset: int,
    zmq_timeout: float = 60.0,
    zmq_discover: bool = True,
) -> tuple[str, str]:
    """Return ``(robot_id, source)``.

    Precedence: CLI → config ``robot:`` → ZMQ (localhost sim) → connection profile → ZMQ → ``stretch``.

    On localhost, the running ZMQ server's ``emet_robot_id`` wins over a saved hardware
    connection profile (e.g. ``innate_mars`` on Herman) so ``emet run dynagraph`` works
    against a local ``emet serve mujoco`` without ``--robot``.
    """
    if cli_robot and not robot_from_default:
        return cli_robot.lower().replace("-", "_"), "cli"

    if config.robot:
        return config.robot.lower().replace("-", "_"), "config"

    host_norm = host.strip().lower()
    is_local = host_norm in _LOCALHOST_HOSTS

    def _discover(timeout: float) -> str | None:
        if not zmq_discover:
            return None
        rid = discover_zmq_server_robot_id(
            host,
            port_offset=port_offset,
            timeout=float(timeout),
            use_remote_computer=True,
        )
        return rid.lower().replace("-", "_") if rid else None

    if is_local:
        # Prefer sim truth; short timeout so a dead port fails fast to stretch.
        discovered = _discover(min(zmq_timeout, 8.0))
        if discovered:
            return discovered, "zmq"
        return "stretch", "default"

    conn_name = connection_name or config.connection
    conn = get_connection(conn_name) if conn_name else get_connection()
    if conn and conn.get("robot"):
        return str(conn["robot"]).lower().replace("-", "_"), "connection"

    discovered = _discover(zmq_timeout)
    if discovered:
        return discovered, "zmq"

    return "stretch", "default"


def apply_runtime_mapping_rules(
    parameters: dict[str, Any],
    *,
    robot_id: str,
    host: str,
    zmq_allow_missing_depth: bool | None = None,
) -> None:
    """In-place localhost depth promotion and allow_missing_depth defaults."""
    depth_mode = str(parameters.get("depth_source", "sensor")).lower()
    host_norm = host.strip().lower()
    if host_norm in _LOCALHOST_HOSTS and depth_mode == "da3":
        parameters["depth_source"] = "auto"

    if zmq_allow_missing_depth is not None:
        return

    # Caller sets allow_missing_depth separately from zmq section; this helper only adjusts depth_source.


def build_parameters_from_config(
    config: ResolvedEmetConfig,
    robot_id: str,
    *,
    host: str = "127.0.0.1",
    extra_mapping: dict[str, Any] | None = None,
) -> tuple[Parameters, bool]:
    """Build :class:`Parameters` with robot overlay and legacy code overrides applied."""
    merged_cfg = config.with_robot_overlay(robot_id)
    mapping = merged_cfg.mapping_dict
    if extra_mapping:
        for k, v in extra_mapping.items():
            mapping[k] = v

    apply_robot_dynav_parameter_overrides(robot_id, mapping)
    apply_runtime_mapping_rules(mapping, robot_id=robot_id, host=host)

    zmq_section = merged_cfg.zmq
    depth_mode = str(mapping.get("depth_source", "sensor")).lower()
    if zmq_section.allow_missing_depth is not None:
        allow_missing = bool(zmq_section.allow_missing_depth)
    else:
        allow_missing = depth_mode in ("da3", "auto", "lingbot") or robot_id in (
            "innate_mars",
            "galaxea_r1",
            "rby1",
            "stretch",
        )

    return Parameters(**mapping), allow_missing


def resolve_runtime_context(
    config: ResolvedEmetConfig,
    *,
    cli_robot: str | None = None,
    robot_from_default: bool = True,
    cli_host: str = "127.0.0.1",
    host_from_default: bool = True,
    connection_name: str | None = None,
    port_offset: int = 0,
    zmq_timeout: float = 60.0,
    zmq_discover: bool = True,
    extra_mapping: dict[str, Any] | None = None,
) -> RuntimeContext:
    """Single entry: resolve host, robot, and mapping parameters."""
    host, host_source = resolve_host(
        cli_host,
        host_from_default=host_from_default,
        connection_name=connection_name,
        config_connection=config.connection,
    )
    robot_id, robot_source = resolve_robot_id(
        cli_robot,
        robot_from_default=robot_from_default,
        config=config,
        connection_name=connection_name,
        host=host,
        port_offset=port_offset,
        zmq_timeout=zmq_timeout,
        zmq_discover=zmq_discover,
    )

    final_config = config.with_robot_overlay(robot_id)
    parameters, allow_missing = build_parameters_from_config(
        final_config,
        robot_id,
        host=host,
        extra_mapping=extra_mapping,
    )

    return RuntimeContext(
        robot_id=robot_id,
        host=host,
        config=final_config,
        parameters=parameters,
        allow_missing_depth=allow_missing,
        config_path=config.source_path,
        robot_source=robot_source,
        host_source=host_source,
    )
