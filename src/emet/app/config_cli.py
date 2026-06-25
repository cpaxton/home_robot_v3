# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared Click options for unified ``--config`` / ``--set`` across emet run apps."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import click
from click.core import ParameterSource

from emet.config.loader import (
    default_config_path,
    load_config,
    resolve_config_path_for_legacy_alias,
)
from emet.config.runtime import RuntimeContext, resolve_runtime_context
from emet.robots import DEFAULT_DYNAV_CONFIG_YAML

_WARNED_ALIASES: set[str] = set()


def _warn_deprecated_alias(old_flag: str, new_flag: str = "--config") -> None:
    if old_flag in _WARNED_ALIASES:
        return
    _WARNED_ALIASES.add(old_flag)
    warnings.warn(
        f"{old_flag} is deprecated; use {new_flag} (unified nested config).",
        DeprecationWarning,
        stacklevel=3,
    )


def emet_config_options(
    *,
    include_deprecated_aliases: bool = True,
    include_connection: bool = True,
) -> Callable:
    """Attach ``--config``, ``--set``, ``--connection``, and legacy aliases to a Click command."""

    def decorator(f: Callable) -> Callable:
        if include_connection:
            f = click.option(
                "--connection",
                default=None,
                help="Named profile from ~/.stretch/connection.json (host + robot hints).",
            )(f)
        if include_deprecated_aliases:
            f = click.option(
                "--dynav-config",
                "--dynav_config",
                "dynav_config",
                default=None,
                help="Deprecated alias for --config.",
            )(f)
            f = click.option(
                "--agent-config",
                "agent_config",
                default=None,
                help="Deprecated alias for --config.",
            )(f)
        f = click.option(
            "--set",
            "-O",
            "config_sets",
            multiple=True,
            help="Override nested config (e.g. mapping.depth_source=auto, agent.eqa=true).",
        )(f)
        f = click.option(
            "--config",
            "-C",
            "emet_config",
            default=default_config_path(),
            show_default=True,
            help="Unified nested config YAML (mapping, agent, sim, robots). Env: EMET_CONFIG.",
        )(f)
        return f

    return decorator


def resolve_effective_config_path(
    ctx: click.Context,
    *,
    emet_config: str,
    agent_config: str | None = None,
    dynav_config: str | None = None,
) -> str:
    """Pick config path; warn when legacy aliases are used."""
    if agent_config is not None and ctx.get_parameter_source("agent_config") != ParameterSource.DEFAULT:
        _warn_deprecated_alias("--agent-config")
        return agent_config
    if dynav_config is not None and ctx.get_parameter_source("dynav_config") != ParameterSource.DEFAULT:
        _warn_deprecated_alias("--dynav-config")
        return dynav_config
    if ctx.get_parameter_source("emet_config") == ParameterSource.DEFAULT:
        legacy_default = DEFAULT_DYNAV_CONFIG_YAML
        if emet_config == legacy_default or emet_config.endswith(legacy_default):
            return resolve_config_path_for_legacy_alias(legacy_default)
    return emet_config


def load_resolved_config(
    ctx: click.Context,
    *,
    emet_config: str,
    config_sets: tuple[str, ...] = (),
    agent_config: str | None = None,
    dynav_config: str | None = None,
    robot: str | None = None,
    robot_from_default: bool = True,
) -> Any:
    """Load :class:`~emet.config.loader.ResolvedEmetConfig` from CLI args."""
    path = resolve_effective_config_path(
        ctx,
        emet_config=emet_config,
        agent_config=agent_config,
        dynav_config=dynav_config,
    )
    return load_config(
        path,
        overrides=list(config_sets) if config_sets else None,
        robot=robot if not robot_from_default and robot else None,
    )


def load_runtime_from_cli(
    ctx: click.Context,
    *,
    emet_config: str,
    config_sets: tuple[str, ...] = (),
    agent_config: str | None = None,
    dynav_config: str | None = None,
    robot: str | None = None,
    robot_ip: str = "127.0.0.1",
    connection: str | None = None,
    port_offset: int = 0,
    zmq_timeout: float = 60.0,
    zmq_discover: bool = True,
    extra_mapping: dict[str, Any] | None = None,
) -> RuntimeContext:
    """Load config and resolve robot/host/parameters."""
    robot_from_default = robot is None or ctx.get_parameter_source("robot") == ParameterSource.DEFAULT
    host_from_default = ctx.get_parameter_source("robot_ip") == ParameterSource.DEFAULT

    cfg = load_resolved_config(
        ctx,
        emet_config=emet_config,
        config_sets=config_sets,
        agent_config=agent_config,
        dynav_config=dynav_config,
        robot=robot,
        robot_from_default=robot_from_default,
    )

    return resolve_runtime_context(
        cfg,
        cli_robot=robot,
        robot_from_default=robot_from_default,
        cli_host=robot_ip,
        host_from_default=host_from_default,
        connection_name=connection,
        port_offset=port_offset,
        zmq_timeout=zmq_timeout,
        zmq_discover=zmq_discover,
        extra_mapping=extra_mapping,
    )
