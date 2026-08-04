# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for runtime context resolution."""

from __future__ import annotations

from unittest.mock import patch

from emet.config.loader import ResolvedEmetConfig, default_config_path, load_config
from emet.config.runtime import (
    build_parameters_from_config,
    resolve_robot_id,
    resolve_runtime_context,
)


def test_localhost_da3_promoted_to_auto():
    cfg = load_config(default_config_path())
    raw = cfg.raw
    raw.setdefault("mapping", {})["depth_source"] = "da3"
    resolved = ResolvedEmetConfig(raw=raw)
    params, _ = build_parameters_from_config(resolved, "stretch", host="127.0.0.1")
    assert params.get("depth_source") == "auto"


def test_innate_mars_allow_missing_depth_from_robot_overlay():
    cfg = load_config(default_config_path())
    ctx = resolve_runtime_context(
        cfg,
        cli_robot="innate_mars",
        robot_from_default=False,
        cli_host="192.168.1.10",
        host_from_default=False,
        connection_name=None,
        port_offset=0,
        zmq_discover=False,
    )
    assert ctx.allow_missing_depth is True
    assert ctx.parameters.get("depth_source") == "auto"


def test_robot_precedence_config_over_default():
    cfg = load_config(default_config_path())
    cfg.raw["robot"] = "rby1"
    rid, source = resolve_robot_id(
        None,
        robot_from_default=True,
        config=cfg,
        connection_name=None,
        host="127.0.0.1",
        port_offset=0,
        zmq_discover=False,
    )
    assert rid == "rby1"
    assert source == "config"


def test_robot_zmq_discovery_when_unset():
    cfg = load_config(default_config_path())
    with patch("emet.config.runtime.get_connection", return_value={"robot": "innate_mars", "host": "herman"}):
        with patch("emet.config.runtime.discover_zmq_server_robot_id", return_value="stretch"):
            rid, source = resolve_robot_id(
                None,
                robot_from_default=True,
                config=cfg,
                connection_name=None,
                host="127.0.0.1",
                port_offset=0,
                zmq_discover=True,
            )
    assert rid == "stretch"
    assert source == "zmq"


def test_localhost_zmq_wins_over_profile_config_robot():
    """Active Herman config YAML must not force innate_mars against local Stretch sim."""
    cfg = load_config("configs/agent_innate_mars.yaml")
    assert cfg.robot == "innate_mars"
    with patch("emet.config.runtime.discover_zmq_server_robot_id", return_value="stretch"):
        rid, source = resolve_robot_id(
            None,
            robot_from_default=True,
            config=cfg,
            connection_name=None,
            host="127.0.0.1",
            port_offset=0,
            zmq_discover=True,
        )
    assert rid == "stretch"
    assert source == "zmq"


def test_localhost_skips_connection_robot_without_zmq():
    cfg = load_config(default_config_path())
    with patch("emet.config.runtime.get_connection", return_value={"robot": "innate_mars", "host": "herman"}):
        with patch("emet.config.runtime.discover_zmq_server_robot_id", return_value=None):
            rid, source = resolve_robot_id(
                None,
                robot_from_default=True,
                config=cfg,
                connection_name=None,
                host="127.0.0.1",
                port_offset=0,
                zmq_discover=True,
            )
    assert rid == "stretch"
    assert source == "default"


def test_set_override_wins_over_robot_overlay():
    cfg = load_config(default_config_path())
    ctx = resolve_runtime_context(
        cfg,
        cli_robot="innate_mars",
        robot_from_default=False,
        cli_host="192.168.1.10",
        host_from_default=False,
        connection_name=None,
        port_offset=0,
        zmq_discover=False,
        overrides=["mapping.depth_source=sensor"],
    )
    assert ctx.parameters.get("depth_source") == "sensor"
    assert ctx.parameters.get("local_radius") == 0.85


def test_get_parameters_applies_robot_from_preset():
    from emet.core.parameters import get_parameters

    params = get_parameters("configs/agent_innate_mars.yaml")
    assert params.get("depth_source") == "auto"
    assert params.get("local_radius") == 0.85


def test_remote_host_uses_connection_robot_before_zmq():
    cfg = load_config(default_config_path())
    with patch("emet.config.runtime.get_connection", return_value={"robot": "innate_mars", "host": "192.168.1.42"}):
        with patch("emet.config.runtime.discover_zmq_server_robot_id") as discover:
            rid, source = resolve_robot_id(
                None,
                robot_from_default=True,
                config=cfg,
                connection_name=None,
                host="192.168.1.42",
                port_offset=0,
                zmq_discover=True,
            )
    assert rid == "innate_mars"
    assert source == "connection"
    discover.assert_not_called()


def test_resolve_host_connection_wins_when_robot_ip_is_default():
    """``emet run`` must not inject default ``--robot_ip`` or connection profiles lose."""
    from emet.config.runtime import resolve_host

    with patch(
        "emet.config.runtime.get_host_from_connection",
        return_value="192.168.1.43",
    ):
        host, source = resolve_host(
            "127.0.0.1",
            host_from_default=True,
            connection_name="herman",
        )
    assert host == "192.168.1.43"
    assert source == "connection"


def test_resolve_host_explicit_cli_beats_connection():
    from emet.config.runtime import resolve_host

    with patch(
        "emet.config.runtime.get_host_from_connection",
        return_value="192.168.1.43",
    ):
        host, source = resolve_host(
            "10.0.0.9",
            host_from_default=False,
            connection_name="herman",
        )
    assert host == "10.0.0.9"
    assert source == "cli"
