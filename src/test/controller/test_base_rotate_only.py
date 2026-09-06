# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""EMET_BASE_ROTATE_ONLY: yaw-only / no XY drive guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_base_rotate_only_stubs_drive_tools(monkeypatch):
    monkeypatch.setenv("EMET_BASE_ROTATE_ONLY", "1")
    from emet.agent.tools import get_tools

    tools = {t.name: t for t in get_tools({})}
    assert "rotate_base" in tools
    assert "scan_environment" in tools
    assert "describe_scene" in tools
    assert "move_forward" in tools  # still listed (stub) so the LLM does not invent a call
    assert "explore" in tools
    out = tools["move_forward"].func(meters=0.5)
    assert "can't drive" in out.lower() or "EMET_BASE_ROTATE_ONLY" in out
    assert tools["move_forward"].returns_info is True


def test_move_base_to_refuses_xy_when_rotate_only(monkeypatch):
    monkeypatch.setenv("EMET_BASE_ROTATE_ONLY", "1")
    from emet.controller.generic_zmq_client import GenericZmqClient

    client = MagicMock(spec=GenericZmqClient)
    client.get_base_pose.return_value = np.array([1.0, 2.0, 0.0])
    client.send_action = MagicMock()
    # Bind the real method
    client.move_base_to = GenericZmqClient.move_base_to.__get__(client, GenericZmqClient)

    assert client.move_base_to([0.0, 0.0, 0.5], relative=True, blocking=False) is True
    assert client.send_action.called
    sent = client.send_action.call_args[0][0]
    assert sent.get("nav_blocking") is False
    assert "nav_timeout_s" in sent
    assert sent.get("nav_relative") is True

    client.send_action.reset_mock()
    assert client.move_base_to([0.3, 0.0, 0.0], relative=True, blocking=False) is False
    assert not client.send_action.called

    assert client.move_base_to([5.0, 5.0, 0.0], relative=False, blocking=False) is False


def test_move_base_to_blocking_uses_command_receipt(monkeypatch):
    """Server nav stays non-blocking; only the matching receipt can establish arrival."""
    monkeypatch.delenv("EMET_BASE_ROTATE_ONLY", raising=False)
    from emet.controller.generic_zmq_client import GenericZmqClient

    client = MagicMock(spec=GenericZmqClient)
    client.send_action = MagicMock()
    client._nav_goal_reset_seen = MagicMock(return_value=True)
    client._wait_at_goal = MagicMock(return_value=True)
    client._robosuite_sim_zmq = MagicMock(return_value=False)
    client.move_base_to = GenericZmqClient.move_base_to.__get__(client, GenericZmqClient)

    with patch("emet.core.command_client.wait_navigation", return_value=True) as wait:
        assert client.move_base_to([0.0, 0.0, 0.5], relative=True, blocking=True, timeout=12.0) is True
    sent = client.send_action.call_args[0][0]
    assert sent.get("nav_blocking") is False
    assert sent.get("nav_timeout_s") == pytest.approx(12.0)
    client._wait_at_goal.assert_not_called()
    wait.assert_called_once_with(client, client.send_action.return_value, 12.0)
