# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Motion progress announcements (terminal vs Discord)."""

from __future__ import annotations

from unittest.mock import MagicMock

from emet.controller.base_controller import BaseController


class _Agent(BaseController):
    def get_voxel_map(self):
        return None


def test_announce_motion_progress_skips_discord(monkeypatch):
    monkeypatch.delenv("EMET_AGENT_MOTION_STATUS", raising=False)
    robot = MagicMock()
    agent = _Agent.__new__(_Agent)
    agent.robot = robot
    discord = MagicMock()
    agent.discord_bot = discord
    agent.announce_motion_progress("Look around: head pan 1/4")
    discord.push_task_to_all_channels.assert_not_called()


def test_announce_motion_progress_respects_env_off(monkeypatch):
    monkeypatch.setenv("EMET_AGENT_MOTION_STATUS", "0")
    agent = _Agent.__new__(_Agent)
    agent.robot = MagicMock()
    agent.discord_bot = None
    cb = MagicMock()
    agent._progress_callback = cb
    agent.announce_motion_progress("Look around: head pan 1/4")
    cb.assert_not_called()


def test_announce_action_can_skip_discord():
    agent = _Agent.__new__(_Agent)
    agent.robot = MagicMock()
    discord = MagicMock()
    agent.discord_bot = discord
    agent.announce_action("Look around: sweeping head", discord=False)
    discord.push_task_to_all_channels.assert_not_called()
    agent.announce_action("Look around: sweeping head", discord=True)
    discord.push_task_to_all_channels.assert_called_once()
