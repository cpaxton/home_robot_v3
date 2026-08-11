# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Closer-look / aim_arm_at helper unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from emet.agent.tools import get_tools
from emet.controller.manipulation.closer_look import (
    CloserLookResult,
    aim_wrist_at_phrase,
    consume_closer_look_aim_for_ee_picture,
    record_closer_look_aim,
)


def test_aim_empty_phrase():
    out = aim_wrist_at_phrase(agent=None, robot=None, phrase="")
    assert out.ok is False
    assert out.status_code == "empty_phrase"


def test_aim_localize_failed():
    agent = SimpleNamespace(voxel_map=None, get_voxel_map=lambda: None)
    out = aim_wrist_at_phrase(agent=agent, robot=None, phrase="red cup")
    assert out.ok is False
    assert out.status_code == "localize_failed"


def test_aim_localized_but_no_kinematic():
    class _VM:
        def localize_text(self, text, return_debug=True):
            return np.array([1.0, 2.0, 0.5])

    agent = SimpleNamespace(get_voxel_map=lambda: _VM(), voxel_map=_VM())
    out = aim_wrist_at_phrase(
        agent=agent,
        robot=SimpleNamespace(),
        phrase="red cup",
        manip_mode="teleport",
        visual_servo=False,
    )
    assert out.ok is False
    assert out.status_code == "not_implemented"
    assert out.xyz == (1.0, 2.0, 0.5)
    outcome = out.to_tool_outcome()
    assert outcome.tool == "aim_arm_at"
    assert outcome.ok is False


def test_ee_picture_requires_successful_aim():
    agent = SimpleNamespace()
    ctx: dict = {"agent": agent, "robot": None}
    tools = {t.name: t for t in get_tools(ctx)}
    out = tools["take_ee_picture"].func()
    assert out.ok is False
    assert out.status == "aim_required"


def test_ee_picture_soft_allows_after_not_implemented_aim():
    agent = SimpleNamespace()
    ctx: dict = {"agent": agent}
    record_closer_look_aim(
        CloserLookResult(
            False,
            "not_implemented",
            "no kinematic aim",
            xyz=(1.0, 0.0, 0.5),
            phrase="mug",
        ),
        agent=agent,
        context=ctx,
    )
    allowed, note, payload = consume_closer_look_aim_for_ee_picture(agent=agent, context=ctx)
    assert allowed is True
    assert payload is not None
    assert "not available" in note.lower() or "aim not" in note.lower()
    # Soft-allow is one-shot.
    allowed2, _, _ = consume_closer_look_aim_for_ee_picture(agent=agent, context=ctx)
    assert allowed2 is False


def test_ee_picture_allowed_once_after_aim():
    agent = SimpleNamespace()
    ctx: dict = {"agent": agent}
    record_closer_look_aim(
        CloserLookResult(True, "ok", "aimed", xyz=(1.0, 0.0, 0.5), phrase="mug"),
        agent=agent,
        context=ctx,
    )
    allowed, note, aim = consume_closer_look_aim_for_ee_picture(agent=agent, context=ctx)
    assert allowed is True
    assert aim is not None and aim["phrase"] == "mug"
    assert "mug" in note
    # Second consume refuses (grant spent).
    allowed2, _, _ = consume_closer_look_aim_for_ee_picture(agent=agent, context=ctx)
    assert allowed2 is False


def test_take_ee_picture_tool_queues_after_aim():
    robot = MagicMock()
    robot.get_servo_observation.return_value = SimpleNamespace(ee_rgb=np.zeros((8, 8, 3), dtype=np.uint8))
    agent = SimpleNamespace()
    ctx: dict = {"agent": agent, "robot": robot}
    record_closer_look_aim(
        CloserLookResult(True, "ok", "aimed", phrase="cup"),
        agent=agent,
        context=ctx,
    )
    tools = {t.name: t for t in get_tools(ctx)}
    out = tools["take_ee_picture"].func()
    assert out.ok is True
    assert ctx.get("pending_discord_image") is not None
    # Grant consumed — second call refuses.
    out2 = tools["take_ee_picture"].func()
    assert out2.ok is False
    assert out2.status == "aim_required"
