# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Closer-look / aim_arm_at helper unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.controller.manipulation.closer_look import aim_wrist_at_phrase


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
