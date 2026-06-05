# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.app.run_interactive import PickPlacePromptState, parse_task_mode_line


def test_task_mode_manip_requires_m():
    state = PickPlacePromptState(target_object="mug", target_receptacle="table")
    cmds = parse_task_mode_line("m", list_objects=None, pick_place=state)
    assert cmds == [("pickup", "mug"), ("place", "table")]
    assert parse_task_mode_line("hello", list_objects=None, pick_place=PickPlacePromptState()) is None


def test_task_mode_quit_aliases():
    state = PickPlacePromptState()
    for line in ("", "q", "Q", "quit", "QUIT"):
        assert parse_task_mode_line(line, list_objects=None, pick_place=state) == [("quit", "")]


def test_task_mode_explore_aliases():
    state = PickPlacePromptState()
    for line in ("e", "E", "explore", "map", "nav"):
        assert parse_task_mode_line(line, list_objects=None, pick_place=state) == [("explore", "")]


def test_task_mode_list_requires_callback():
    state = PickPlacePromptState()
    listed: list[str] = []

    def _list() -> None:
        listed.append("ok")

    assert parse_task_mode_line("l", list_objects=_list, pick_place=state) is None
    assert listed == ["ok"]
    assert parse_task_mode_line("l", list_objects=None, pick_place=state) is None
