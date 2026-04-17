# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for no-LLM find command parsing (scripted --command)."""

from emet.agent.loop import parse_manual_find_command


def test_parse_manual_find_command_accepts_find_prefix():
    assert parse_manual_find_command("FIND red cylinder") == "red cylinder"
    assert parse_manual_find_command("find blue cube") == "blue cube"
    assert parse_manual_find_command("Find red cylinder") == "red cylinder"
    assert parse_manual_find_command("  find  red cylinder  ") == "red cylinder"


def test_parse_manual_find_command_accepts_f_shortcut():
    assert parse_manual_find_command("F red cylinder") == "red cylinder"


def test_parse_manual_find_command_rejects_non_find():
    assert parse_manual_find_command("hello") is None
    assert parse_manual_find_command("") is None
