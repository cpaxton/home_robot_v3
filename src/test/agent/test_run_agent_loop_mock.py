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
#
# Copyright (c) Hello Robot, Inc. All rights reserved.

"""Offline smoke: ``run_agent_with_robot`` with mocked ZMQ client and executor (no sim, no DynaMem load)."""

from unittest.mock import MagicMock, patch

from emet.agent.loop import run_agent_with_robot


class _DummyMemBackend:
    """Minimal backend so ``query_memory`` tool falls through to voxel localize_text."""

    def query_answer(self, *args, **kwargs):
        raise NotImplementedError


def _make_executor(robot, parameters, **kwargs):
    ex = MagicMock()
    ex.robot = robot
    ex.discord_bot = None
    ex._last_memory_save_path = None
    ex.agent = MagicMock()
    ex.agent.get_voxel_map.return_value = MagicMock()
    ex.agent.log = "."
    ex.agent.graph_memory = None
    ex.agent.robot = robot
    ex.agent.planner = None
    ex.__call__ = MagicMock(return_value=True)
    return ex


def test_run_agent_with_robot_quit_no_llm_no_discord():
    """Scripted QUIT exits cleanly; StretchZmqClient and DynamemTaskExecutor never touch real sim."""
    last_robot: dict[str, MagicMock] = {}

    def fake_stretch(*args, **kwargs):
        robot = MagicMock()
        robot.stop = MagicMock()
        robot.get_observation = MagicMock(return_value=None)
        last_robot["r"] = robot
        return robot

    with (
        patch("emet.agent.loop.StretchZmqClient", side_effect=fake_stretch),
        patch("emet.agent.loop.DynamemTaskExecutor", side_effect=_make_executor),
        patch("emet.agent.loop.get_memory_backend", return_value=_DummyMemBackend()),
        patch("emet.agent.loop.print_memory_view_help_on_quit"),
    ):
        run_agent_with_robot(
            robot_ip="127.0.0.1",
            robot="stretch",
            discord=False,
            use_llm=False,
            commands=["QUIT"],
            agent_config="dynav_config.yaml",
        )

    assert last_robot["r"].stop.call_count == 1
