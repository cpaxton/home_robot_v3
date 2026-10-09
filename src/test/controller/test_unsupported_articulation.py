# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import Mock

import pytest

from emet.controller.controller_instance_memory import RobotAgent
from emet.utils.llm_plan_wrapper import LLMPlanWrapper


@pytest.mark.parametrize("action", ["open_cabinet", "close_cabinet"])
def test_unsupported_articulation_cannot_report_success(action):
    agent = object.__new__(RobotAgent)
    agent.robot = Mock()
    assert getattr(agent, action)("cabinet") is False
    wrapper = LLMPlanWrapper(agent, "")
    assert getattr(wrapper, action)() is False
    assert agent.robot.mock_calls == []
