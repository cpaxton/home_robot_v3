# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent package: tools, prompt, and loop for robot + memory + optional Discord.
#
# ``run_agent_with_robot`` is loaded lazily (``__getattr__``) so submodules like
# ``emet.agent.env_flags`` can be imported while ``emet.controller.controller_dynamem``
# is still initializing — otherwise ``agent`` → ``loop`` → ``dynamem_task`` → ``RobotAgent``
# creates a circular import.

from emet.agent.prompt import (
    DEFAULT_AGENT_NAME,
    AgentPromptBuilder,
    build_agent_system_prompt,
    parse_tool_calls_response,
)
from emet.agent.tools import (
    Tool,
    get_tool_descriptions_for_prompt,
    get_tool_schemas_for_llm,
    get_tools,
)

__all__ = [
    "DEFAULT_AGENT_NAME",
    "AgentPromptBuilder",
    "Tool",
    "build_agent_system_prompt",
    "get_tool_descriptions_for_prompt",
    "get_tool_schemas_for_llm",
    "get_tools",
    "parse_tool_calls_response",
    "run_agent_with_robot",
]


def __getattr__(name: str):
    if name == "run_agent_with_robot":
        from emet.agent.loop import run_agent_with_robot as _run_agent_with_robot

        globals()["run_agent_with_robot"] = _run_agent_with_robot
        return _run_agent_with_robot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
