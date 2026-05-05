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
# Do not import ``emet.agent.loop`` here: it pulls DynaMem and would circular-import with
# ``controller_dynamem`` (loop → dynamem_task → controller_dynamem → env_flags under emet.agent).
# Use ``from emet.agent.loop import run_agent_with_robot`` at call sites.

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
]
