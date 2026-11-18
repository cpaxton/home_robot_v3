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
"""Public agent API, loaded lazily to keep skill metadata independent of robot IO.

Importing a tool type or mode must not initialize mapping, perception, audio, or
controller modules. Public names retain their original identities once loaded.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "DEFAULT_AGENT_NAME": "prompt",
    "AgentPromptBuilder": "prompt",
    "build_agent_system_prompt": "prompt",
    "parse_tool_calls_response": "prompt",
    "Tool": "tool",
    "get_tool_descriptions_for_prompt": "tool",
    "get_tool_schemas_for_llm": "tools",
    "get_tools": "tools",
    "run_agent_with_robot": "loop",
}

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


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"emet.agent.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazily-exported names so introspection sees them before first access."""
    return sorted(set(globals()) | set(_EXPORT_MODULES))
