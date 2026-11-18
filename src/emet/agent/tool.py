# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Tool value object and prompt rendering shared by CHAT and EQA.

This module depends only on the standard library. Schemas/descriptions belong
in ``skills.specs``; handlers and runtime capability checks belong in their
mode-specific binders. ``emet.agent.tools`` re-exports these names for callers
using the original import path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


class Tool:
    """A single agent tool: name, description, JSON Schema parameters, callable, and executor mapping.

    ``func`` is the direct handler; ``executor_commands`` optionally maps a call
    to legacy executor tuples instead. CHAT's dispatcher uses ``returns_info``
    to feed successful information back to the model; failed or unverified
    outcomes also request feedback. That flag does not establish success.
    This object serializes metadata but does not validate physical preconditions
    or guarantee that the session supports the requested action.
    """

    __slots__ = ("name", "description", "parameters", "func", "executor_commands", "returns_info")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
        executor_commands: Callable[[dict[str, Any]], list[tuple[str, str]]] | None = None,
        returns_info: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.executor_commands = executor_commands
        self.returns_info = returns_info

    def schema(self) -> dict[str, Any]:
        """Return OpenAI-style tool schema (for tools param or prompt generation)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_executor(self, arguments: dict[str, Any]) -> list[tuple[str, str]]:
        """Map tool call arguments to executor (command, args) list. Empty list means use func instead."""
        if self.executor_commands is not None:
            return self.executor_commands(arguments)
        return []


def get_tool_descriptions_for_prompt(tools: list[Tool]) -> str:
    """Build a compact tools block for the system prompt from Tool objects."""
    lines = ["TOOLS (use these names in tool_calls):"]
    for t in tools:
        props = t.parameters.get("properties", {})
        if props:
            arg_names = ", ".join(props.keys())
            lines.append(f"- {t.name}({arg_names}): {t.description}")
            # Local text-only clients do not receive the separate API tools
            # parameter. Preserve types, optionality, nested bounds and meanings.
            lines.append("  Arguments (JSON Schema): " + json.dumps(t.parameters, separators=(",", ":")))
        else:
            lines.append(f"- {t.name}(): {t.description}")
    return "\n".join(lines)
