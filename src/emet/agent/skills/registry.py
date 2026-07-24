# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Assemble mode-specific tool packs from the shared skill library."""

from __future__ import annotations

from typing import Any

from emet.agent.skills.modes import AgentMode
from emet.agent.tools import Tool


def build_skill_pack(
    mode: AgentMode,
    backend: Any,
    *,
    eqa_submode: str | None = None,
) -> list[Tool]:
    """Build the tool list for *mode*.

    * ``CHAT`` — *backend* is the chat ``context`` dict
      (``executor``, ``robot``, ``discord_bot``, …).
    * ``EQA_EPISODE`` — *backend* is an :class:`~emet.memory.graph_eqa.agentic_eqa.AgenticEQAExecutor`.
    """
    if mode is AgentMode.CHAT:
        from emet.agent.tools import build_chat_tools

        if not isinstance(backend, dict):
            raise TypeError(f"CHAT skill pack expects a context dict, got {type(backend)!r}")
        return build_chat_tools(backend)

    if mode is AgentMode.EQA_EPISODE:
        from emet.agent.skills.eqa import bind_eqa_episode_tools

        return bind_eqa_episode_tools(backend, eqa_submode=eqa_submode)

    raise ValueError(f"Unknown AgentMode: {mode!r}")
