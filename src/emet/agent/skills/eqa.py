# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bind EQA_EPISODE skill specs to an :class:`AgenticEQAExecutor`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emet.agent.skills.specs import SkillSpec, eqa_specs_for_submode
from emet.agent.tools import Tool

if TYPE_CHECKING:
    from emet.memory.graph_eqa import AgenticEQAExecutor


def _dispatch(executor: AgenticEQAExecutor, name: str):
    def _fn(**kwargs: Any) -> str:
        out = executor.handle_tool(name, kwargs)
        return str(out)

    return _fn


def bind_eqa_skill(executor: AgenticEQAExecutor, spec: SkillSpec) -> Tool:
    """Materialize one EQA skill as a :class:`Tool` (dispatch via ``handle_tool``)."""
    return Tool(
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        func=_dispatch(executor, spec.name),
        returns_info=spec.returns_info,
    )


def bind_eqa_episode_tools(executor: AgenticEQAExecutor, *, eqa_submode: str | None = None) -> list[Tool]:
    """Assemble the EQA_EPISODE tool pack (stable names/schemas for traces + prefix KV)."""
    submode = eqa_submode if eqa_submode is not None else getattr(executor, "mode", "answer")
    return [bind_eqa_skill(executor, spec) for spec in eqa_specs_for_submode(str(submode))]
