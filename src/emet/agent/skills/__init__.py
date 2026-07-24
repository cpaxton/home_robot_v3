# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared skill library for CHAT and EQA_EPISODE agent modes.

Same robot memory and motion/perception capabilities; different orchestrators assemble
different tool packs and prompts. See :class:`~emet.agent.skills.modes.AgentMode`.
"""

from emet.agent.skills.modes import AgentMode
from emet.agent.skills.registry import build_skill_pack
from emet.agent.skills.specs import (
    CHAT_EXCLUSIVE_TOOL_NAMES,
    EQA_EXCLUSIVE_TOOL_NAMES,
    EQA_SKILL_SPECS,
    SHARED_SKILL_ALIASES,
    SkillSpec,
    eqa_specs_for_submode,
    skill_names_for_mode,
)

__all__ = [
    "AgentMode",
    "CHAT_EXCLUSIVE_TOOL_NAMES",
    "EQA_EXCLUSIVE_TOOL_NAMES",
    "EQA_SKILL_SPECS",
    "SHARED_SKILL_ALIASES",
    "SkillSpec",
    "build_skill_pack",
    "eqa_specs_for_submode",
    "skill_names_for_mode",
]
