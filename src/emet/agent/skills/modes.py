# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agent orchestrator modes (shared skills, different tool packs)."""

from __future__ import annotations

from enum import Enum


class AgentMode(str, Enum):
    """Which tool pack + prompt + stop rules the orchestrator uses.

    Same robot memory / motion primitives; different allowed skills:

    * ``CHAT`` — Discord / terminal embodied agent (:func:`emet.agent.tools.get_tools`).
    * ``EQA_EPISODE`` — scored GraphEQA verify/explore loop
      (:func:`emet.memory.graph_eqa.agentic_tools.build_agentic_eqa_tools`).

    Habitat MCQ scoring stays on ``run_eqa`` / ``emet-habitat`` / ``--eqa-eval`` and never
    routes through the Discord chat router.
    """

    CHAT = "chat"
    EQA_EPISODE = "eqa_episode"
