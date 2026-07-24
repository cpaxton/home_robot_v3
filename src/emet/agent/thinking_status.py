# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Optional user-facing status while the agent waits on the LLM or tools."""

from __future__ import annotations

import os
from typing import Any

from emet.agent.model_debug import hf_or_model_id_from_client
from emet.llms.base import AbstractVLLMClient

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def env_agent_thinking_status() -> bool:
    """When unset, thinking status is on. Set ``EMET_AGENT_THINKING_STATUS=0`` to disable."""
    v = os.environ.get("EMET_AGENT_THINKING_STATUS", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return True


def short_llm_label(llm_key: str, llm_client: Any | None = None) -> str:
    """Compact model name for status lines (e.g. ``Qwen3-VL-8B``)."""
    if llm_client is not None:
        wid = hf_or_model_id_from_client(llm_client)
        if wid:
            return str(wid).split("/")[-1]
        if isinstance(llm_client, AbstractVLLMClient):
            key = str(llm_client.canonical_model_key)
            return key.removesuffix("-eqa")
    return str(llm_key).removesuffix("-eqa")


def format_llm_thinking_status(
    *,
    llm_label: str,
    round_idx: int,
    max_rounds: int,
    has_image: bool,
    followup: bool,
) -> str:
    """One-line status before an LLM call."""
    step = f" (step {round_idx}/{max_rounds})" if max_rounds > 1 else ""
    if followup or round_idx > 1:
        return f"*Thinking…*{step} {llm_label} — summarizing tool results…"
    if has_image:
        return f"*Thinking…*{step} {llm_label} — reading your message and head camera…"
    return f"*Thinking…*{step} {llm_label} — reading your message and choosing tools…"


def format_tool_running_status(tool_names: list[str]) -> str:
    """One-line status before executing parsed tool calls."""
    names = [n for n in tool_names if n]
    joined = ", ".join(names) if names else "tools"
    return f"*Running tools…* {joined}"


# Tools that move the robot / take noticeable wall-clock time — Discord gets live status.
_ACTION_STATUS_LABELS: dict[str, str] = {
    "explore": "Exploring",
    "scan_environment": "Looking around",
    "rotate_base": "Turning",
    "face_toward": "Turning toward",
    "describe_scene": "Looking",
    "send_image": "Looking",
    "move_forward": "Moving forward",
    "find_objects": "Searching",
    "go_home": "Going home",
    "pick_place": "Manipulating",
    "hand_over": "Handing over",
}


def format_action_running_status(tool_names: list[str], detail: str | None = None) -> str | None:
    """Italic action status for Discord/TTY (``*Exploring…*``, optional detail).

    When several action tools are queued (e.g. rotate then describe), joins labels so
    Discord does not only show the first (``*Turning, then looking…*``).

    Returns None when none of *tool_names* are long-running action tools (caller may
    fall back to :func:`format_tool_running_status` for the terminal only).
    """
    names = [n for n in tool_names if n]
    labels: list[str] = []
    for n in names:
        lab = _ACTION_STATUS_LABELS.get(n)
        if lab and lab not in labels:
            labels.append(lab)
    if not labels:
        return None
    if len(labels) == 1:
        phrase = labels[0]
    elif len(labels) == 2:
        phrase = f"{labels[0]}, then {labels[1].lower()}"
    else:
        phrase = ", then ".join([labels[0]] + [x.lower() for x in labels[1:]])
    base = f"*{phrase}…*"
    if detail and str(detail).strip():
        return f"{base} {str(detail).strip()}"
    return base


def is_action_status_tool(name: str) -> bool:
    return str(name or "") in _ACTION_STATUS_LABELS


def format_single_action_status(tool_name: str) -> str | None:
    """Per-tool Discord status while tools run sequentially (``*Looking…*``)."""
    lab = _ACTION_STATUS_LABELS.get(str(tool_name or ""))
    if not lab:
        return None
    return f"*{lab}…*"
