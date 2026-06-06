# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared interactive REPL loops and user-facing command text for ``emet run *`` apps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import click

# Graph EQA / Dynagraph: natural-language questions and frontier explore.
EXPLORE_ALIASES = frozenset({"explore", "e", "map", "nav"})
QUIT_ALIASES = frozenset({"q", "quit"})

GRAPH_EQA_HELP = (
    "Interactive mode: type a **question** for graph EQA, "
    "**explore** (or **e**) to extend the map without the EQA model, "
    "or **Q** / Enter to quit."
)

TASK_MODE_HELP = "Interactive mode: **E** / **explore** map frontiers · **M** pick and place · **Q** quit"
TASK_MODE_HELP_WITH_LIST = TASK_MODE_HELP + " · **L** list scene-graph objects"

TASK_MODE_EXPLORE = frozenset({"e", "explore", *EXPLORE_ALIASES})
TASK_MODE_LIST = frozenset({"l", "list"})
TASK_MODE_MANIP = frozenset({"m", "manip", "pick", "pickup", "place"})


def graph_eqa_prompt(app_name: str) -> str:
    return f"{app_name} [question | explore | Q=quit]: "


def task_mode_prompt(app_name: str, *, list_objects: bool = False) -> str:
    if list_objects:
        return f"{app_name} [E=explore | L=list | M=manip | Q=quit]: "
    return f"{app_name} [E=explore | M=manip | Q=quit]: "


def _should_quit(line: str) -> bool:
    low = line.strip().lower()
    return not low or low in QUIT_ALIASES


def echo_explore_step_result(finished: bool | None) -> None:
    """Shared feedback after one frontier explore step (GraphEQA / Dynagraph)."""
    if finished is None:
        click.echo("Explore step failed (no plan / blocked).")
    elif finished:
        click.echo("Explore step finished at target pose.")
    else:
        click.echo("Explore step advanced; ask a question or explore again.")


class _GraphEqaAgent(Protocol):
    def execute_action(self, text: str) -> tuple[bool | None, Any]: ...


class _GraphEqaExecutor(Protocol):
    def __call__(self, question: str) -> tuple[str, Any]: ...


class _GraphEqaRobot(Protocol):
    def move_to_nav_posture(self) -> None: ...
    def switch_to_navigation_mode(self) -> None: ...
    def say(self, msg: str) -> None: ...


def run_graph_eqa_loop(
    agent: _GraphEqaAgent,
    executor: _GraphEqaExecutor,
    robot: _GraphEqaRobot,
    *,
    app_name: str,
) -> None:
    """Question / explore REPL for ``run_graph_eqa`` and ``run_dynagraph``."""
    click.echo(GRAPH_EQA_HELP)
    while True:
        line = input(graph_eqa_prompt(app_name)).strip()
        if _should_quit(line):
            break
        robot.move_to_nav_posture()
        robot.switch_to_navigation_mode()
        low = line.lower()
        if low in EXPLORE_ALIASES:
            click.echo("- Exploring (no EQA call)…")
            finished, _pt = agent.execute_action("")
            echo_explore_step_result(finished)
            continue
        robot.say("Answering the question " + line)
        discord_text, _imgs = executor(line)
        if not discord_text.strip():
            print("(Empty EQA reply — check graph memory / observations.)")


class _TaskExecutor(Protocol):
    def __call__(self, response: list[tuple[str, str]], channel: Any = None) -> bool: ...


@dataclass
class PickPlacePromptState:
    target_object: str | None = None
    target_receptacle: str | None = None


def _pick_place_commands(state: PickPlacePromptState) -> list[tuple[str, str]]:
    obj = state.target_object
    if not obj:
        obj = input("Enter the target object: ").strip()
    rec = state.target_receptacle
    if not rec:
        rec = input("Enter the target receptacle: ").strip()
    state.target_object = None
    state.target_receptacle = None
    return [("pickup", obj), ("place", rec)]


def parse_task_mode_line(
    line: str,
    *,
    list_objects: Callable[[], None] | None,
    pick_place: PickPlacePromptState,
) -> list[tuple[str, str]] | None:
    """
    Parse one task-mode REPL line.

    Returns command list for the executor, or ``None`` to re-prompt (e.g. after list).
    """
    token = line.strip().lower()
    if _should_quit(line):
        return [("quit", "")]
    if token in TASK_MODE_EXPLORE:
        return [("explore", "")]
    if token in TASK_MODE_LIST:
        if list_objects is None:
            click.echo("List is not available in this app.")
            return None
        list_objects()
        return None
    if token in TASK_MODE_MANIP:
        return _pick_place_commands(pick_place)
    click.echo(
        "Unknown command. Use E=explore, M=manip, Q=quit" + (", L=list" if list_objects is not None else "") + "."
    )
    return None


def run_task_executor_loop(
    executor: _TaskExecutor,
    *,
    app_name: str,
    list_objects: Callable[[], None] | None = None,
    llm_query: Callable[[], list[tuple[str, str]]] | None = None,
    pick_place: PickPlacePromptState | None = None,
    debug_llm: bool = False,
    log_llm_response: Callable[[list[tuple[str, str]]], None] | None = None,
) -> None:
    """
    Task-mode REPL for ``run_dynamem`` and ``run_scene_graph``.

    Manual mode uses single-letter commands (E/L/M/Q) plus the same explore aliases as graph EQA.
    When *llm_query* is set, each turn uses the language model instead of the manual prompt.
    """
    click.echo(TASK_MODE_HELP_WITH_LIST if list_objects else TASK_MODE_HELP)
    state = pick_place if pick_place is not None else PickPlacePromptState()
    ok = True
    while ok:
        if llm_query is not None:
            llm_response = llm_query()
            if debug_llm and log_llm_response is not None:
                log_llm_response(llm_response)
        else:
            line = input(task_mode_prompt(app_name, list_objects=list_objects is not None)).strip()
            llm_response = parse_task_mode_line(line, list_objects=list_objects, pick_place=state)
            if llm_response is None:
                continue
        ok = executor(llm_response)
        state.target_object = None
        state.target_receptacle = None
