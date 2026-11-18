# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Shared CHAT capability restrictions used by binding, dispatch and prompts."""

ROTATE_ONLY_BLOCKED_TOOLS = (
    "explore",
    "find_objects",
    "move_forward",
    "go_home",
    "pick_place",
    "execute_pick_place_plan",
    "execute_task_plan",
    "hand_over",
)
