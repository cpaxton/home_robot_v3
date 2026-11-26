# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Canonical RoboCasa fixture tokens shared by task, articulation, and teacher paths."""

from __future__ import annotations

ROBOCASA_FIXTURE_ALIASES: dict[str, str] = {
    "cabinet": "cab",
    "countertop": "counter",
    "toaster": "toaster",
    "toasteroven": "toasteroven",
    "standmixer": "standmixer",
    "fridgedrawer": "fridgedrawer",
    "fridgeshelf": "fridgeshelf",
    "electrickettle": "electrickettle",
    "coffeemachine": "coffeemachine",
}


def robocasa_fixture_token(name: str) -> str:
    """Normalize a task/fixture name to the wizard/short fixture token."""
    token = str(name).strip().lower().replace(" ", "")
    return ROBOCASA_FIXTURE_ALIASES.get(token, token)


__all__ = ["ROBOCASA_FIXTURE_ALIASES", "robocasa_fixture_token"]
