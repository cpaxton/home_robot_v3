# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""MolmoSpaces-related process environment toggles (see docs/molmospaces_environment_variables.md)."""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, *, default: str = "1") -> bool:
    """Parse a boolean environment variable (``1``/``true``/``yes``/``on`` vs ``0``/``false``/``no``/``off``)."""
    raw = os.environ.get(name, default)
    v = str(raw).strip().lower()
    if v in _FALSY:
        return False
    if v in _TRUTHY:
        return True
    return bool(v)


def molmospaces_nav_teleport_enabled() -> bool:
    """Whether MolmoSpaces ZMQ navigation uses free-joint teleport (default on).

    Set ``EMET_MOLMOSPACES_NAV_TELEPORT=0`` to force wheel / ``set_goal_pose`` drive for experiments.
    Only applies when the server detects a MolmoSpaces merge session.
    """
    return env_flag("EMET_MOLMOSPACES_NAV_TELEPORT", default="1")
