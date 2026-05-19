# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.

"""Compatibility shim: scene spawn / autoplace lives in :mod:`emet.simulation.scene_base_spawn`.

New code should import ``emet.simulation.scene_base_spawn``. This module mirrors all public and
private attributes so existing ``from emet.simulation import molmospaces_spawn`` and tests keep
working.
"""

from __future__ import annotations

import emet.simulation.scene_base_spawn as _scene_base_spawn

__doc__ = _scene_base_spawn.__doc__

for _name in dir(_scene_base_spawn):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_scene_base_spawn, _name)
