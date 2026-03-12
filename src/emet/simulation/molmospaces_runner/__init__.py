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
# MolmoSpaces runner: entrypoint for subprocess. Run with the MolmoSpaces venv
# (python -m emet.simulation.molmospaces_runner ...). Do not import molmo_spaces
# here so the main emet process stays free of that dependency.

from __future__ import annotations

__all__ = ["main"]

from emet.simulation.molmospaces_runner._entrypoint import main
