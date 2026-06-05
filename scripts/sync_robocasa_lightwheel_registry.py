#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Register LightWheel fixture mesh dirs (Sink025, Stove074, …) in fixture_registry YAML."""

from __future__ import annotations

import sys

from emet.simulation.robocasa_registry_sync import (
    restore_fixture_registry_from_vcs,
    sync_lightwheel_registry,
)


def main() -> int:
    restored = restore_fixture_registry_from_vcs()
    if restored:
        print(f"Restored {restored} fixture_registry YAML file(s) from robocasa git.")
    n = sync_lightwheel_registry()
    print(f"Added {n} LightWheel fixture registry entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
