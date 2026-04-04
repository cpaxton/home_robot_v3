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

from __future__ import annotations

from emet_molmospaces.runner import main_runner


def main() -> int:
    """Console script entrypoint for emet-molmospaces."""
    return main_runner()


if __name__ == "__main__":
    raise SystemExit(main())
