# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import sys

from emet_molmospaces.runner import main_runner


def main() -> int:
    """Console script entrypoint for emet-molmospaces."""
    return main_runner()


if __name__ == "__main__":
    raise SystemExit(main())
