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

"""emet-molmospaces must not declare emet as a PyPI dependency (install uses --no-deps -e .)."""

from __future__ import annotations

from pathlib import Path


def test_emet_molmospaces_dependencies_no_standalone_emet():
    path = Path(__file__).resolve().parents[3] / "packages" / "emet_molmospaces" / "pyproject.toml"
    if not path.exists():
        return
    in_deps = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps and s.startswith("]"):
            break
        if in_deps and s in ('"emet",', '"emet"'):
            raise AssertionError(
                "emet must not be a pip dependency of emet-molmospaces; emet is not on PyPI. "
                "Install scripts use: uv pip install --no-deps -e . then --no-deps -e packages/emet_molmospaces"
            )
