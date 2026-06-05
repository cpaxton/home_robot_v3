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

import pytest

from emet.simulation.molmospaces_env import env_flag, molmospaces_nav_teleport_enabled


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_env_flag(monkeypatch, value, expected):
    monkeypatch.setenv("EMET_TEST_FLAG", value)
    assert env_flag("EMET_TEST_FLAG", default="0") is expected


def test_molmospaces_nav_teleport_default(monkeypatch):
    monkeypatch.delenv("EMET_MOLMOSPACES_NAV_TELEPORT", raising=False)
    assert molmospaces_nav_teleport_enabled() is True


def test_molmospaces_nav_teleport_disabled(monkeypatch):
    monkeypatch.setenv("EMET_MOLMOSPACES_NAV_TELEPORT", "0")
    assert molmospaces_nav_teleport_enabled() is False
