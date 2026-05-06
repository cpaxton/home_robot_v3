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
# Copyright (c) Hello Robot, Inc. All rights reserved.

import builtins
import sys

import pytest

from emet.utils.mujoco_import import assert_mujoco_available


def test_assert_mujoco_skips_when_installed():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("mujoco not installed (install sim extra to run this assertion)")
    assert_mujoco_available()


def test_assert_mujoco_raises_when_import_fails(monkeypatch):
    monkeypatch.delitem(sys.modules, "mujoco", raising=False)  # type: ignore[arg-type]
    real = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mujoco":
            raise ImportError("simulated missing mujoco")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="uv sync"):
        assert_mujoco_available()


def test_assert_mujoco_osmesa_failure_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "mujoco", raising=False)  # type: ignore[arg-type]
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    real = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mujoco":
            raise AttributeError("'NoneType' object has no attribute 'glGetError'")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="MUJOCO_GL=egl"):
        assert_mujoco_available()
