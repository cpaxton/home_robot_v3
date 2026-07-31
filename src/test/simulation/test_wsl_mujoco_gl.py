import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from emet.simulation import wsl_mujoco_gl


def test_enable_software_gl_skips_when_not_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
    assert not wsl_mujoco_gl.enable_software_gl_for_wsl_headless_cameras(headless=True, no_cameras=False)
    assert "LIBGL_ALWAYS_SOFTWARE" not in os.environ


def test_enable_software_gl_sets_on_wsl_kernel(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)

    def fake_read_text(self: Path, *args, **kwargs) -> str:
        if self == Path("/proc/sys/kernel/osrelease"):
            return "6.6.1-microsoft-standard-WSL2\n"
        raise AssertionError(f"unexpected read: {self!r}")

    with patch.object(Path, "read_text", fake_read_text):
        assert wsl_mujoco_gl.enable_software_gl_for_wsl_headless_cameras(headless=True, no_cameras=False)

    assert os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1"


def test_enable_software_gl_noop_when_no_cameras(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)

    def fake_read_text(self: Path, *args, **kwargs) -> str:
        if self == Path("/proc/sys/kernel/osrelease"):
            return "6.6.1-microsoft-standard-WSL2\n"
        raise AssertionError(str(self))

    with patch.object(Path, "read_text", fake_read_text):
        assert not wsl_mujoco_gl.enable_software_gl_for_wsl_headless_cameras(headless=True, no_cameras=True)

    assert "LIBGL_ALWAYS_SOFTWARE" not in os.environ


@pytest.mark.parametrize(
    "release",
    ["6.6.1-generic", "6.1.0-azure"],
)
def test_enable_software_gl_skips_non_wsl_kernel(monkeypatch, release: str):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)

    def fake_read_text(self: Path, *args, **kwargs) -> str:
        if self == Path("/proc/sys/kernel/osrelease"):
            return f"{release}\n"
        raise AssertionError(str(self))

    with patch.object(Path, "read_text", fake_read_text):
        assert not wsl_mujoco_gl.enable_software_gl_for_wsl_headless_cameras(headless=True, no_cameras=False)

    assert "LIBGL_ALWAYS_SOFTWARE" not in os.environ
