# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""PYTHONPATH sanitizer: venv site-packages must match the active interpreter tag."""

from __future__ import annotations

from pathlib import Path

from emet.utils import pythonpath as pp


def test_venv_site_packages_only_active_interpreter_tag(tmp_path: Path, monkeypatch):
    """A stray python3.12 dir in a 3.10 venv must not leak onto PYTHONPATH."""
    venv = tmp_path / ".venv"
    (venv / "lib" / "python3.10" / "site-packages").mkdir(parents=True)
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\nversion_info = 3.10.14.final.0\n", encoding="utf-8")

    monkeypatch.setattr(pp, "_repo_root", lambda: tmp_path)
    paths = pp._venv_site_packages_paths()
    assert paths == [str(venv / "lib" / "python3.10" / "site-packages")]


def test_venv_site_packages_glob_fallback_without_tag(tmp_path: Path, monkeypatch):
    venv = tmp_path / ".venv"
    sp = venv / "lib" / "python3.10" / "site-packages"
    sp.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")

    monkeypatch.setattr(pp, "_repo_root", lambda: tmp_path)
    assert pp._venv_python_tag() is None
    assert pp._venv_site_packages_paths() == [str(sp)]
