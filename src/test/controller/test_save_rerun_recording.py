# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""save_rerun must not rr.init while a live RerunVisualizer is streaming.

Injects a fake ``rerun`` module so this file never loads rerun-sdk native
extensions (those SIGSEGV on i9-14900KF turbo P-cores).
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from emet.controller.dynamem.look import maybe_save_rerun_recording


def _install_fake_rerun(monkeypatch) -> tuple[list[object], list[str]]:
    inits: list[object] = []
    saves: list[str] = []
    fake = ModuleType("rerun")
    fake.init = lambda *a, **k: inits.append(1)
    fake.save = lambda p: saves.append(p)
    monkeypatch.setitem(sys.modules, "rerun", fake)
    return inits, saves


def test_maybe_save_rerun_noop_when_disabled(monkeypatch, tmp_path):
    inits, saves = _install_fake_rerun(monkeypatch)
    agent = SimpleNamespace(
        save_rerun=False,
        log=str(tmp_path),
        rerun_iter=0,
        rerun_visualizer=SimpleNamespace(enabled=True),
    )
    maybe_save_rerun_recording(agent)
    assert inits == []
    assert saves == []


def test_maybe_save_rerun_skips_init_when_live(monkeypatch, tmp_path):
    inits, saves = _install_fake_rerun(monkeypatch)
    agent = SimpleNamespace(
        save_rerun=True,
        log=str(tmp_path),
        rerun_iter=3,
        rerun_visualizer=SimpleNamespace(enabled=True),
    )
    maybe_save_rerun_recording(agent)
    assert inits == []
    assert saves == [str(tmp_path / "data_3.rrd")]


def test_maybe_save_rerun_inits_when_no_live_viewer(monkeypatch, tmp_path):
    inits, saves = _install_fake_rerun(monkeypatch)
    agent = SimpleNamespace(
        save_rerun=True,
        log=str(tmp_path),
        rerun_iter=1,
        rerun_visualizer=SimpleNamespace(enabled=False),
    )
    maybe_save_rerun_recording(agent)
    assert inits == [1]
    assert saves == [str(tmp_path / "data_1.rrd")]
