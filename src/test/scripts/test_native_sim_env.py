# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""A physical experiment must not inherit assisted benchmark navigation."""
from pathlib import Path

from emet.eval.sim_eval_session import benchmark_subprocess_env


def test_physical_benchmark_disables_both_teleport_paths(monkeypatch):
    monkeypatch.setenv('EMET_PHYSICAL_EXECUTION', '1')
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '1')
    monkeypatch.setenv('EMET_MOLMOSPACES_NAV_TELEPORT', '1')
    env = benchmark_subprocess_env(Path('/tmp/repo'), cpu_only=False)
    assert env['EMET_SIM_NAV_TELEPORT'] == '0'
    assert env['EMET_MOLMOSPACES_NAV_TELEPORT'] == '0'


def test_assisted_benchmark_retains_legacy_default(monkeypatch):
    monkeypatch.delenv('EMET_PHYSICAL_EXECUTION', raising=False)
    assert benchmark_subprocess_env(Path('/tmp/repo'), cpu_only=False)['EMET_SIM_NAV_TELEPORT'] == '1'
