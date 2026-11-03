# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Freeze real inputs without actuation; retain failed/incomplete audit outcomes."""
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_scene import model_signature


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    return importlib.import_module('eval_tamp_contact_stability')


def arguments(monkeypatch, out, *extra):
    monkeypatch.setattr('sys.argv', ['eval_tamp_contact_stability', '--case-id', 'two-goal-shift',
                        '--policy', 'fine_yaw', '--port-offset', '1940', '--output', str(out), *extra])


def test_prepare_only_binds_shifted_fixture_and_never_starts_controllers(runner, monkeypatch, tmp_path):
    original = Path(runner.default_scene_xml_path).parent / 'scene_environment.xml'
    before = original.read_bytes()
    monkeypatch.setattr(runner, 'live_table_robot', lambda *a, **kw: pytest.fail('must not start a simulator'))
    out = tmp_path / 'prepared'
    arguments(monkeypatch, out, '--prepare-only')
    assert runner.main() == 0
    protocol = json.loads((out / 'protocol.json').read_text())
    source = mujoco.MjModel.from_xml_path(runner.default_scene_xml_path)
    model = mujoco.MjModel.from_xml_path(str(out / 'fixture/scene.xml'))
    assert protocol['model_signature'] == model_signature(source) == model_signature(model)
    for name, y in (('object1', -.55), ('object2', -.7)):
        np.testing.assert_allclose(model.body(name).pos, [-.005, y, .6], atol=1e-12)
    assert original.read_bytes() == before
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['run_status'] == 'prepared' and summary['passed'] is None
    assert summary['audit_commands'] == 0
    saved = (out / 'protocol.json').read_bytes()
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert (out / 'protocol.json').read_bytes() == saved


def test_infrastructure_error_still_has_terminal_failed_summary(runner, monkeypatch, tmp_path):
    def fail(*args):
        raise TimeoutError('test_stalled_physics')
    monkeypatch.setattr(runner, 'run', fail)
    out = tmp_path / 'failed'
    arguments(monkeypatch, out)
    assert runner.main() == 1
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['run_status'] == 'infrastructure_error'
    assert summary['error_type'] == 'TimeoutError' and summary['passed'] is False


def test_truncated_audit_cannot_become_a_pass_or_erase_summary(runner, monkeypatch, tmp_path):
    def corrupt(args, pilot, case, out, result):
        result.update(run_status='complete', passed=True)
        (out / 'actuation.jsonl').write_text('{"forbidden": []}\n{truncated')
    monkeypatch.setattr(runner, 'run', corrupt)
    out = tmp_path / 'corrupt'
    arguments(monkeypatch, out)
    assert runner.main() == 1
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['passed'] is False and summary['malformed_audit_rows'] == 1
