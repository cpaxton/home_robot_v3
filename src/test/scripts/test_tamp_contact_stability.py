# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Freeze real inputs without actuation; retain failed/incomplete audit outcomes."""
import importlib
import inspect
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_scene import model_signature
from emet.eval.contact_stability import TracedNativeAdapter
from emet.motion.release import ReleaseLimits


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


def test_bounded_release_protocol_and_constructor_wiring(runner, monkeypatch, tmp_path):
    inspect.signature(TracedNativeAdapter).bind(object(), model=object(), trace=object(),
        policy='fine_yaw', payload_binding='verified_lift', release_limits=ReleaseLimits(.04, .04))
    monkeypatch.setattr(runner, 'live_table_robot', lambda *a, **kw: pytest.fail('prepare only'))
    out = tmp_path / 'bounded'
    arguments(monkeypatch, out, '--prepare-only', '--release-policy', 'bounded')
    assert runner.main() == 0
    protocol = json.loads((out / 'protocol.json').read_text())
    assert protocol['release_limits'] == {'max_raise_m': .04, 'max_drop_m': .04, 'clearance_m': .003}
    assert 'src/emet/motion/release.py' in protocol['source_sha256']


@pytest.mark.parametrize('lift_complete', (False, True))
def test_plot_does_not_label_later_place_rejection_as_a_failed_lift(monkeypatch, lift_complete):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    plotter = importlib.import_module('plot_tamp_contact_stability')
    def row(phase):
        return {'object_body': 'cube', 'event': {'phase': phase}, 'rotation_residual_rad': .1}
    rows = [row('gripper_settled'), row('lift')]
    if lift_complete:
        rows += [row('lift_complete'), row('preplace_complete')]
    rows += [row('payload_retention')]
    _, lift, later = plotter.lift_samples(rows, 'cube')
    assert later == lift_complete
    assert any(r['event']['phase'] == 'payload_retention' for r in lift) is not lift_complete
