"""Full registry coverage and honest result accounting without a simulator."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


def runner():
    path=Path(__file__).resolve().parents[3]/'scripts/run_tamp_experiments.py'
    spec=importlib.util.spec_from_file_location('run_tamp_experiments',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_matrix_retains_all_200_rows_and_modes():
    cases=runner().build_cases('full',python='python')
    assert len(cases)==200 and len({c['id'] for c in cases})==200
    assert Counter(c['robot'] for c in cases)=={'rby1':110,'stretch':30,'innate_mars':30,'nori':30}
    assert {c['execution_mode'] for c in cases}=={'oracle_teleport','kinematic_latch'}
    assert all('--episode-id' in c['command'] for c in cases)


def test_robot_revalidation_is_an_explicit_subset_without_rewriting_cases():
    module = runner()
    full = module.build_cases('full', python='python')
    subset = module.build_cases('full', python='python', robot='stretch')
    assert len(subset) == 30
    assert subset == [case for case in full if case['robot'] == 'stretch']
    with pytest.raises(ValueError, match='No registry cases'):
        module.build_cases('full', robot='missing_robot')
    with pytest.raises(ValueError, match='only for small/full'):
        module.build_cases('protocol', robot='stretch')


def test_protocol_has_24_cases_and_floor_includes_deferred_find():
    module=runner()
    assert sum(c['expected_episodes'] for c in module.build_cases('protocol'))==24
    cases=module.build_cases('floor')
    assert len(cases)==4 and any('find_only' in c['id'] for c in cases)
    assert all('--gt-only' in c['command'] for c in cases)


def test_process_success_is_not_task_success(tmp_path):
    case={'id':'episode','execution_mode':'oracle_teleport','expected_episodes':1}
    (tmp_path/'episode.json').write_text(json.dumps({'task_success':False,'goal_reached':False}))
    assert not runner().summarize_case(case,tmp_path,0)['task_success']


def test_missing_and_invalid_rows_remain_visible(tmp_path):
    module=runner()
    case={'id':'episode','execution_mode':'oracle_teleport','expected_episodes':1}
    assert module.summarize_case(case,tmp_path,0)['status']=='error'
    (tmp_path/'episode.json').write_text(json.dumps({'skipped_invalid':True,'task_success':False}))
    assert module.summarize_case(case,tmp_path,0)['status']=='invalid_fixture'


def test_crash_overrides_success_artifact_and_missing_artifact(tmp_path):
    module = runner()
    case = {'id': 'episode', 'execution_mode': 'oracle_teleport', 'expected_episodes': 1}
    assert module.summarize_case(case, tmp_path, -11)['status'] == 'crashed'
    (tmp_path / 'episode.json').write_text(json.dumps({'task_success': True}))
    result = module.summarize_case(case, tmp_path, -11)
    assert result['status'] == 'crashed' and not result['task_success']
    assert result['evidence'] and result['exit_code'] == -11
    assert not module.summarize_case(case, tmp_path, 1)['task_success']


def test_explicit_case_subset_keeps_order_and_rejects_unknown(monkeypatch, capsys):
    import sys

    module = runner()
    cases = module.build_cases('full')
    selected = [cases[2]['id'], cases[0]['id']]
    argv = ['runner', '--suite', 'full', '--output-dir', '/unused', '--dry-run']
    for name in selected:
        argv.extend(['--case-id', name])
    monkeypatch.setattr(sys, 'argv', argv)
    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['cases'] == [cases[0], cases[2]]
    assert result['expected_episodes'] == 2
    monkeypatch.setattr(sys, 'argv', argv + ['--case-id', 'typo'])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_unvalidated_candidates_require_admission_and_keep_rejections(tmp_path):
    import yaml

    module = runner()
    registry = tmp_path / 'candidates.yaml'
    row = {'id': 'candidate', 'robot': 'stretch', 'requires_fixture': True}
    registry.write_text(yaml.safe_dump({'episodes': [row]}))
    with pytest.raises(ValueError, match='Unvalidated candidate'):
        module.build_cases('full', registry=registry)
    cases = module.build_cases('full', registry=registry, validate_fixtures=True)
    assert cases[0]['command'][-1] == '--validate-fixture'
    assert cases[0]['stage'] == 'fixture_validation'
    for status in ('fixture_admitted', 'fixture_rejected'):
        (tmp_path/'candidate.json').write_text(json.dumps({'status': status, 'task_success': False}))
        result = module.summarize_case(cases[0], tmp_path, 0)
        assert result['status'] == status and not result['task_success']
    registry.write_text(yaml.safe_dump({'episodes': [row, row]}))
    with pytest.raises(ValueError, match='Duplicate'):
        module.build_cases('full', registry=registry, validate_fixtures=True)
