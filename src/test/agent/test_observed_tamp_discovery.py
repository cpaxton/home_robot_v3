# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Observed discovery through the real registry, with no simulator or actuation."""
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tamp_session import tamp_followup
from emet.agent.tools import get_tools
from emet.controller.manipulation.native_inputs import ObjectPoseEvidence
from emet.controller.task.tamp.observed_discovery import ObservedTaskCandidate, discover_observed_tasks


def candidate(*, source='rgbd_tracking', age=0, name='cup'):
    pose = ObjectPoseEvidence(np.eye(4), time.monotonic() - age, source, 'private-observation-id')
    return ObservedTaskCandidate(name, 'counter', pose, pose)


def provider(*candidates):
    return SimpleNamespace(input_mode='observed', observe_tasks=Mock(return_value=tuple(candidates)))


def test_real_registry_discovery_never_reads_truth_and_cannot_enable_execution():
    robot = Mock()
    robot.get_emet_session.side_effect = AssertionError('privileged metadata read')
    observed = provider(candidate())
    tools = {t.name: t for t in get_tools({'robot': robot, 'manip_mode': 'physical',
        'tamp_inputs': 'observed', 'observed_tamp_provider': observed})}
    discovery = tools['scene_tasks'].func()
    result = json.loads(discovery)
    assert result['status'] == 'ok'
    assert result['data']['tasks'][0]['object_name'] == 'cup'
    assert 'private-observation-id' not in discovery
    assert result['data']['planning_available'] is False
    assert tamp_followup('Move the cup', ['[scene_tasks] ' + discovery]) is None
    for name, kwargs in [('plan_pick_place', {'task_ref': result['data']['tasks'][0]['task_ref']}),
                         ('execute_pick_place_plan', {'plan_ref': 'invented'}),
                         ('pick_place', {'object_name': 'cup', 'receptacle_name': 'counter'})]:
        assert json.loads(tools[name].func(**kwargs))['status'] == 'error'
    assert robot.mock_calls == []


@pytest.mark.parametrize('source,age,code', [('simulator_ground_truth', 0, 'privileged_input_forbidden'),
    ('rgbd_tracking', 3, 'object_observation_stale'), ('rgbd_tracking', -3, 'object_observation_stale')])
def test_invalid_evidence_rejects_entire_snapshot_even_if_filtered(source, age, code):
    result = discover_observed_tasks(provider(candidate(), candidate(source=source, age=age, name='hidden')),
                                    object_filter='cup')
    assert result['code'] == code
    assert 'tasks' not in result['data']


def test_unknown_entities_not_fabricated_and_handles_refresh():
    assert discover_observed_tasks(provider())['data']['tasks'] == []
    observed = provider(candidate(), candidate(name='bottle'))
    first = discover_observed_tasks(observed, 'CUP')['data']['tasks']
    second = discover_observed_tasks(observed, 'cup')['data']['tasks']
    assert len(first) == 1
    assert first[0]['task_ref'] != second[0]['task_ref']
    assert discover_observed_tasks(None)['code'] == 'observed_scene_unavailable'


def test_wrong_provider_and_invalid_snapshot_are_rejected():
    observed = provider(candidate())
    observed.input_mode = 'privileged'
    assert discover_observed_tasks(observed)['code'] == 'input_provider_mismatch'
    observed.observe_tasks.assert_not_called()
    observed.input_mode = 'observed'
    observed.observe_tasks.return_value = [{'object_name': 'cup'}]
    assert discover_observed_tasks(observed)['code'] == 'invalid_observed_task_snapshot'


def test_lookup_does_not_renew_evidence():
    original = candidate()
    observed = provider(original)
    assert discover_observed_tasks(observed, now=original.object_pose.received_monotonic + 1)['status'] == 'ok'
    assert discover_observed_tasks(observed, now=original.object_pose.received_monotonic + 3)['code'] == 'object_observation_stale'
