# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Measured occupancy, native first-move probes and honest failure receipts."""
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from emet.agent.tools import get_tools
from emet.controller.task.tamp.task_contracts import SceneBinding
from emet.controller.task.tamp.task_session import TaskExecutionResult, TaskSession
from emet.controller.task.tamp.transfer_provider import RegionTransferProvider, TransferObservation, TransferRegion


def make_domain(*, lie=False):
    regions = tuple(TransferRegion(name, ((i*2., 0.), (i*2.+1., 1.)))
                    for i, name in enumerate(('left', 'right', 'buffer')))
    current = [TransferObservation(SceneBinding('a'*64, 'b'*64, 'boot', 'reset', 0),
                  (('alpha', 'left'), ('beta', 'right')), (('alpha', (.2, .2)), ('beta', (.2, .2))))]
    calls = []
    def transfer(obj, region):
        calls.append((obj, region.name))
        if not lie:
            old = current[0]
            locations = dict(old.locations)
            locations[obj] = region.name
            current[0] = replace(old, locations=tuple(locations.items()),
                                  binding=replace(old.binding, snapshot_revision=old.binding.snapshot_revision+1))
        return True
    domain = RegionTransferProvider(regions, observe=lambda: current[0], transfer=transfer, input_mode='privileged')
    return domain, current, calls


@pytest.mark.parametrize('strategy', ['beam', 'mcts'])
def test_grounded_swap_through_production_tools(strategy):
    domain, _, calls = make_domain()
    tools = {t.name: t for t in get_tools({'tamp_task_session': TaskSession(domain)})}
    discovery = json.loads(tools['scene_tasks'].func())
    assert discovery['data']['free_regions'] == ['buffer']
    plan = json.loads(tools['plan_task'].func(instruction='swap alpha and beta', strategy=strategy))
    assert plan['data']['steps'] == 3 and plan['data']['planning_scope'] == 'task_skeleton'
    assert not calls
    result = json.loads(tools['execute_task_plan'].func(plan_ref=plan['data']['plan_ref']))
    assert result['status'] == 'ok' and result['data']['completed_steps'] == 3
    assert dict(domain.observe().locations) == {'alpha': 'right', 'beta': 'left'}


def test_claimed_success_without_measured_effect_stops():
    domain, _, calls = make_domain(lie=True)
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    assert session.execute(ref)['code'] == 'task_effect_verification_failed'
    assert len(calls) == 1


@pytest.mark.parametrize('text', ['move alpha to missing', 'swap alpha and missing', 'do whatever',
    'move alpha to right; move alpha to buffer', 'move alpha to buffer; move beta to buffer'])
def test_bad_goals_do_not_execute(text):
    domain, _, calls = make_domain()
    domain.snapshot()
    with pytest.raises(ValueError):
        domain.compile_goal(text)
    assert not calls


def test_no_buffer_is_an_explicit_search_failure():
    domain, current, calls = make_domain()
    current[0] = replace(current[0], blocked_regions=frozenset({'buffer'}))
    assert TaskSession(domain).plan('swap alpha and beta')['code'] == 'search_exhausted'
    assert not calls


def test_privileged_observation_cannot_enter_observed_provider():
    domain, _, _ = make_domain()
    domain.input_mode = 'observed'
    with pytest.raises(ValueError, match='source_mismatch'):
        domain.snapshot()


@pytest.mark.parametrize('prior_progress', [False, True])
def test_refinement_failure_preserves_actual_progress(prior_progress):
    domain, _, calls = make_domain()
    original = domain.transfer
    def transfer(obj, region):
        if prior_progress and not calls:
            return original(obj, region)
        return TaskExecutionResult(False, False, 'task_motion_planning_failed')
    domain.transfer = transfer
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    done = session.execute(ref)
    assert done['code'] == 'task_motion_planning_failed'
    assert done['status'] == ('partial' if prior_progress else 'error')
    assert done['data']['execution_started'] is prior_progress
    assert done['data']['completed_steps'] == int(prior_progress)


def native_probe_domain():
    from emet.controller.task.tamp.native_transfer import NativeTableTransferProvider
    domain, current, _ = make_domain()
    native = NativeTableTransferProvider.__new__(NativeTableTransferProvider)
    RegionTransferProvider.__init__(native, tuple(domain.regions.values()),
        observe=lambda: current[0], transfer=domain.transfer, input_mode='privileged')
    native.roles, native.support = {'alpha': 'private_a', 'beta': 'private_b'}, 'private_table'
    native._ik_remaining, native._motion_cache, native.motion_evidence = 100, {}, []
    calls = []
    adapter = SimpleNamespace(lock=threading.RLock(), max_planning_ik_calls=20,
                              planning_ik_calls=0, tasks={}, plans={})
    def plan(task_ref, placement_region):
        obj = adapter.tasks[task_ref][2]
        calls.append(obj)
        adapter.planning_ik_calls = min(5, adapter.max_planning_ik_calls)
        ok = obj == 'beta'
        if ok:
            adapter.plans['probe'] = object()
        return {'status': 'ok' if ok else 'error', 'code': 'ok' if ok else 'no_plan_within_budget',
                'data': {'plan_ref': 'probe', 'planning_rejections': {'pregrasp:pose_ik_failed': 5}}}
    adapter.plan = plan
    native.adapter = adapter
    return native, calls


def test_native_first_move_failure_selects_another_causal_sequence_without_commands():
    domain, probes = native_probe_domain()
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    assert session._plans[ref].operators[0].ref.startswith('transfer:beta:')
    assert sorted(probes) == ['alpha', 'beta']
    assert not domain.adapter.plans  # Probe handles never leak into execution.
    assert domain._ik_remaining == 90
    assert domain.adapter.max_planning_ik_calls == 20


def test_native_probes_share_remaining_ik_cap():
    domain, probes = native_probe_domain()
    domain._ik_remaining = 3
    session = TaskSession(domain)
    result = session.plan('swap alpha and beta')
    assert result['code'] == 'task_motion_budget_exhausted' and len(probes) == 1
    assert domain._ik_remaining == 0 and domain.adapter.max_planning_ik_calls == 20


def test_failed_native_probe_consumes_calls_and_restores_primitive_cap():
    domain, _ = native_probe_domain()
    def timeout(**_):
        domain.adapter.planning_ik_calls = 2
        raise TimeoutError('private transport information')
    domain.adapter.plan = timeout
    with pytest.raises(TimeoutError):
        domain._refine('alpha', domain.regions['buffer'])
    assert domain._ik_remaining == 98
    assert domain.adapter.max_planning_ik_calls == 20


def test_second_preflight_failure_preserves_step_number():
    domain, _, calls = make_domain()
    original = domain.feasible
    domain.feasible = lambda state, op: not calls and original(state, op)
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    result = session.execute(ref)
    assert result['status'] == 'partial'
    assert result['data']['completed_steps'] == 1
    assert result['data']['failed_step_1based'] == 2
    assert result['data']['failed_phase'] == 'preflight'


def test_debug_state_is_json_in_logs_but_not_public_motion_evidence(caplog):
    import numpy as np
    domain, _ = native_probe_domain()
    domain.diagnostic_state = True
    domain.adapter.signature = 'a'*64
    domain.adapter.events = []
    domain.adapter.data = SimpleNamespace(qpos=np.array([.1]), qvel=np.array([0.]), time=1.)
    domain._refine('alpha', domain.regions['buffer'])
    payload = json.loads(caplog.records[-1].getMessage().split('Native task refinement: ', 1)[1])
    assert payload['debug_state']['qpos'] == [.1]
    assert payload['stage'] == 'task_search_probe'
    assert 'debug_state' not in domain.motion_evidence[-1]


@pytest.mark.parametrize('tool', ['scene_tasks', 'plan_task'])
@pytest.mark.parametrize('reason,expected', [
    ('joint_observation_stale', 'stale_observation'),
    ('object_state_unknown', 'observation_unavailable'),
])
def test_observation_errors_are_public_codes_through_json_tools(tool, reason, expected):
    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    domain, _, calls = make_domain()
    def unavailable(*args):
        raise ObservationUnavailable(reason)
    if tool == 'scene_tasks':
        domain.describe = unavailable
    else:
        domain.snapshot = unavailable
    session = TaskSession(domain)
    tools = {t.name: t for t in get_tools({'tamp_task_session': session})}
    result = json.loads(tools[tool].func(**({'instruction': 'swap alpha and beta'} if tool == 'plan_task' else {})))
    assert result['code'] == expected and result['status'] == 'error'
    assert not session._plans and not calls


def test_scene_change_during_planning_returns_replan_without_storing_handle():
    domain, _, calls = make_domain()
    snapshot = domain.snapshot
    count = 0
    def changing():
        nonlocal count
        count += 1
        state = snapshot()
        return replace(state, binding=replace(state.binding, snapshot_revision=count))
    domain.snapshot = changing
    session = TaskSession(domain)
    tools = {t.name: t for t in get_tools({'tamp_task_session': session})}
    result = json.loads(tools['plan_task'].func(instruction='swap alpha and beta'))
    assert result['code'] == 'stale_task_plan' and result['recovery'] == 'replan'
    assert not session._plans and not calls


@pytest.mark.parametrize('phase', ['discover', 'snapshot', 'compile_goal', 'search', 'execute'])
def test_provider_exception_details_never_escape(phase):
    domain, _, calls = make_domain()
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref'] if phase == 'execute' else None
    def broken(*args):
        raise RuntimeError('/private/path body_123 secret')
    setattr(domain, {'discover': 'describe', 'search': 'feasible'}.get(phase, phase), broken)
    if phase == 'discover':
        result = session.discover()
    elif phase == 'execute':
        result = session.execute(ref)
    else:
        result = session.plan('swap alpha and beta')
    assert result['code'] == 'task_provider_failed'
    assert 'private' not in json.dumps(result) and 'body_123' not in json.dumps(result)
    assert not calls


@pytest.mark.parametrize('code', ['task_object_not_settled_on_support', 'ambiguous_region_occupancy',
                                 'native_execution_unavailable', 'object_not_lifted'])
@pytest.mark.parametrize('receipt', [False, True])
def test_actionable_provider_failures_preserve_code_and_progress(code, receipt):
    domain, _, calls = make_domain()
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    if receipt:
        domain.transfer = lambda *args: TaskExecutionResult(False, False, code)
    else:
        def reject(*args):
            raise ValueError(code)
        domain.snapshot = reject
    result = session.execute(ref)
    assert result['code'] == code and result['status'] == 'error'
    assert result['data']['execution_started'] is False
    assert result['data']['completed_steps'] == 0 and not calls
    assert session.execute(ref)['code'] == 'unknown_plan'


def test_receipt_rejects_unknown_private_codes():
    with pytest.raises(ValueError, match='invalid_task_execution_code'):
        TaskExecutionResult(False, False, '/private/path body_123')


def test_native_transfer_preserves_adapter_failure_code():
    domain, _ = native_probe_domain()
    domain._refine = lambda *args, **kwargs: 'plan'
    domain.adapter.execute = lambda ref: {'status': 'error', 'code': 'object_not_lifted',
                                         'data': {'execution_started': True}}
    result = domain._transfer_native('alpha', domain.regions['buffer'])
    assert result == TaskExecutionResult(False, True, 'object_not_lifted')


def test_semantic_failure_after_progress_stops_and_preserves_first_transfer():
    domain, _, calls = make_domain()
    original = domain.transfer
    def transfer(obj, region):
        if not calls:
            return original(obj, region)
        return TaskExecutionResult(False, False, 'native_execution_unavailable')
    domain.transfer = transfer
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    result = session.execute(ref)
    assert result['code'] == 'native_execution_unavailable' and result['status'] == 'partial'
    assert result['data']['completed_steps'] == 1
    assert result['data']['execution_started'] is True
    assert result['data']['failed_step_1based'] == 2 and len(calls) == 1


@pytest.mark.parametrize('prior_progress', [False, True])
def test_last_preflight_ik_call_cannot_claim_execution_started(prior_progress):
    domain, probes = native_probe_domain()
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    domain._motion_cache.clear()
    domain.transfer = domain._transfer_native
    # The adapter probe consumes five calls and selects beta first.
    domain._ik_remaining = 5
    commands = []
    domain.adapter.execute = lambda handle: commands.append(handle)
    if prior_progress:
        # Exercise this same precommand failure after one measured transfer.
        symbolic, _, calls = make_domain()
        session = TaskSession(symbolic)
        ref = session.plan('swap alpha and beta')['data']['plan_ref']
        first = symbolic.transfer
        def transfer(obj, region):
            if not calls:
                return first(obj, region)
            return domain._transfer_native('beta', domain.regions['buffer'])
        symbolic.transfer = transfer
        # Represent the successful preflight that consumed the last allowance.
        domain._refine('beta', domain.regions['buffer'])
    result = session.execute(ref)
    assert result['code'] == 'task_motion_budget_exhausted'
    assert result['status'] == ('partial' if prior_progress else 'error')
    assert result['data']['execution_started'] is prior_progress
    assert result['data']['completed_steps'] == int(prior_progress)
    assert result['data']['failed_step_1based'] == 1 + int(prior_progress)
    assert domain._ik_remaining == 0 and commands == []


@pytest.mark.parametrize('stage', ['snapshot', 'feasible'])
@pytest.mark.parametrize('reason', ['task_object_not_settled_on_support', 'budget'])
def test_provider_internal_preflight_failure_is_no_actuation(stage, reason):
    from emet.controller.task.tamp.task_budget import BudgetExhausted
    domain, _, calls = make_domain()
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    original = domain.execute
    def execute(operator):
        def reject(*args):
            if reason == 'budget':
                raise BudgetExhausted('task_motion_budget_exhausted')
            raise ValueError(reason)
        setattr(domain, stage, reject)
        return original(operator)
    domain.execute = execute
    result = session.execute(ref)
    assert result['code'] == ('task_motion_budget_exhausted' if reason == 'budget' else reason)
    assert result['status'] == 'error' and result['data']['execution_started'] is False
    assert result['data']['completed_steps'] == 0 and calls == []


def test_unknown_transfer_exception_remains_conservative():
    domain, _, calls = make_domain()
    session = TaskSession(domain)
    ref = session.plan('swap alpha and beta')['data']['plan_ref']
    def uncertain(*args):
        raise RuntimeError('transport outcome unknown')
    domain.transfer = uncertain
    result = session.execute(ref)
    assert result['status'] == 'partial' and result['data']['execution_started'] is True


def test_explicit_move_cannot_relocate_unrequested_object():
    domain, _, calls = make_domain()
    session = TaskSession(domain)
    domain.snapshot()
    goal = domain.compile_goal('move alpha to right')
    assert goal.protected == frozenset({('at', 'beta', 'right')})
    assert session.plan('move alpha to right')['code'] == 'search_exhausted'
    assert not calls
    result = session.plan('move alpha to buffer')
    assert result['status'] == 'ok'
    done = session.execute(result['data']['plan_ref'])
    assert done['status'] == 'ok' and calls == [('alpha', 'buffer')]
    assert dict(domain.observe().locations)['beta'] == 'right'
