# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""CPU-only checks for control accounting, identity, search and measured effects."""
import json
from dataclasses import asdict, replace

import pytest

from emet.controller.task.tamp.task_contracts import (
    BudgetSpec,
    EpisodeSpec,
    GoalSpec,
    SceneBinding,
    TaskState,
)
from emet.controller.task.tamp.task_planning import Operator, search_task
from emet.controller.task.tamp.task_session import TaskSession
from emet.eval.tamp_benchmark.adapters import GymnasiumAdapter, RobosuiteAdapter
from emet.eval.tamp_benchmark.registry import load_registry
from emet.eval.tamp_benchmark.runtime import EpisodeRuntime, EpisodeStopped, StepResult


def episode(**kwargs):
    return replace(EpisodeSpec('e1', 'robosuite', 'Lift', 'reset0', 0, 'Lift the object', 3), **kwargs)


class Env:
    horizon = 3
    ignore_done = False

    def __init__(self):
        self.steps, self.closes = 0, 0

    def reset(self, spec):
        return {'reset': spec.reset_id}

    def step(self, action):
        self.steps += 1
        return StepResult({'step': self.steps}, 0, False, False, {})

    def success(self):
        return True

    def close(self):
        self.closes += 1


def test_horizon_counts_skill_internal_commands_and_closes_once():
    env = Env()
    with EpisodeRuntime(env, episode()) as runtime:
        for _ in range(3):
            result = runtime.step('gripper_settle')
        assert result.truncated
        with pytest.raises(EpisodeStopped):
            runtime.step('next_skill')
        assert runtime.scores(final_goal=False, physical_integrity=None) == {
            'upstream_success': True, 'final_goal_success': False, 'physical_integrity': None,
            'control_steps': 3, 'stop_reason': 'horizon_exhausted'}
    runtime.close()
    assert env.steps == 3 and env.closes == 1


def test_wall_budget_stops_before_next_command():
    now = [0.0]
    env = Env()
    with EpisodeRuntime(env, episode(budgets=BudgetSpec(episode_seconds=1)), clock=lambda: now[0]) as runtime:
        now[0] = 1.0
        with pytest.raises(EpisodeStopped, match='wall_budget_exhausted'):
            runtime.step('move')
    assert env.steps == 0


@pytest.mark.parametrize('stage', ['reset', 'step'])
def test_failed_environment_is_closed_and_cannot_continue(stage):
    env = Env()
    def fail(*_):
        raise TimeoutError('private path')
    setattr(env, stage, fail)
    runtime = EpisodeRuntime(env, episode())
    with pytest.raises(TimeoutError):
        with runtime:
            runtime.step('move')
    with pytest.raises(EpisodeStopped):
        runtime.step('retry')
    assert env.closes == 1


@pytest.mark.parametrize('kwargs', [{'suite': 'sapien'}, {'strategy': 'not_mcts'},
                                  {'horizon': True}, {'input_mode': 'oracle'}, {'seed': -1}])
def test_invalid_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        episode(**kwargs)


def test_identity_covers_condition_and_budget():
    spec = episode()
    assert spec == EpisodeSpec.from_dict(asdict(spec))
    assert spec.identity != replace(spec, robot_track='stretch').identity
    assert spec.identity != replace(spec, budgets=BudgetSpec(llm_calls=1)).identity


def test_registry_never_executes_design_only_or_duplicate_population(tmp_path):
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps({'schema_version': 1, 'executable_registry': False}))
    with pytest.raises(ValueError, match='design_only'):
        load_registry(path)
    raw = {'schema_version': 1, 'executable_registry': True,
           'provenance': {'implementation_sha': 'abc', 'upstream_revision': 'def',
                          'reset_manifest_sha256': 'a' * 64},
           'episodes': [asdict(episode())]}
    path.write_text(json.dumps(raw))
    assert len(load_registry(path)) == 1
    raw['episodes'] *= 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='duplicate'):
        load_registry(path)


def test_upstream_bridge_does_not_replace_controller_or_success():
    env = Env()
    env.step = lambda action: (action, 0.0, True, {'native': True})
    adapter = RobosuiteAdapter(env, suite='robosuite',
                              install_reset=lambda e, spec: e.reset(spec), success=lambda: False)
    with EpisodeRuntime(adapter, episode()) as runtime:
        assert runtime.step('native_action').observation == 'native_action'
        assert runtime.scores(final_goal=True, physical_integrity=True)['upstream_success'] is False
    env.ignore_done = True
    with pytest.raises(ValueError, match='ignore_done'):
        adapter.reset(episode())


def test_gymnasium_preserves_truncation_and_info():
    env = Env()
    env.step = lambda action: (action, 1, False, True, {'success': 0})
    adapter = GymnasiumAdapter(env, suite='metaworld',
                               install_reset=lambda e, spec: ('obs', {}),
                               success=lambda e, info: bool(info['success']))
    adapter.reset(episode(suite='metaworld'))
    assert adapter.step('native_action').truncated
    assert adapter.success() is False


def binding():
    return SceneBinding('a' * 64, 'b' * 64, 'boot', 'reset0', 0)


class SwapDomain:
    """Two occupied destinations require temporary storage (three transfers)."""
    def __init__(self):
        self.state = TaskState(frozenset({('at', 'a', 'left'), ('at', 'b', 'right'),
                                          ('free', 'buffer')}), binding())
        self.commands = []
        self.lie = False
        self.failure = False

    def snapshot(self):
        return self.state

    def compile_goal(self, instruction):
        return GoalSpec(frozenset({('at', 'a', 'right'), ('at', 'b', 'left'), ('free', 'buffer')}))

    def operators(self, state):
        result = []
        for _, obj, source in sorted(p for p in state.predicates if p[0] == 'at'):
            for _, target in sorted(p for p in state.predicates if p[0] == 'free'):
                result.append(Operator(f'{obj}:{source}:{target}', 'transfer',
                    frozenset({('at', obj, source), ('free', target)}),
                    frozenset({('at', obj, target), ('free', source)}),
                    frozenset({('at', obj, source), ('free', target)})))
        return tuple(result)

    def feasible(self, state, operator):
        return True

    def execute(self, operator):
        self.commands.append(operator.ref)
        if self.failure:
            return False
        if not self.lie:
            predicted = operator.apply(self.state)
            self.state = replace(predicted, binding=replace(self.state.binding,
                                 snapshot_revision=self.state.binding.snapshot_revision + 1))
        return True


@pytest.mark.parametrize("strategy", ["beam", "mcts"])
def test_beam_solves_causal_buffer_dependency_without_mutating_scene(strategy):
    domain = SwapDomain()
    before = domain.state
    result = search_task(domain, before, domain.compile_goal('swap'), strategy=strategy, budget=BudgetSpec())
    assert result.reason == 'ok' and len(result.operators) == 3
    assert domain.state == before and not domain.commands
    session = TaskSession(domain)
    plan = session.plan('swap')
    assert plan['status'] == 'ok'
    done = session.execute(plan['data']['plan_ref'])
    assert done['data']['completed_steps'] == 3 and done['data']['goal_verified']
    assert session.execute(plan['data']['plan_ref'])['code'] == 'unknown_plan'


@pytest.mark.parametrize('change', ['boot_id', 'reset_id', 'snapshot_revision'])
def test_stale_plan_never_commands(change):
    domain = SwapDomain()
    session = TaskSession(domain)
    ref = session.plan('swap')['data']['plan_ref']
    value = 1 if change == 'snapshot_revision' else 'changed'
    domain.state = replace(domain.state, binding=replace(domain.state.binding, **{change: value}))
    assert session.execute(ref)['code'] == 'stale_task_plan'
    assert domain.commands == []


@pytest.mark.parametrize('failure', ['lie', 'failure'])
def test_execution_uses_measured_effects_and_stops_without_retry(failure):
    domain = SwapDomain()
    session = TaskSession(domain)
    ref = session.plan('swap')['data']['plan_ref']
    setattr(domain, failure, True)
    result = session.execute(ref)
    assert result['status'] == 'partial'
    assert result['data']['execution_started'] and result['data']['completed_steps'] == 0
    assert len(domain.commands) == 1


def test_strategy_names_do_not_alias_another_algorithm():
    domain = SwapDomain()
    result = search_task(domain, domain.state, domain.compile_goal('swap'),
                         strategy='pddlstream', budget=BudgetSpec())
    assert result.reason == 'strategy_unavailable'


def test_search_budget_is_shared_across_branches():
    domain = SwapDomain()
    result = search_task(domain, domain.state, domain.compile_goal('swap'),
                         strategy='beam', budget=BudgetSpec(expansions=1))
    assert result.reason == 'expansion_budget_exhausted' and result.expansions == 1


def test_chat_tools_bind_session_and_fail_explicitly_without_provider():
    from emet.agent.tool_outcome import ToolOutcome
    from emet.agent.tools import get_tools
    tools = {tool.name: tool for tool in get_tools({})}
    result = tools['plan_task'].func(instruction='swap')
    assert ToolOutcome.coerce('plan_task', result).status == 'task_provider_unavailable'
    domain = SwapDomain()
    tools = {tool.name: tool for tool in get_tools({'tamp_task_session': TaskSession(domain)})}
    plan = json.loads(tools['plan_task'].func(instruction='swap'))
    result = tools['execute_task_plan'].func(plan_ref=plan['data']['plan_ref'])
    assert ToolOutcome.coerce('execute_task_plan', result).ok
    assert len(domain.commands) == 3


def test_fixed_population_keeps_unsupported_in_denominator():
    from emet.controller.task.tamp.task_contracts import EpisodeResult
    from emet.eval.tamp_benchmark.qualification import summarize_population
    first, second = episode(), episode(episode_id='e2')
    success = EpisodeResult(first.identity, 'completed', True, True, True, 3, 1, 'ok', 'abc')
    unavailable = EpisodeResult(second.identity, 'unsupported', None, None, None, 0, 0, 'no_adapter', 'abc')
    summary = summarize_population((first, second), (success, unavailable))
    assert summary['strict_success_fraction'] == 0.5
    assert summary['categories']['unsupported'] == 1
    assert summarize_population((first, second), (success,))['missing'] == 1
    with pytest.raises(ValueError, match='duplicate_episode_result'):
        summarize_population((first, second), (success, success))


def test_shared_budget_counts_failed_and_nested_calls():
    from emet.controller.task.tamp.task_budget import BudgetExhausted, TaskBudget
    budget = TaskBudget(BudgetSpec(llm_calls=2))
    budget.reserve('llm_calls')  # Outer production chat call.
    budget.reserve('llm_calls')  # Inner affordance ranking; also counted.
    with pytest.raises(BudgetExhausted, match='llm_calls'):
        budget.reserve('llm_calls')
    assert budget.snapshot()['llm_calls'] == 2


@pytest.mark.parametrize('strategy', ['beam', 'mcts'])
def test_slow_feasibility_cannot_accept_goal_after_deadline(strategy):
    domain = SwapDomain()
    now = [0.0]
    def slow(*_):
        now[0] += 2
        return True
    domain.feasible = slow
    result = search_task(domain, domain.state, domain.compile_goal('swap'), strategy=strategy,
                         budget=BudgetSpec(planning_seconds=1), clock=lambda: now[0])
    assert result.reason == 'planning_timeout'


def test_scene_change_during_execution_preflight_prevents_command():
    domain = SwapDomain()
    session = TaskSession(domain)
    ref = session.plan('swap')['data']['plan_ref']
    def moved(*_):
        domain.state = replace(domain.state, binding=replace(domain.state.binding, snapshot_revision=1))
        return True
    domain.feasible = moved
    result = session.execute(ref)
    assert result['code'] == 'stale_task_plan'
    assert not result['data']['execution_started'] and not domain.commands


@pytest.mark.parametrize('strategy', ['beam', 'mcts'])
def test_protected_goal_cannot_be_sacrificed_to_complete_task(strategy):
    domain = SwapDomain()
    goal = replace(domain.compile_goal('swap'), protected=frozenset({('at', 'a', 'left')}))
    result = search_task(domain, domain.state, goal, strategy=strategy, budget=BudgetSpec())
    assert result.reason != 'ok'
