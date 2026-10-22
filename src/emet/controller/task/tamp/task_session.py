# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Single-use multi-operation plans; measured verification after every skill."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from .api import response
from .task_contracts import BudgetSpec, GoalSpec, TaskState
from .task_planning import Operator, TaskDomain, search_task


class TaskProvider(TaskDomain, Protocol):
    def snapshot(self) -> TaskState: ...
    def compile_goal(self, instruction: str) -> GoalSpec: ...
    def execute(self, operator: Operator) -> bool: ...


@dataclass(frozen=True)
class StoredTaskPlan:
    initial: TaskState
    goal: GoalSpec
    operators: tuple[Operator, ...]


class TaskSession:
    """Provider owns fresh grounding and measured execution, never symbolic success.

    Scene revision after each action is expected to advance. Boot/reset/model/
    robot identity must remain stable throughout, and every subsequent action is
    revalidated from a new measured snapshot. No retries after partial progress.
    """
    def __init__(self, provider: TaskProvider, *, budget: BudgetSpec | None = None):
        self.provider = provider
        self.budget = budget or BudgetSpec()
        self._plans: dict[str, StoredTaskPlan] = {}
        self._lock = threading.RLock()

    def plan(self, instruction: str, strategy: str = 'beam') -> dict:
        if not isinstance(instruction, str) or not instruction.strip():
            return response("plan_task", code="invalid_input")
        with self._lock:
            initial = self.provider.snapshot()
            goal = self.provider.compile_goal(instruction)
            result = search_task(self.provider, initial, goal, strategy=strategy, budget=self.budget)
            if result.reason != 'ok':
                return response('plan_task', code=result.reason)
            initial.binding.require_current(self.provider.snapshot().binding)
            ref = 'task_plan_' + uuid.uuid4().hex
            # Bound memory even if an agent continually asks for plans.
            if len(self._plans) >= self.budget.tool_rounds:
                self._plans.pop(next(iter(self._plans)))
            self._plans[ref] = StoredTaskPlan(initial, goal, result.operators)
            return response('plan_task', data={'plan_ref': ref, 'steps': len(result.operators),
                                              'strategy': strategy, 'execution_started': False})

    def execute(self, plan_ref: str) -> dict:
        with self._lock:
            plan = self._plans.pop(plan_ref, None)
            if plan is None:
                return response('execute_task_plan', code='unknown_plan', recovery='replan')
            started, completed = False, 0
            try:
                current = self.provider.snapshot()
                plan.initial.binding.require_current(current.binding)
                if current.predicates != plan.initial.predicates:
                    raise ValueError('stale_task_plan')
                for operator in plan.operators:
                    if not operator.requires <= current.predicates or not self.provider.feasible(current, operator):
                        raise ValueError('task_precondition_failed')
                    predicted = operator.apply(current)
                    if not plan.goal.protected <= predicted.predicates:
                        raise ValueError('protected_goal_regression')
                    current.binding.require_current(self.provider.snapshot().binding)
                    started = True
                    if self.provider.execute(operator) is not True:
                        raise ValueError('task_execution_failed')
                    measured = self.provider.snapshot()
                    before, after = current.binding, measured.binding
                    if (before.model_sha256, before.robot_profile_sha256, before.boot_id, before.reset_id) != (
                        after.model_sha256, after.robot_profile_sha256, after.boot_id, after.reset_id
                    ) or after.snapshot_revision < before.snapshot_revision:
                        raise ValueError('scene_identity_changed')
                    if not operator.adds <= measured.predicates or operator.deletes & measured.predicates:
                        raise ValueError('task_effect_verification_failed')
                    if not plan.goal.protected <= measured.predicates:
                        raise ValueError('protected_goal_regression')
                    completed += 1
                    current = measured
                if not plan.goal.satisfied(current):
                    raise ValueError('task_goal_verification_failed')
            except Exception as error:
                allowed = {'stale_task_plan', 'task_precondition_failed', 'task_execution_failed',
                           'protected_goal_regression', 'scene_identity_changed',
                           'task_effect_verification_failed', 'task_goal_verification_failed'}
                code = str(error) if str(error) in allowed else 'task_provider_failed'
                return response('execute_task_plan', code=code, status='partial' if started else 'error',
                                data={'execution_started': started, 'completed_steps': completed})
            return response('execute_task_plan', data={'execution_started': started,
                                                       'completed_steps': completed, 'goal_verified': True})
