# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Single-use multi-operation plans; measured verification after every skill."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from emet.controller.manipulation.native_inputs import ObservationUnavailable

from .api import failure_code, response
from .task_budget import BudgetExhausted
from .task_contracts import BudgetSpec, GoalSpec, TaskState
from .task_planning import Operator, TaskDomain, search_task


def provider_failure_code(error: Exception) -> str:
    """Return only public semantic reasons, never provider exception details."""
    if isinstance(error, BudgetExhausted):
        return "task_motion_budget_exhausted"
    if isinstance(error, ObservationUnavailable):
        return ("stale_observation" if str(error) in {
            "object_observation_stale", "joint_observation_stale", "stale_observation"
        } else "observation_unavailable")
    code = failure_code(str(error))
    return code if code != "operation_failed" else "task_provider_failed"


@dataclass(frozen=True)
class TaskExecutionResult:
    success: bool
    execution_started: bool
    failure_code: str = 'task_execution_failed'

    def __post_init__(self):
        if type(self.success) is not bool or type(self.execution_started) is not bool:
            raise ValueError('invalid_task_execution_receipt')
        if (not isinstance(self.failure_code, str) or self.failure_code == 'operation_failed'
                or failure_code(self.failure_code) != self.failure_code):
            raise ValueError('invalid_task_execution_code')


class TaskProvider(TaskDomain, Protocol):
    def snapshot(self) -> TaskState: ...
    def compile_goal(self, instruction: str) -> GoalSpec: ...
    def execute(self, operator: Operator) -> bool | TaskExecutionResult: ...


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

    def invalidate(self):
        """External motion invalidates stored handles, including failed motion."""
        with self._lock:
            self._plans.clear()

    def discover(self, object_filter: str = "", robot: str = "") -> dict:
        with self._lock:
            if object_filter or robot:
                return response("scene_tasks", code="task_filter_unsupported")
            describe = getattr(self.provider, "describe", None)
            if describe is None:
                return response("scene_tasks", code="task_discovery_unavailable")
            try:
                return response("scene_tasks", data=describe())
            except Exception as error:
                return response("scene_tasks", code=provider_failure_code(error))

    def plan(self, instruction: str, strategy: str = 'beam') -> dict:
        if not isinstance(instruction, str) or not instruction.strip():
            return response("plan_task", code="invalid_input")
        with self._lock:
            try:
                initial = self.provider.snapshot()
                goal = self.provider.compile_goal(instruction)
                result = search_task(self.provider, initial, goal, strategy=strategy, budget=self.budget)
                if result.reason != 'ok':
                    return response('plan_task', code=result.reason)
                initial.binding.require_current(self.provider.snapshot().binding)
            except Exception as error:
                return response("plan_task", code=provider_failure_code(error))
            ref = 'task_plan_' + uuid.uuid4().hex
            # Bound memory even if an agent continually asks for plans.
            if len(self._plans) >= self.budget.tool_rounds:
                self._plans.pop(next(iter(self._plans)))
            self._plans[ref] = StoredTaskPlan(initial, goal, result.operators)
            return response('plan_task', data={'plan_ref': ref, 'steps': len(result.operators),
                                              'strategy': strategy, 'execution_started': False,
                                              'planning_scope': getattr(self.provider, 'planning_scope', 'provider_defined'),
                                              'motion_refinement': getattr(self.provider, 'motion_refinement', 'provider_defined')})

    def execute(self, plan_ref: str) -> dict:
        with self._lock:
            plan = self._plans.pop(plan_ref, None)
            if plan is None:
                return response('execute_task_plan', code='unknown_plan', recovery='replan')
            started, completed, active_step = False, 0, None
            phase = "initial_validation"
            try:
                current = self.provider.snapshot()
                plan.initial.binding.require_current(current.binding)
                if current.predicates != plan.initial.predicates:
                    raise ValueError('stale_task_plan')
                for index, operator in enumerate(plan.operators):
                    active_step, phase = index+1, "preflight"
                    if not operator.requires <= current.predicates:
                        raise ValueError('task_precondition_failed')
                    if not self.provider.feasible(current, operator):
                        raise ValueError('task_feasibility_failed')
                    predicted = operator.apply(current)
                    if not plan.goal.protected <= predicted.predicates:
                        raise ValueError('protected_goal_regression')
                    current.binding.require_current(self.provider.snapshot().binding)
                    phase = "execution"
                    previous_started, started = started, True
                    receipt = self.provider.execute(operator)
                    if isinstance(receipt, TaskExecutionResult):
                        started = previous_started or receipt.execution_started
                        success = receipt.success
                        if not success and not receipt.execution_started:
                            phase = "preflight"
                    else:
                        success = receipt is True
                    if not success:
                        raise ValueError(receipt.failure_code if isinstance(receipt, TaskExecutionResult)
                                         else 'task_execution_failed')
                    phase = "effect_verification"
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
                active_step, phase = None, "goal_verification"
                if not plan.goal.satisfied(current):
                    raise ValueError('task_goal_verification_failed')
            except Exception as error:
                code = provider_failure_code(error)
                return response('execute_task_plan', code=code, status='partial' if started else 'error',
                                data={'execution_started': started, 'completed_steps': completed,
                                      'planned_steps': len(plan.operators), 'failed_step_1based': active_step,
                                      'failed_phase': phase})
            return response('execute_task_plan', data={'execution_started': started,
                                                       'completed_steps': completed, 'goal_verified': True})
