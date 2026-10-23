# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Unwired shared-budget infrastructure; not a production episode limit.

Current task search uses BudgetSpec and native probes use their own IK allowance.
No production caller constructs TaskBudget yet. Matched whole-agent budget claims
require wiring one ledger through CHAT, ranking, execution replanning and IK.
"""
import threading
import time

from .task_contracts import BudgetSpec


class BudgetExhausted(RuntimeError):
    pass


class TaskBudget:
    """Reserve before work; failed calls still consume the budget.

    The same instance must be passed to outer CHAT and inner ranking/IK code.
    Constructing a new ledger per tool would invalidate matched budget claims.
    """
    def __init__(self, spec: BudgetSpec, *, clock=time.monotonic):
        self.spec, self.clock = spec, clock
        self.started = clock()
        self._used = dict.fromkeys(('llm_calls', 'tool_rounds', 'ik_calls', 'expansions'), 0)
        self._lock = threading.Lock()

    def reserve(self, resource: str, amount: int = 1):
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError('invalid_budget_reservation')
        with self._lock:
            if resource not in self._used:
                raise ValueError('unknown_budget_resource')
            if self.clock() - self.started >= self.spec.episode_seconds:
                raise BudgetExhausted('episode_wall_budget_exhausted')
            if self._used[resource] + amount > getattr(self.spec, resource):
                raise BudgetExhausted(f'{resource}_budget_exhausted')
            self._used[resource] += amount

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._used)
