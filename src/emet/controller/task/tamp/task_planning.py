# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded symbolic search over one shared grounded operator provider.

Provider feasibility checks must use private planning state. They do not step
physics, attach payloads, or consult the benchmark's evaluator-only goals.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .task_contracts import BudgetSpec, GoalSpec, TaskState


@dataclass(frozen=True)
class Operator:
    ref: str
    name: str
    requires: frozenset[tuple[str, ...]]
    adds: frozenset[tuple[str, ...]]
    deletes: frozenset[tuple[str, ...]] = frozenset()

    def apply(self, state: TaskState) -> TaskState:
        if not self.requires <= state.predicates:
            raise ValueError('operator_preconditions_failed')
        return TaskState((state.predicates - self.deletes) | self.adds, state.binding)


class TaskDomain(Protocol):
    def operators(self, state: TaskState) -> tuple[Operator, ...]: ...
    def feasible(self, state: TaskState, operator: Operator) -> bool: ...


@dataclass(frozen=True)
class SearchResult:
    operators: tuple[Operator, ...]
    reason: str
    expansions: int


def search_task(domain: TaskDomain, start: TaskState, goal: GoalSpec, *,
                strategy: str, budget: BudgetSpec,
                clock: Callable[[], float] = time.monotonic) -> SearchResult:
    """Greedy, beam or existing UCT using identical operators and feasibility.

    Other algorithms are explicitly unsupported here; a baseline label must not
    silently select a different algorithm. LLM and PDDLStream bindings are separate.
    """
    if strategy == 'mcts':
        from .task_mcts import search_mcts
        return search_mcts(domain, start, goal, budget, clock=clock)
    if strategy not in ('greedy', 'beam'):
        return SearchResult((), 'strategy_unavailable', 0)
    if not goal.protected <= start.predicates:
        return SearchResult((), 'protected_goals_not_satisfied', 0)
    if goal.satisfied(start):
        return SearchResult((), 'ok', 0)
    width = 1 if strategy == 'greedy' else budget.beam_width
    frontier = [(start, ())]
    visited = {start.predicates}
    expansions = 0
    deadline = clock() + budget.planning_seconds
    for _ in range(budget.depth):
        candidates = []
        for state, path in frontier:
            for operator in domain.operators(state):
                if clock() >= deadline:
                    return SearchResult((), 'planning_timeout', expansions)
                if expansions >= budget.expansions:
                    return SearchResult((), 'expansion_budget_exhausted', expansions)
                expansions += 1
                if not operator.requires <= state.predicates:
                    continue
                next_state = operator.apply(state)
                if next_state.predicates in visited or not goal.protected <= next_state.predicates:
                    continue
                if not domain.feasible(state, operator):
                    continue
                if clock() >= deadline:
                    return SearchResult((), "planning_timeout", expansions)
                visited.add(next_state.predicates)
                next_path = path + (operator,)
                if goal.satisfied(next_state):
                    return SearchResult(next_path, 'ok', expansions)
                candidates.append((next_state, next_path))
        if not candidates:
            return SearchResult((), 'search_exhausted', expansions)
        candidates.sort(key=lambda candidate: (
            len(goal.required - candidate[0].predicates), len(candidate[1]),
            tuple(op.ref for op in candidate[1])))
        frontier = candidates[:width]
    return SearchResult((), 'depth_budget_exhausted', expansions)
