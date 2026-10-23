# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Task-domain binding to the existing UCT planner (not beam search)."""
from __future__ import annotations

import time

from .task_contracts import BudgetSpec, GoalSpec, TaskState
from .task_planning import SearchResult, TaskDomain


class SearchBudgetExceeded(RuntimeError):
    pass


def search_mcts(domain: TaskDomain, start: TaskState, goal: GoalSpec, budget: BudgetSpec,
                *, clock=time.monotonic) -> SearchResult:
    from emet.motion.agent_mcts import AgentMCTSPlanner, MCTSAction, MCTSConfig, ScoredCandidate

    if not goal.protected <= start.predicates:
        return SearchResult((), 'protected_goals_not_satisfied', 0)
    if goal.satisfied(start):
        return SearchResult((), 'ok', 0)
    deadline, expansions = clock() + budget.planning_seconds, 0
    cache = {}
    solutions = []

    def distance(node, _goal):
        return len(goal.required - node[0].predicates)

    def policy(node, _goal, count):
        nonlocal expansions
        state, path, history = node
        if clock() >= deadline:
            raise SearchBudgetExceeded('planning_timeout')
        if len(path) >= budget.depth:
            return []
        if node in cache:
            return cache[node]
        result = []
        for operator in domain.operators(state):
            if clock() >= deadline:
                raise SearchBudgetExceeded('planning_timeout')
            if expansions >= budget.expansions:
                raise SearchBudgetExceeded('expansion_budget_exhausted')
            expansions += 1
            if not operator.requires <= state.predicates:
                continue
            successor = operator.apply(state)
            if successor.predicates in history or not goal.protected <= successor.predicates:
                continue
            if not domain.feasible(state, operator):
                continue
            if clock() >= deadline:
                raise SearchBudgetExceeded('planning_timeout')
            action = MCTSAction(operator.ref, {'operator': operator})
            prior = 1.0 / (1 + len(goal.required - successor.predicates))
            result.append(ScoredCandidate(action, prior=prior))
        result.sort(key=lambda candidate: (-candidate.prior, candidate.action.name))
        # Expansion breadth is a separate finite baseline setting.
        cache[node] = result[:count]
        return cache[node]

    def simulate(node, action):
        state, path, history = node
        operator = action.args['operator']
        successor = operator.apply(state)
        next_path = path + (operator,)
        done = goal.satisfied(successor)
        if done:
            solutions.append(next_path)
        return ((successor, next_path, history | {successor.predicates}),
                1.0 if done else -0.01, done)

    planner = AgentMCTSPlanner(policy, simulate, distance, MCTSConfig(
        n_iterations=budget.mcts_iterations, expansion_breadth=budget.beam_width,
        depth_limit=budget.depth, seed=0))
    reason = 'search_exhausted'
    try:
        planner.search((start, (), frozenset({start.predicates})), goal)
    except SearchBudgetExceeded as error:
        reason = str(error)
    # UCT's highest-Q leaf need not be a goal. Return only witnessed goal paths.
    if solutions:
        best = min(solutions, key=lambda path: (len(path), tuple(op.ref for op in path)))
        return SearchResult(best, 'ok', expansions)
    return SearchResult((), reason, expansions)
