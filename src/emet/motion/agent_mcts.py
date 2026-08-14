# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agent-call-wrapping MCTS for TAMP-style mobile pick & place.

The search tree is deliberately small: **expansion is one policy call** that
proposes ``K`` candidate next tool calls for a state (the LLM agent later; a
distance-based heuristic + sampling today), and **rollouts run a simulator**
(the existing executor / MuJoCo server later; a deterministic step in tests).
This is the "wrap our agent calls" layer — the policy proposes, MCTS looks ahead
and returns the best action sequence, which the agent then executes for real.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MCTSAction:
    """One candidate tool call, e.g. ``MCTSAction("pickup", {"object": "apple"})``."""

    name: str
    args: dict = field(default_factory=dict)
    cost: float = 1.0


@dataclass
class ScoredCandidate:
    """A proposed action plus its policy prior (used for sampling / rollouts)."""

    action: MCTSAction
    prior: float = 0.0
    predicted_reward: float = 0.0


@dataclass
class MCTSConfig:
    n_iterations: int = 200
    expansion_breadth: int = 4  # K candidates per node (policy calls)
    depth_limit: int = 6
    uct_c: float = 1.4
    discount: float = 0.95
    seed: int | None = None


class MCTSNode:
    """A search node: a state plus the action that produced it."""

    __slots__ = ("state", "action", "parent", "children", "visits", "value", "prior", "step_reward", "done")

    def __init__(self, state: Any, action: MCTSAction | None = None, parent: MCTSNode | None = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0
        self.step_reward = 0.0
        self.done = False

    @property
    def q(self) -> float:
        return self.value / max(self.visits, 1)


class AgentMCTSPlanner:
    """UCT search over candidate tool-call sequences.

    ``policy(state, goal, K) -> list[ScoredCandidate]`` proposes candidate next
    actions (heuristic today, LLM later). ``simulate(state, action) ->
    (next_state, reward, done)`` is a functional step — it must return a *new*
    state, not mutate the input. ``goal_distance(state, goal)`` is 0 when the
    goal is achieved; defaults to ``policy.goal_distance`` when available.
    """

    def __init__(
        self,
        policy: Callable[..., list[ScoredCandidate]],
        simulate: Callable[..., tuple[Any, float, bool]],
        goal_distance: Callable[..., float] | None = None,
        config: MCTSConfig | None = None,
    ):
        self.policy = policy
        self.simulate = simulate
        pd = policy.goal_distance if hasattr(policy, "goal_distance") else None
        self.goal_distance: Callable[..., float] = goal_distance or pd  # type: ignore[assignment]
        if self.goal_distance is None:  # pragma: no cover
            raise ValueError("goal_distance required (or policy.goal_distance)")
        self.config = config or MCTSConfig()
        self._rng = random.Random(self.config.seed)
        self.root: MCTSNode | None = None
        self._goal: Any = None

    # -- public ----------------------------------------------------------------

    def search(self, start_state: Any, goal: Any, *, verbose: bool = False) -> list[MCTSAction]:
        """Run MCTS and return the best action sequence (root-to-best-child path)."""
        self._goal = goal
        self.root = MCTSNode(state=start_state)
        self._expand(self.root)
        for _ in range(int(self.config.n_iterations)):
            leaf = self._select(self.root)
            if leaf.done:
                self._backprop(leaf, leaf.step_reward)
                continue
            self._expand(leaf)
            for child in leaf.children:
                value = self._rollout(child) if not child.done else child.step_reward
                self._backprop(child, value)
        best = self._best_path(self.root)
        return self._trace(best) if best else []

    # -- MCTS core ---------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCT descent to an unexpanded leaf."""
        while node.children:
            if node.done:
                return node
            node = self._best_uct_child(node)
        return node

    def _expand(self, node: MCTSNode) -> None:
        """One policy call: expand ``K`` candidate children (states simulated)."""
        if node.children or node.done or self._is_terminal(node.state):
            node.done = node.done or self._is_terminal(node.state)
            return
        for cand in self.policy(node.state, self._goal, self.config.expansion_breadth):
            try:
                next_state, reward, done = self.simulate(node.state, cand.action)
            except Exception:
                next_state, reward, done = node.state, 0.0, True
            child = MCTSNode(state=next_state, action=cand.action, parent=node)
            child.prior = float(cand.prior)
            child.step_reward = float(reward)
            child.done = bool(done)
            node.children.append(child)
        if not node.children:
            node.done = True

    def _best_uct_child(self, node: MCTSNode) -> MCTSNode:
        log_n = math.log(max(node.visits, 1))
        c = float(self.config.uct_c)

        def uct(child: MCTSNode) -> float:
            if child.visits == 0:
                # unvisited -> exploration dominates, biased by the policy prior
                return child.prior + c * math.sqrt(log_n + 1.0)
            q = child.q
            return q + c * math.sqrt(log_n / child.visits)

        return max(node.children, key=uct)

    def _rollout(self, node: MCTSNode, *, depth: int | None = None) -> float:
        """Greedy best-prior rollout from *node* (fast default policy)."""
        state = node.state
        depth = self.config.depth_limit if depth is None else int(depth)
        value = 0.0
        discount = 1.0
        for _ in range(depth):
            if self._is_terminal(state):
                break
            cands = self.policy(state, self._goal, self.config.expansion_breadth)
            if not cands:
                break
            best = max(cands, key=lambda c: c.prior)
            try:
                state, reward, done = self.simulate(state, best.action)
            except Exception:
                done = True
                reward = 0.0
            value += discount * float(reward)
            discount *= float(self.config.discount)
            if done:
                break
        return value

    def _backprop(self, node: MCTSNode, value: float) -> None:
        cur: MCTSNode | None = node
        while cur is not None:
            cur.visits += 1
            cur.value += value
            value *= float(self.config.discount)
            cur = cur.parent

    def _is_terminal(self, state: Any) -> bool:
        try:
            return float(self.goal_distance(state, self._goal)) <= 1e-6
        except Exception:
            return False

    def _best_path(self, node: MCTSNode) -> MCTSNode:
        """Greedy descend from *node* along highest-Q children to a visited leaf."""
        cur = node
        while cur.children and any(c.visits > 0 for c in cur.children):
            cur = max((c for c in cur.children if c.visits > 0), key=lambda c: (c.q, c.visits))
        return cur

    def _best_child(self, node: MCTSNode) -> MCTSNode | None:
        if not node.children:
            return None
        return max(node.children, key=lambda c: (c.q, c.visits))

    def _trace(self, node: MCTSNode) -> list[MCTSAction]:
        """Root-to-*node* action sequence (skipping the virtual root)."""
        seq: list[MCTSAction] = []
        cur: MCTSNode | None = node
        while cur is not None and cur.parent is not None:
            if cur.action is not None:
                seq.append(cur.action)
            cur = cur.parent
        seq.reverse()
        return seq


# ---------------------------------------------------------------------------
# Distance-based heuristic policy with sampling
# ---------------------------------------------------------------------------


class PickPlaceDistancePolicy:
    """Distance-based candidate proposer for mobile pick & place.

    State schema (dict)::

        {"robot": (x, y), "object": (x, y), "carrying": bool, "receptacle": (x, y)}

    Goal: the object ends up at the receptacle. Candidates are the minimal TAMP
    action set for this task (move / pickup / place). Each candidate is scored by
    how much it reduces distance-to-goal (``1 / (1 + goal_dist_after)``, plus a
    reachability bonus), and ``__call__`` samples ``K`` candidates from that
    distribution so the search explores instead of going purely greedy.
    """

    def __init__(
        self,
        *,
        reach_m: float = 0.25,
        place_tol_m: float = 0.30,
        explore_eps: float = 0.1,
        seed: int | None = None,
    ):
        self.reach_m = float(reach_m)
        self.place_tol_m = float(place_tol_m)
        self.explore_eps = float(explore_eps)
        self._rng = np.random.default_rng(seed)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _xy(v: Any) -> np.ndarray:
        return np.asarray(v, dtype=float).reshape(-1)[:2]

    @staticmethod
    def _dist(a: Any, b: Any) -> float:
        return float(np.linalg.norm(PickPlaceDistancePolicy._xy(a) - PickPlaceDistancePolicy._xy(b)))

    def goal_distance(self, state: Any, goal: Any | None = None) -> float:
        """Distance the object still has to travel to reach the goal."""
        obj = self._xy(state["object"])
        target = self._xy(goal) if goal is not None else self._xy(state["receptacle"])
        return float(np.linalg.norm(obj - target))

    # -- candidate generation ---------------------------------------------------

    def _propose_all(self, state: Any, goal: Any) -> list[ScoredCandidate]:
        obj = self._xy(state["object"])
        rec = self._xy(goal) if goal is not None else self._xy(state["receptacle"])
        robot = self._xy(state["robot"])
        carrying = bool(state.get("carrying", False))
        dist_goal = float(np.linalg.norm(obj - rec))

        out: list[ScoredCandidate] = []
        if not carrying:
            # (1) go get the object
            out.append(
                self._scored(
                    MCTSAction("move_to", {"xy": obj.tolist()}, cost=self._dist(robot, obj)), dist_goal, dist_goal
                )
            )
            # (2) pickup, only meaningful when within reach
            if self._dist(robot, obj) <= self.reach_m:
                out.append(
                    self._scored(
                        MCTSAction("pickup", {"object": "target"}, cost=0.1), dist_goal, dist_goal, reach_bonus=1.0
                    )
                )
        else:
            # (3) carry to the receptacle
            out.append(
                self._scored(MCTSAction("move_to", {"xy": rec.tolist()}, cost=self._dist(robot, rec)), dist_goal, 0.0)
            )
            # (4) place, only when at the receptacle
            if self._dist(robot, rec) <= self.place_tol_m:
                out.append(
                    self._scored(
                        MCTSAction("place", {"receptacle": "target"}, cost=0.1), dist_goal, 0.0, reach_bonus=1.0
                    )
                )
        return out

    def _scored(
        self,
        action: MCTSAction,
        dist_before: float,
        dist_after: float,
        *,
        reach_bonus: float = 0.0,
    ) -> ScoredCandidate:
        reduction = max(0.0, float(dist_before) - float(dist_after))
        prior = 1.0 + reduction / max(1e-6, 1.0 + dist_after) + reach_bonus
        return ScoredCandidate(action=action, prior=float(prior))

    def __call__(self, state: Any, goal: Any, k: int) -> list[ScoredCandidate]:
        """Sample ``k`` candidates (without replacement) weighted by prior."""
        all_cands = self._propose_all(state, goal)
        if not all_cands:
            return []
        weights = np.array([float(c.prior) for c in all_cands], dtype=np.float64)
        weights += max(float(self.explore_eps), 1e-9)
        weights = weights / weights.sum()
        n = min(int(k), len(all_cands))
        idx = self._rng.choice(len(all_cands), size=n, replace=False, p=weights)
        return [all_cands[i] for i in idx]
