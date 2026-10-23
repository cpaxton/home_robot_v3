# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Immutable episode identity and explicit benchmark evidence contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SUITES = ('robosuite', 'robocasa', 'molmospaces', 'libero', 'franka_kitchen', 'metaworld')
STRATEGIES = ('react', 'llm_plan', 'greedy', 'beam', 'mcts', 'pddlstream', 'llm_affordance')


def fingerprint(value: Any) -> str:
    """Stable identity; reject NaN rather than producing non-standard JSON."""
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class BudgetSpec:
    tool_rounds: int = 32
    llm_calls: int = 64
    depth: int = 16
    expansions: int = 500
    beam_width: int = 8
    mcts_iterations: int = 500
    ik_calls: int = 2048
    planning_seconds: float = 120.0
    episode_seconds: float = 1800.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
                raise ValueError(f'invalid_budget:{name}')
            if name not in ('planning_seconds', 'episode_seconds') and not isinstance(value, int):
                raise ValueError(f'integer_budget_required:{name}')


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    suite: str
    task: str
    reset_id: str
    seed: int
    instruction: str
    horizon: int
    strategy: str = 'react'
    input_mode: Literal['privileged', 'observed'] = 'privileged'
    execution_mode: Literal['native_stepped', 'live_async'] = 'native_stepped'
    robot_track: Literal['upstream', 'stretch', 'sourccey'] = 'upstream'
    budgets: BudgetSpec = field(default_factory=BudgetSpec)
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.suite not in SUITES or self.strategy not in STRATEGIES:
            raise ValueError('unsupported_episode_contract')
        for value in (self.episode_id, self.task, self.reset_id, self.instruction):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('empty_episode_identity')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('invalid_seed')
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError('invalid_horizon')
        if self.input_mode not in ('privileged', 'observed'):
            raise ValueError('invalid_input_mode')
        if self.execution_mode not in ('native_stepped', 'live_async'):
            raise ValueError('invalid_execution_mode')
        if self.robot_track not in ('upstream', 'stretch', 'sourccey'):
            raise ValueError('invalid_robot_track')

    @classmethod
    def from_dict(cls, value: dict) -> EpisodeSpec:
        value = dict(value)
        value['budgets'] = BudgetSpec(**value.get('budgets', {}))
        return cls(**value)

    @property
    def identity(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class SceneBinding:
    model_sha256: str
    robot_profile_sha256: str
    boot_id: str
    reset_id: str
    snapshot_revision: int

    def __post_init__(self):
        for digest in (self.model_sha256, self.robot_profile_sha256):
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid_binding_digest')
        if not self.boot_id or not self.reset_id or self.snapshot_revision < 0:
            raise ValueError('invalid_scene_binding')

    def require_current(self, current: SceneBinding) -> None:
        if self != current:
            raise ValueError('stale_task_plan')


@dataclass(frozen=True)
class RobotCommandProfile:
    robot: str
    end_effector: str
    joints: tuple[str, ...]
    action_interface: str
    capabilities: tuple[str, ...]

    def __post_init__(self):
        if not all((self.robot, self.end_effector, self.joints, self.action_interface)):
            raise ValueError('incomplete_robot_profile')
        if len(set(self.joints)) != len(self.joints):
            raise ValueError('duplicate_profile_joint')

    @property
    def identity(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class TaskState:
    """Grounded public predicates only; evaluator truth is kept separately."""
    predicates: frozenset[tuple[str, ...]]
    binding: SceneBinding


@dataclass(frozen=True)
class GoalSpec:
    required: frozenset[tuple[str, ...]]
    protected: frozenset[tuple[str, ...]] = frozenset()

    def satisfied(self, state: TaskState) -> bool:
        return (self.required | self.protected) <= state.predicates


@dataclass(frozen=True)
class EpisodeResult:
    episode_identity: str
    category: Literal['completed', 'unsupported', 'construction_failure', 'reference_failure',
                      'agent_failure', 'infrastructure_failure']
    upstream_success: bool | None
    final_goal_success: bool | None
    physical_integrity: bool | None
    control_steps: int
    wall_seconds: float
    reason: str
    implementation_sha: str
    evidence_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)
