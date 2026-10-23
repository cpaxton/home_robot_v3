# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Single native step seam shared by skill commands and benchmark scoring."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from emet.controller.task.tamp.task_contracts import EpisodeSpec


class EpisodeStopped(RuntimeError):
    """Terminal episodes cannot accept another skill's low-level command."""


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class NativeEnvironment(Protocol):
    """Adapter owns native reset/controller/termination and original success."""
    def reset(self, episode: EpisodeSpec) -> Any: ...
    def step(self, action: Any) -> StepResult: ...
    def success(self) -> bool: ...
    def close(self) -> None: ...


class EpisodeRuntime:
    """Every action, including waits and gripper settling, consumes horizon.

    No action repeat or direct simulator state mutation is exposed here. A step
    failure poisons the runtime: the environment may already have advanced.
    """
    def __init__(self, env: NativeEnvironment, episode: EpisodeSpec,
                 *, clock: Callable[[], float] = time.monotonic):
        if episode.execution_mode != 'native_stepped':
            raise ValueError('live_async_requires_separate_runtime')
        self.env, self.episode, self.clock = env, episode, clock
        self.control_steps = 0
        self.started = clock()
        self.stopped = False
        self.closed = False
        self.observation = None
        self.initialized = False
        self.stop_reason = None

    def reset(self):
        if self.initialized or self.closed or self.stopped:
            raise EpisodeStopped('runtime_requires_new_instance')
        self.initialized = True
        try:
            self.observation = self.env.reset(self.episode)
        except BaseException:
            self.stopped = True
            self.stop_reason = 'reset_failed'
            raise
        self.started = self.clock()
        return self.observation

    def step(self, action: Any) -> StepResult:
        if not self.initialized or self.closed or self.stopped:
            raise EpisodeStopped(self.stop_reason or 'episode_not_running')
        if self.clock() - self.started >= self.episode.budgets.episode_seconds:
            self.stopped, self.stop_reason = True, 'wall_budget_exhausted'
            raise EpisodeStopped(self.stop_reason)
        # Count the attempted native action even if the transport raises.
        self.control_steps += 1
        try:
            result = self.env.step(action)
            if not isinstance(result, StepResult):
                raise TypeError('invalid_native_step_result')
        except BaseException:
            self.stopped, self.stop_reason = True, 'native_step_failed'
            raise
        self.observation = result.observation
        horizon = self.control_steps >= self.episode.horizon
        truncated = result.truncated or horizon
        if result.terminated or truncated:
            self.stopped = True
            self.stop_reason = ('terminated' if result.terminated else
                                'upstream_truncated' if result.truncated else 'horizon_exhausted')
        return StepResult(result.observation, result.reward, result.terminated, truncated, result.info)

    def scores(self, *, final_goal: bool | None, physical_integrity: bool | None) -> dict:
        if not self.initialized or self.closed or self.stop_reason in ('reset_failed', 'native_step_failed'):
            raise EpisodeStopped('scoring_unavailable')
        # Never replace the upstream metric with our stricter audit.
        return {'upstream_success': bool(self.env.success()),
                'final_goal_success': final_goal, 'physical_integrity': physical_integrity,
                'control_steps': self.control_steps, 'stop_reason': self.stop_reason}

    def close(self):
        if not self.closed:
            self.closed = self.stopped = True
            self.env.close()

    def __enter__(self):
        try:
            self.reset()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_):
        self.close()
