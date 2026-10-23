# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Native environment bridges, without importing optional simulator packages.

Construct the full task environment upstream. Reset installers must restore the
published initialization, not sample a new reset with the same seed and claim
it is the fixed benchmark episode. Robot adaptations require their own track.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from emet.controller.task.tamp.task_contracts import EpisodeSpec

from .runtime import StepResult


class RobosuiteAdapter:
    """Retain the full robosuite/RoboCasa/LIBERO task and controller.

    ``install_reset`` handles the suite-specific reset certificate/init state.
    LIBERO's public ``check_success`` or robosuite's ``_check_success`` is bound
    explicitly by the caller; neither geometry nor reward substitutes for it.
    """
    def __init__(self, env, *, suite: str,
                 install_reset: Callable[[Any, EpisodeSpec], Any],
                 success: Callable[[], bool]):
        if suite not in ('robosuite', 'robocasa', 'libero'):
            raise ValueError('unsupported_robosuite_family')
        self.env, self.suite = env, suite
        self.install_reset, self._success = install_reset, success

    def reset(self, episode: EpisodeSpec):
        if episode.suite != self.suite:
            raise ValueError('suite_binding_mismatch')
        # Prevent the full upstream environment from silently using another cap.
        if getattr(self.env, 'ignore_done', False):
            raise ValueError('upstream_ignore_done_forbidden')
        horizon = getattr(self.env, 'horizon', None)
        if horizon is not None and horizon != episode.horizon:
            raise ValueError('upstream_horizon_mismatch')
        return self.install_reset(self.env, episode)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return StepResult(observation, float(reward), bool(done), False, dict(info))

    def success(self):
        return bool(self._success())

    def close(self):
        self.env.close()


class GymnasiumAdapter:
    """Franka Kitchen / Meta-World bridge using the native five-value step.

    An explicit task-specific success extractor avoids interpreting reward or
    TimeLimit truncation as successful completion. Multi-task Kitchen callers
    must use the task set's original upstream completion definition.
    """
    def __init__(self, env, *, suite: str,
                 install_reset: Callable[[Any, EpisodeSpec], Any],
                 success: Callable[[Any, dict], bool]):
        if suite not in ('franka_kitchen', 'metaworld'):
            raise ValueError('unsupported_gymnasium_family')
        self.env, self.suite = env, suite
        self.install_reset, self._success = install_reset, success
        self.last_info: dict = {}

    def reset(self, episode: EpisodeSpec):
        if episode.suite != self.suite:
            raise ValueError('suite_binding_mismatch')
        limit = getattr(getattr(self.env, 'spec', None), 'max_episode_steps', None)
        if limit is not None and limit != episode.horizon:
            raise ValueError('upstream_horizon_mismatch')
        observation, info = self.install_reset(self.env, episode)
        self.last_info = dict(info)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = dict(info)
        return StepResult(observation, float(reward), bool(terminated), bool(truncated), self.last_info)

    def success(self):
        return bool(self._success(self.env, self.last_info))

    def close(self):
        self.env.close()
