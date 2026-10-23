# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Measured region-transfer skeletons, with native motion refined per step."""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from .task_contracts import GoalSpec, SceneBinding, TaskState
from .task_planning import Operator
from .task_session import TaskExecutionResult, provider_failure_code


@dataclass(frozen=True)
class TransferRegion:
    name: str
    bounds: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name
                or self.name != self.name.strip().casefold() or self.name == 'outside'):
            raise ValueError('invalid_region_name')
        if (len(self.bounds) != 2 or any(len(row) != 2 for row in self.bounds)
                or not all(math.isfinite(v) for row in self.bounds for v in row)
                or not all(self.bounds[0][i] < self.bounds[1][i] for i in range(2))):
            raise ValueError('invalid_region_bounds')


@dataclass(frozen=True)
class TransferObservation:
    binding: SceneBinding
    # None means measured, supported outside all regions; never unknown or held.
    locations: tuple[tuple[str, str | None], ...]
    footprints: tuple[tuple[str, tuple[float, float]], ...]
    blocked_regions: frozenset[str] = frozenset()
    input_mode: str = 'privileged'


class RegionTransferProvider:
    planning_scope = 'task_skeleton'
    motion_refinement = 'per_step'

    def __init__(self, regions: tuple[TransferRegion, ...], *,
                 observe: Callable[[], TransferObservation],
                 transfer: Callable[[str, TransferRegion], bool | TaskExecutionResult], input_mode: str):
        if input_mode not in ('privileged', 'observed'):
            raise ValueError('invalid_input_mode')
        self.regions = {r.name: r for r in regions}
        if not regions or len(regions) != len(self.regions):
            raise ValueError('duplicate_or_empty_regions')
        for i, region in enumerate(regions):
            for other in regions[i+1:]:
                if all(region.bounds[0][k] < other.bounds[1][k]
                       and other.bounds[0][k] < region.bounds[1][k] for k in range(2)):
                    raise ValueError('overlapping_transfer_regions')
        self.observe, self.transfer, self.input_mode = observe, transfer, input_mode
        self._footprints = {}
        self._last = None

    def snapshot(self) -> TaskState:
        obs = self.observe()
        if obs.input_mode != self.input_mode:
            raise ValueError('task_input_source_mismatch')
        locations, sizes = dict(obs.locations), dict(obs.footprints)
        if (not locations or len(locations) != len(obs.locations) or set(locations) != set(sizes)
                or len(sizes) != len(obs.footprints) or not obs.blocked_regions <= self.regions.keys()):
            raise ValueError('incomplete_transfer_observation')
        occupied, predicates = set(), set()
        for obj, location in locations.items():
            if not obj or obj != obj.strip().casefold():
                raise ValueError('invalid_object_role')
            if location is not None:
                if location not in self.regions or location in occupied or location in obs.blocked_regions:
                    raise ValueError('ambiguous_region_occupancy')
                occupied.add(location)
            if len(sizes[obj]) != 2 or not all(math.isfinite(v) and v > 0 for v in sizes[obj]):
                raise ValueError('invalid_object_footprint')
            predicates.add(('at', obj, location or 'outside'))
        predicates.update(('free', name) for name in self.regions
                          if name not in occupied and name not in obs.blocked_regions)
        self._footprints = sizes
        self._last = TaskState(frozenset(predicates), obs.binding)
        return self._last

    def describe(self) -> dict:
        state = self.snapshot()
        return {'planning_scope': self.planning_scope, 'motion_refinement': self.motion_refinement,
                'inputs': self.input_mode, 'task_planning_available': True,
                'objects': sorted(self._footprints), 'regions': sorted(self.regions),
                'locations': [{'object_name': p[1], 'region': p[2]}
                              for p in sorted(state.predicates) if p[0] == 'at'],
                'free_regions': sorted(p[1] for p in state.predicates if p[0] == 'free'),
                'instruction_forms': ['move OBJECT to REGION; move OBJECT to REGION', 'swap OBJECT and OBJECT']}

    def compile_goal(self, instruction: str) -> GoalSpec:
        if self._last is None:
            raise ValueError('task_snapshot_required')
        text = instruction.strip().casefold().rstrip('.')
        swap = re.fullmatch(r'swap (.+?) and (.+)', text)
        if swap:
            a, b = swap.groups()
            locations = {p[1]: p[2] for p in self._last.predicates if p[0] == 'at'}
            if a == b or a not in locations or b not in locations:
                raise ValueError('unknown_task_object')
            if locations[a] not in self.regions or locations[b] not in self.regions:
                raise ValueError('swap_requires_two_grounded_regions')
            protected = frozenset(p for p in self._last.predicates if p[0] == 'at' and p[1] not in {a, b})
            return GoalSpec(frozenset({('at', a, locations[b]), ('at', b, locations[a])}), protected)
        goals, assigned = set(), set()
        for clause in text.split(';'):
            match = re.fullmatch(r'(?:move|put|place) (.+?) (?:to|in|on) (.+)', clause.strip())
            if match is None:
                raise ValueError('unsupported_task_instruction')
            obj, target = match.groups()
            if obj not in self._footprints or target not in self.regions:
                raise ValueError('unknown_task_role')
            if obj in assigned or any(p[2] == target for p in goals):
                raise ValueError('conflicting_task_goals')
            assigned.add(obj)
            goals.add(('at', obj, target))
        protected = frozenset(p for p in self._last.predicates if p[0] == 'at' and p[1] not in assigned)
        return GoalSpec(frozenset(goals), protected)

    def operators(self, state: TaskState) -> tuple[Operator, ...]:
        result = []
        for p in sorted(p for p in state.predicates if p[0] == 'at'):
            _, obj, source = p
            for free in sorted(p for p in state.predicates if p[0] == 'free'):
                adds = {('at', obj, free[1])}
                if source in self.regions:
                    adds.add(('free', source))
                result.append(Operator(f'transfer:{obj}:{source}:{free[1]}', 'transfer',
                    frozenset({p, free}), frozenset(adds), frozenset({p, free})))
        return tuple(result)

    def _selection(self, operator):
        targets = [p for p in operator.adds if p[0] == 'at']
        if (operator.name != 'transfer' or len(targets) != 1 or targets[0][1] not in self._footprints
                or targets[0][2] not in self.regions):
            raise ValueError('invalid_transfer_operator')
        return targets[0][1], self.regions[targets[0][2]]

    def feasible(self, state, operator):
        obj, target = self._selection(operator)
        return (operator.requires <= state.predicates and ('free', target.name) in state.predicates
                and all(self._footprints[obj][i] <= target.bounds[1][i] - target.bounds[0][i] for i in range(2)))

    def execute(self, operator):
        # These calls are read-only preflight. Only transfer may actuate; never
        # catch its unknown exceptions as proof that execution did not start.
        try:
            if not self.feasible(self.snapshot(), operator):
                return TaskExecutionResult(False, False, 'task_feasibility_failed')
            selection = self._selection(operator)
        except Exception as error:
            return TaskExecutionResult(False, False, provider_failure_code(error))
        return self.transfer(*selection)
