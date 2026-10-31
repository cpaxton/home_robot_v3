# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Observed multi-step region transfers through the same native JSON adapter.

All configured roles must be currently observed and visually settled. Unobserved
workspace is blocked, not a free destination; object absence is never 'outside'.
"""
from __future__ import annotations

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import evaluate_place, rotation_distance

from .task_contracts import SceneBinding, fingerprint
from .task_session import TaskExecutionResult
from .transfer_provider import RegionTransferProvider, TransferObservation


class ObservedRegionTransferProvider(RegionTransferProvider):
    def __init__(self, adapter, regions, *, robot_profile_id, reset_id):
        if not robot_profile_id or not reset_id:
            raise ValueError('observed_robot_profile_required')
        self.adapter, self.reset_id = adapter, reset_id
        self._profile = fingerprint({'profile': robot_profile_id, 'inputs': 'observed'})
        self.roles = {task.object_name.strip().casefold(): task for task in adapter.tasks.values()}
        if (len(self.roles) != len(adapter.tasks)
                or len({task.support_ref for task in self.roles.values()}) != 1):
            raise ValueError('observed_transfer_roles_require_one_support')
        self._revision, self._reference = 0, None
        self._calibration = None
        super().__init__(tuple(regions), observe=self._observe, transfer=self._transfer, input_mode='observed')

    def _observe(self):
        with self.adapter.lock:
            snapshots = {}
            for role, task in self.roles.items():
                initial = self.adapter._observe(task)
                frames = tuple(self.adapter.source.collect(task.object_ref, task.support_ref,
                                                           after=initial.frame.capture.latest))
                result = evaluate_place(initial.frame, frames, support_bounds=initial.support_bounds,
                    after=initial.frame.capture.latest, now=self.adapter.clock())
                if not result.satisfied:
                    raise ObservationUnavailable('task_object_not_settled_on_support')
                # Final fresh capture, not the old pre-settling state, binds predicates.
                current = self.adapter._observe(task)
                a, b = frames[-1].payload, current.frame.payload
                if (a.reference_id != b.reference_id or a.revision != b.revision
                        or np.linalg.norm(a.pose.world_from_object[:3, 3] - b.pose.world_from_object[:3, 3]) > .002
                        or rotation_distance(a.pose.world_from_object[:3, :3], b.pose.world_from_object[:3, :3]) > .01):
                    raise ObservationUnavailable('task_object_not_settled_on_support')
                snapshots[role] = current
            revisions = {s.scene_revision for s in snapshots.values()}
            calibrations = {s.frame.capture.calibration_id for s in snapshots.values()}
            if len(revisions) != 1 or len(calibrations) != 1:
                raise ObservationUnavailable('observed_scene_unavailable')
            calibration = next(iter(calibrations))
            if self._calibration is not None and self._calibration != calibration:
                raise ValueError('scene_identity_changed')
            self._calibration = calibration
            now = self.adapter.clock()
            locations, footprints, blocked = [], [], set()
            boxes, positions = {}, {}
            for role, snapshot in snapshots.items():
                snapshot.frame.require(now=now)
                box = snapshot.frame.bounds()
                boxes[role] = box
                positions[role] = snapshot.frame.payload.pose.world_from_object.copy()
                footprints.append((role, tuple(box[1, :2] - box[0, :2])))
                contained, intersecting = [], []
                for name, region in self.regions.items():
                    bounds = np.asarray(region.bounds)
                    if np.all(box[0, :2] >= bounds[0]) and np.all(box[1, :2] <= bounds[1]):
                        contained.append(name)
                    if np.all(box[0, :2] < bounds[1]) and np.all(box[1, :2] > bounds[0]):
                        intersecting.append(name)
                    if (np.any(bounds[0] < snapshot.support_bounds[0, :2])
                            or np.any(bounds[1] > snapshot.support_bounds[1, :2])):
                        raise ValueError('task_region_outside_support')
                if len(contained) > 1 or contained != intersecting:
                    raise ValueError('ambiguous_region_occupancy')
                locations.append((role, contained[0] if contained else None))
            occupied = {location for _, location in locations if location is not None}
            for name, region in self.regions.items():
                if name in occupied:
                    continue
                # Every required region cell up through the largest possible
                # payload height must be known free in every coherent snapshot.
                height = max(box[1, 2] - box[0, 2] for box in boxes.values())
                for snapshot in snapshots.values():
                    top = snapshot.support_bounds[1, 2]
                    query = np.array([[*region.bounds[0], top + .001], [*region.bounds[1], top + height + .02]])
                    if snapshot.scene.collides(query):
                        blocked.add(name)
            reference = (next(iter(revisions)), positions)
            previous = self._reference
            if (previous is None or previous[0] != reference[0] or any(
                    np.linalg.norm(previous[1][r][:3, 3] - p[:3, 3]) > .005
                    or rotation_distance(previous[1][r][:3, :3], p[:3, :3]) > .02
                    for r, p in positions.items())):
                self._reference = reference
                self._revision += 1
            binding = SceneBinding(fingerprint({'calibration': calibration}), self._profile,
                self.adapter.namespace, self.reset_id, self._revision)
            return TransferObservation(binding, tuple(locations), tuple(footprints), frozenset(blocked), input_mode='observed')

    def _transfer(self, role, region):
        task = self.roles[role]
        planned = self.adapter.plan(object_name=task.object_name, receptacle_name=task.receptacle_name,
                                    placement_region=region.bounds)
        if planned['status'] != 'ok':
            return TaskExecutionResult(False, False, planned['code'] if planned['code'] != 'operation_failed'
                                       else 'task_motion_planning_failed')
        result = self.adapter.execute(planned['data']['plan_ref'])
        return TaskExecutionResult(result['status'] == 'ok', result.get('data', {}).get('execution_started') is True,
            result['code'] if result['code'] not in {'ok', 'operation_failed'} else 'task_execution_failed')
