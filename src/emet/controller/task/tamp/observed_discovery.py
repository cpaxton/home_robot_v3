# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Read-only observed task proposals; neither reachability nor execution admission.

Providers must pair tracked objects with observed supports, not infer supports
from labels alone. No simulator metadata or fallback provider is consulted here.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.task.tamp.api import response


@dataclass(frozen=True)
class ObservedPositionEvidence:
    """A discovery position from segmented depth; no object rotation is inferred."""
    position: np.ndarray
    received_monotonic: float
    source: str
    observation_id: str

    def __post_init__(self):
        point = np.asarray(self.position, dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or not np.isfinite(self.received_monotonic) or not self.observation_id
                or self.source not in {'rgbd_pointcloud', 'estimated_depth_pointcloud'}):
            raise ValueError('invalid_observed_position')
        object.__setattr__(self, 'position', np.frombuffer(point.tobytes(), dtype=float))

    def require(self, inputs: str, *, now: float, max_age_s: float = 2.) -> np.ndarray:
        if inputs != 'observed':
            raise ObservationUnavailable('input_provider_mismatch')
        if not 0 <= now - self.received_monotonic <= max_age_s:
            raise ObservationUnavailable('object_observation_stale')
        return self.position.copy()


@dataclass(frozen=True)
class ObservedTaskCandidate:
    object_name: str
    receptacle_name: str
    object_pose: ObjectPoseEvidence | ObservedPositionEvidence
    support_pose: ObjectPoseEvidence | ObservedPositionEvidence

    def __post_init__(self):
        if any(not isinstance(name, str) or not name.strip()
               for name in (self.object_name, self.receptacle_name)):
            raise ValueError('invalid_observed_task_label')
        if not all(isinstance(pose, (ObjectPoseEvidence, ObservedPositionEvidence)) for pose in (self.object_pose, self.support_pose)):
            raise ValueError('invalid_observed_task_pose')


class ObservedTaskProvider(Protocol):
    input_mode: str

    def observe_tasks(self) -> tuple[ObservedTaskCandidate, ...]:
        """Return one coherent snapshot of observed object/support proposals.

        Retain original observation receive times; querying or republishing memory
        must not renew their freshness. Occluded/unknown entities are omitted.
        """
        ...


def discover_observed_tasks(provider: ObservedTaskProvider | None, object_filter: str = '',
                            robot: str = '', *, now: float | None = None) -> dict:
    """Validate the entire snapshot before publishing opaque discovery handles.

    Handles are proposals only. Native planning remains blocked independently;
    a provider cannot enable motion by returning a candidate or capability flag.
    """
    if provider is None:
        return response('scene_tasks', code='observed_scene_unavailable')
    if provider.input_mode != 'observed':
        return response('scene_tasks', code='input_provider_mismatch')
    if robot:
        return response('scene_tasks', code='observed_robot_filter_unsupported')
    snapshot = provider.observe_tasks()
    if not isinstance(snapshot, tuple) or not all(isinstance(c, ObservedTaskCandidate) for c in snapshot):
        return response('scene_tasks', code='invalid_observed_task_snapshot')
    now = time.monotonic() if now is None else now
    rows = []
    try:
        for candidate in snapshot:
            obj = candidate.object_pose.require('observed', now=now)
            support = candidate.support_pose.require('observed', now=now)
            if object_filter.casefold() not in candidate.object_name.casefold():
                continue
            rows.append({
                'task_ref': 'observed-task:' + uuid.uuid4().hex,
                'object_name': candidate.object_name,
                'receptacle_name': candidate.receptacle_name,
                'object_position': (obj if obj.shape == (3,) else obj[:3, 3]).tolist(),
                'support_position': (support if support.shape == (3,) else support[:3, 3]).tolist(),
                'sources': {'object': candidate.object_pose.source, 'support': candidate.support_pose.source},
                'observation_age_s': {'object': now - candidate.object_pose.received_monotonic,
                                      'support': now - candidate.support_pose.received_monotonic},
            })
    except ObservationUnavailable as exc:
        return response('scene_tasks', code=str(exc))
    return response('scene_tasks', data={
        'tasks': rows, 'inputs': 'observed', 'discovery_only': True,
        'execution_available': False, 'planning_available': False,
        'unavailable_reason': 'observed_execution_unavailable',
        'assistance': [],
    })


class RgbdTaskProvider:
    """Fresh single-frame proposals from the existing segmentation/depth pipeline.

    Small segmented objects are paired with broad horizontal patches observed
    in that same frame. Patch extents are proposal heuristics; holes, occlusion,
    load bearing, graspability and collision clearance are not certified.
    Accumulated graph nodes and cached semantic localizations are never used.
    """
    input_mode = 'observed'

    def __init__(self):
        self._lock = threading.Lock()
        self._candidates = ()

    def observe_tasks(self):
        with self._lock:
            return self._candidates

    def update(self, observations, *, received_monotonic=None, source='rgbd_pointcloud'):
        candidates = []
        if received_monotonic is not None and np.isfinite(received_monotonic):
            objects, supports = [], []
            for index, observation in enumerate(observations):
                points = observation.points_3d
                if hasattr(points, 'detach'):
                    points = points.detach().cpu().numpy()
                points = np.asarray(points, dtype=float)
                if points.ndim != 2 or points.shape[1] != 3:
                    continue
                points = points[np.isfinite(points).all(axis=1)]
                if len(points) < 10 or not observation.label.strip():
                    continue
                lo, hi = np.quantile(points, [.05, .95], axis=0)
                span = hi - lo
                identity = f'frame:{observation.timestep}:segment:{index}'
                if .01 <= span.max() <= .30:
                    objects.append((index, observation.label, ObservedPositionEvidence(
                        np.median(points, axis=0), received_monotonic, source, identity)))
                # Require depth points covering a horizontal upper band, not a
                # furniture label or an accumulated bounding-box top.
                top = np.quantile(points[:, 2], .90)
                patch = points[np.abs(points[:, 2] - top) <= .01]
                if len(patch) < 30:
                    continue
                lower, upper = np.quantile(patch[:, :2], [.05, .95], axis=0)
                extent = upper - lower
                if (extent >= .15).all() and np.prod(extent) >= .03:
                    position = np.r_[(lower + upper) / 2, np.median(patch[:, 2])]
                    supports.append((index, observation.label, ObservedPositionEvidence(
                        position, received_monotonic, source, identity)))
            for object_index, object_name, object_position in objects:
                for support_index, support_name, support_position in supports:
                    if object_index != support_index:
                        candidates.append(ObservedTaskCandidate(
                            object_name, support_name, object_position, support_position))
        with self._lock:
            self._candidates = tuple(candidates)
