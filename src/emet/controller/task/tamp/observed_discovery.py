# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Read-only observed task proposals; neither reachability nor execution admission.

Providers must pair tracked objects with observed supports, not infer supports
from labels alone. No simulator metadata or fallback provider is consulted here.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.task.tamp.api import response


@dataclass(frozen=True)
class ObservedTaskCandidate:
    object_name: str
    receptacle_name: str
    object_pose: ObjectPoseEvidence
    support_pose: ObjectPoseEvidence

    def __post_init__(self):
        if any(not isinstance(name, str) or not name.strip()
               for name in (self.object_name, self.receptacle_name)):
            raise ValueError('invalid_observed_task_label')
        if not all(isinstance(pose, ObjectPoseEvidence) for pose in (self.object_pose, self.support_pose)):
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
                'object_position': obj[:3, 3].tolist(),
                'support_position': support[:3, 3].tolist(),
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
