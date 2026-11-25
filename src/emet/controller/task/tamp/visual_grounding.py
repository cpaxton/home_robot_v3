# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Shared visual proposals rebound to fresh geometry before native planning."""
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.perception.grasps.visual_grounding import grounding_profile, query_pick_hint, refine_object_hint

from .observed_capture import capture_arrays
from .transfer_grounding import TransferGroundingUnavailable, select_transfer


@dataclass(frozen=True)
class GroundingResolution:
    sample: object
    selection: object
    profile: dict
    decision: dict
    reference_capture_id: str


class VisualTransferGrounder:
    """No robot commands or benchmark metadata; explicit phase permission required."""

    def __init__(self, *, endpoint, model, output, can_query, clock=time.monotonic):
        if not callable(can_query):
            raise ValueError('visual_query_phase_guard_required')
        self.endpoint, self.model, self.output = endpoint, model, Path(output)
        self.can_query, self.clock = can_query, clock
        self._lock = threading.Lock()
        self._sequence = 0

    def _permit(self):
        if self.can_query() is not True:
            raise ObservationUnavailable('visual_query_phase_unavailable')

    def resolve(self, reference, read_fresh, task):
        with self._lock:
            return self._resolve(reference, read_fresh, task)

    def _resolve(self, reference, read_fresh, task):
        self._permit()
        self._sequence += 1
        root = self.output/f'grounding-{self._sequence:04d}'
        root.mkdir(parents=True, exist_ok=False)
        feedback = None
        last_reason, last_role = None, None
        for attempt in range(3):
            self._permit()
            if attempt:
                reference = read_fresh()
            reference.capture.require(now=self.clock())
            output = root/f'attempt-{attempt+1}'
            output.mkdir()
            try:
                views = {name: obs.rgb for name, obs in reference.observations.items()}
                if not views:
                    raise ObservationUnavailable('observation_unavailable')
                decision = query_pick_hint(next(iter(views.values())), endpoint=self.endpoint, model=self.model,
                    object_name=task.object_name, destination_name=task.receptacle_name, output=output,
                    views=views, feedback=feedback, infer_support_from_depth=True, estimate_grasp=False)
                self._permit()
                decision = refine_object_hint(reference.observations[decision.get('camera', next(iter(views)))].rgb,
                    decision, endpoint=self.endpoint, model=self.model, object_name=task.object_name,
                    output=output/'refinement', estimate_grasp_site=True)
                profile = grounding_profile(decision, infer_support_from_depth=True, destination_below_source=False)
                prior = select_transfer(reference, profile=profile, grow_object_region=True)
                self._permit()
                fresh = read_fresh()
                fresh.capture.require(now=self.clock(), after=reference.capture.latest)
                if fresh.capture.observation_id == reference.capture.observation_id:
                    raise ObservationUnavailable('capture_replayed')
                for camera in {prior.object_camera, prior.source_camera, prior.destination_camera}:
                    old = reference.observations[camera]
                    current = fresh.observations.get(camera)
                    if (current is None or not np.array_equal(old.camera_K, current.camera_K)
                            or not np.allclose(old.camera_pose, current.camera_pose, atol=.005, rtol=0)):
                        raise ObservationUnavailable('grounding_view_changed')
                selected = select_transfer(fresh, profile=profile, grow_object_region=True)
                old_points = capture_arrays(reference.observations[prior.object_camera])[2][prior.object_mask]
                new_points = capture_arrays(fresh.observations[selected.object_camera])[2][selected.object_mask]
                if np.linalg.norm(np.median(old_points, axis=0)-np.median(new_points, axis=0)) > .03:
                    raise ObservationUnavailable('grounding_object_changed')
                (output/'binding.json').write_text(json.dumps({
                    'reference_capture_id': reference.capture.observation_id,
                    'geometry_capture_id': fresh.capture.observation_id,
                    'semantic_identity_verified': False, 'object_cad': False,
                    'geometry_reacquired': True})+'\n')
                return GroundingResolution(fresh, selected, profile, decision, reference.capture.observation_id)
            except OSError as error:
                raise ObservationUnavailable('visual_grounding_service_unavailable') from error
            except (ValueError, ObservationUnavailable, KeyError, TypeError) as error:
                last_reason = 'invalid_visual_grounding_response' if isinstance(error, (KeyError, TypeError)) else str(error)
                last_role = getattr(error, 'region_role', None)
                feedback = last_reason
                if last_role == 'source':
                    feedback += ': no usable support around the object; revise its region or choose a clearer object camera. Do not predict a support box.'
                elif last_role == 'object':
                    feedback += ': object region lacks a unique measured component; revise its box or use a clearer object camera.'
                elif last_role == 'destination':
                    feedback += ': destination region lacks horizontal support; choose its visible support surface, not a wall, rim, rack or robot.'
                (output/'rejection.json').write_text(json.dumps({'reason': last_reason, 'region_role': last_role})+'\n')
                if last_reason == 'visual_query_phase_unavailable':
                    raise
        if last_role is not None:
            raise TransferGroundingUnavailable(last_reason, last_role)
        raise ObservationUnavailable(last_reason)
