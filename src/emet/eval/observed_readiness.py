# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Live sensor readiness diagnostics; never promote discovery into execution."""
from __future__ import annotations

import json
import time

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.rigid_rgbd_tracking import reference_features
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.motion.observed_workspace import ObservedWorkspace


def capture_metadata(observation):
    """Preserve original stream identities and receive times, without clock mapping."""
    return {'received_monotonic': observation.received_monotonic, 'image_timing': observation.image_timing}


def assess_live_readiness(*, observation, reference_rgb, reference_mask, tools, refreshes,
                          execution_source_installed=False):
    """Audit this camera-only harness, not arbitrary injected-provider qualification.

    Complete geometry, timing bounds, support and occupancy provenance are not
    supplied by this harness. Passing image/feature checks cannot qualify them.
    Default physical planning must continue rejecting an uninstalled source.
    """
    _, depth, _ = capture_arrays(observation)
    age = time.monotonic() - observation.received_monotonic
    valid_depth = np.isfinite(depth) & (depth > .1) & (depth <= 5.)
    features = {'state': 'unknown', 'reason': 'rigid_tracking_insufficient'}
    try:
        _, pixels = reference_features(reference_rgb, reference_mask)
        features = {'state': 'passed', 'reference_landmarks': len(pixels),
                    'scope': 'reference feature admission only; no measured pose/retention qualification'}
    except ObservationUnavailable as exc:
        features['reason'] = str(exc)
    pose = np.asarray(observation.camera_pose)
    center = pose[:3, :3] @ np.array([0., 0., .75]) + pose[:3, 3]
    workspace = ObservedWorkspace.from_depth(depth, observation.camera_K, pose,
        workspace=[center-.3, center+.3], resolution=.025)
    counts = {name: int(np.count_nonzero(workspace.cells == index))
              for index, name in enumerate(('unknown', 'free', 'occupied'))}
    planned = json.loads(tools['plan_pick_place'].func(object_name='diagnostic object', receptacle_name='diagnostic support'))
    execution = json.loads(tools['execute_pick_place_plan'].func(plan_ref='not-an-executable-plan'))
    fail_closed = (planned['code'] == 'observed_scene_unavailable'
                   and execution['code'] == 'observed_scene_unavailable')
    return {
        'scope': 'live assembly prerequisite audit, not a manipulation or agent benchmark',
        'qualification': 'not_qualified',
        'task_success_assessed': False,
        'execution_source_installed': execution_source_installed,
        'gates': {
            'live_rgbd': {'state': 'passed' if 0 <= age <= 2 and valid_depth.any() else 'failed', 'receive_age_s': age,
                          'valid_depth_fraction': float(valid_depth.mean()),
                          'capture': capture_metadata(observation)},
            'surface_refresh': {'state': 'passed' if refreshes and all(refreshes) else 'failed',
                                'passed': sum(bool(value) for value in refreshes), 'attempted': len(refreshes)},
            'reference_features': features,
            'depth_workspace': {'state': 'passed', 'cells': counts,
                                'scope': 'depth carving only; not target/robot-separated execution geometry'},
            'capture_encoder_alignment': {'state': 'not_configured',
                'reason': 'stream timestamps do not supply qualified CaptureInterval or alignment bounds'},
            'complete_payload_geometry': {'state': 'not_configured',
                'reason': 'partial visible surface is not a complete shape prior'},
            'robot_target_occupancy_provenance': {'state': 'not_configured'},
            'support_grounding': {'state': 'not_configured', 'reason': 'single object seed, no support seed'},
            'visual_manipulation_effects': {'state': 'not_assessed', 'reason': 'no manipulation attempted'},
            'default_execution_boundary': {'state': 'passed' if fail_closed else 'failed',
                'planning_code': planned['code'], 'execution_code': execution['code']},
        },
    }
