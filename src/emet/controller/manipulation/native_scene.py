# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicit privileged simulator snapshots for native-debug manipulation."""
from __future__ import annotations

import copy
import hashlib
import time

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable


def model_signature(model):
    """Bind geometry, dynamics and actuator mapping, excluding live initial state."""
    digest = hashlib.sha256(model.names)
    for field in ('body_parentid', 'body_pos', 'body_quat', 'body_mass', 'body_inertia',
                  'jnt_type', 'jnt_bodyid', 'jnt_pos', 'jnt_axis', 'jnt_range',
                  'geom_type', 'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_size',
                  'geom_contype', 'geom_conaffinity', 'geom_friction', 'mesh_vert',
                  'mesh_face', 'actuator_trnid', 'actuator_ctrlrange', 'actuator_gainprm',
                  'actuator_biasprm', 'eq_data', 'pair_geom1', 'pair_geom2', 'exclude_signature'):
        values = np.asarray(getattr(model, field))
        if field in ('body_pos', 'body_quat'):
            # A free body's initial pose is replaced by measured qpos in FK.
            # Bind its geometry/inertia, but allow different fixture initial poses.
            values = values.copy()
            free_bodies = model.jnt_bodyid[model.jnt_type == mujoco.mjtJoint.mjJNT_FREE]
            values[free_bodies] = 0
        digest.update(values.tobytes())
    return digest.hexdigest()


def read_snapshot(robot, signature=None):
    """A physics capture timestamp is required, not merely fresh republication."""
    with robot._state_lock:
        state = copy.deepcopy(robot._state)
        received = getattr(robot, '_state_received_monotonic', None)
    if received is None or not 0 <= time.monotonic() - received <= 2:
        raise ObservationUnavailable('stale_observation')
    value = state.get('native_tamp_snapshot')
    if not isinstance(value, dict) or value.get('source') != 'simulator_ground_truth':
        raise ObservationUnavailable('native_execution_unavailable')
    age = time.time() - float(value.get('captured_at', float('nan')))
    if not np.isfinite(age) or not 0 <= age <= 2:
        raise ObservationUnavailable('stale_observation')
    if signature is not None and value.get('model_signature') != signature:
        raise ObservationUnavailable('native_model_mismatch')
    boot = (state.get('command_protocol') or {}).get('server_boot_id')
    if not boot:
        raise ObservationUnavailable('server_identity_missing')
    return str(boot), value
