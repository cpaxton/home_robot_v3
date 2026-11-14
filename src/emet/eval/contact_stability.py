# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Privileged native-contact diagnostics, separate from agent task scores.

Collision reconstruction uses private MjData. No contact force or passive-joint
reading here is advertised as a hardware sensor. Recording adds measured wall
time; comparisons must use the same recorder on both conditions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from emet.controller.task.tamp.native_adapter import NativeTableAdapter
from emet.controller.task.tamp.native_policy import FINE_GRASP_YAWS

GRASP_POLICIES = {
    'baseline': NativeTableAdapter.grasp_yaw_offsets,
    'fine_yaw': FINE_GRASP_YAWS,
}


class ContactTrace:
    """Record selected controller events without modifying execution/planning data."""

    phases = frozenset(('grasp_complete', 'gripper_settled', 'lift', 'lift_complete',
                        'payload_retention', 'preplace_complete', 'place_complete', 'release',
                        'loaded_grasp_check', 'loaded_grasp_binding'))

    def __init__(self, model, path: Path):
        self.model, self.scratch = model, mujoco.MjData(model)
        self.stream = path.open('x')
        self.records = 0
        self.max_record_wall_s = 0.
        self.extrema = {}

    def close(self):
        self.stream.close()

    def record(self, obj, executor, row, *, motion_phase=None):
        phase = row.get('phase', '')
        if phase not in self.phases and phase not in ('release_adjustment', 'release_adjustment_complete', 'release_open_verified') and not phase.startswith(('grasp_replan', 'release_plan')):
            return
        start = time.monotonic()
        data = executor.data
        event = {'source': 'simulator_ground_truth_debug', 'sim_time': float(data.time),
                 'object_body': obj, 'motion_phase': motion_phase, 'event': row}
        if phase.startswith(('grasp_replan', 'release_plan')):
            event['kind'] = 'planning_targets'
        else:
            event['kind'] = 'measured_qpos_private_fk_and_collision'
            scratch = self.scratch
            mujoco.mj_copyData(scratch, self.model, data)
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_collision(self.model, scratch)
            body = self.model.body(obj)
            ee, payload = scratch.body(executor.ee_body), scratch.body(obj)
            rotation = ee.xmat.reshape(3, 3).T
            relative = np.eye(4)
            relative[:3, :3] = rotation @ payload.xmat.reshape(3, 3)
            relative[:3, 3] = rotation @ (payload.xpos - ee.xpos)
            contacts = []
            for contact in scratch.contact:
                bodies = self.model.geom_bodyid[contact.geom]
                if body.id in bodies:
                    contacts.append({'bodies': [self.model.body(int(b)).name for b in bodies],
                                     'distance_m': float(contact.dist)})
            event.update(qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                         object_position_m=payload.xpos.tolist(), object_rotation=payload.xmat.reshape(3, 3).tolist(),
                         relative_pose=relative.tolist(), contacts=contacts)
            expected = executor.collision.payload_transform
            if expected is not None:
                displacement = float(np.linalg.norm(relative[:3, 3] - expected[:3, 3]))
                angle = float(np.arccos(np.clip((np.sum(relative[:3, :3] * expected[:3, :3])-1)/2, -1, 1)))
                event.update(translation_residual_m=displacement, rotation_residual_rad=angle,
                             payload_reference=expected.tolist())
                maxima = self.extrema.setdefault(obj, {'translation_m': 0., 'rotation_rad': 0.})
                maxima['translation_m'] = max(maxima['translation_m'], displacement)
                maxima['rotation_rad'] = max(maxima['rotation_rad'], angle)
        self.stream.write(json.dumps(event, allow_nan=False) + '\n')
        self.stream.flush()
        self.records += 1
        self.max_record_wall_s = max(self.max_record_wall_s, time.monotonic() - start)


class TracedNativeAdapter(NativeTableAdapter):
    """Reuse production execution with explicit diagnostic policy selections."""

    def __init__(self, robot, *, model, trace, policy=None, payload_binding='preserve', release_limits=None, native_policy='legacy'):
        if payload_binding not in ('preserve', 'verified_lift_keep', 'verified_lift'):
            raise ValueError('invalid_payload_binding_policy')
        super().__init__(robot, model=model, policy=native_policy)
        self.trace = trace
        if policy is None:
            return
        self.policy = 'diagnostic'
        self.grasp_yaw_offsets = GRASP_POLICIES[policy]
        self.payload_binding = payload_binding
        self.release_limits = release_limits

    def _executor(self, obj):
        executor = super()._executor(obj)
        original = executor.event
        execute_path = executor._execute_path
        context = [None]

        def path(phase, waypoints):
            context[0] = phase
            try:
                return execute_path(phase, waypoints)
            finally:
                context[0] = None

        def record(**row):
            original(**row)
            self.trace.record(obj, executor, row, motion_phase=context[0])

        executor._execute_path = path
        executor.event = record
        return executor
