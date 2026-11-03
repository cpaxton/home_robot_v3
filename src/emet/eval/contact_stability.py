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

GRASP_POLICIES = {
    'baseline': NativeTableAdapter.grasp_yaw_offsets,
    'fine_yaw': (0., -.175, .175, -.35, .35, -.7, .7, -1.57, 1.57),
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
        if phase not in self.phases and not phase.startswith('grasp_replan'):
            return
        start = time.monotonic()
        data = executor.data
        event = {'source': 'simulator_ground_truth_debug', 'sim_time': float(data.time),
                 'object_body': obj, 'motion_phase': motion_phase, 'event': row}
        if phase.startswith('grasp_replan'):
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
    """Reuse the production planner/executor; change only declared yaw candidates."""

    def __init__(self, robot, *, model, trace, policy, payload_binding='preserve'):
        if payload_binding not in ('preserve', 'verified_lift_keep', 'verified_lift'):
            raise ValueError('invalid_payload_binding_policy')
        self.trace = trace
        self.grasp_yaw_offsets = GRASP_POLICIES[policy]
        self.payload_binding = payload_binding
        super().__init__(robot, model=model)

    def _bind_loaded_payload(self, plan, *, rebind=True):
        """One measured primitive handoff; never renew a violated/creeping bound.

        The caller already verified the lift and one-second bilateral hold.
        Check the original bound throughout an additional 0.3 s interval, with
        <=2 mm / 0.01 rad change relative to its first sample. Placement still
        replans all segments using the resulting measured transform.
        """
        executor = plan.executor
        collision = executor.collision
        if (getattr(executor, '_loaded_binding_done', False)
                or executor.payload_body != plan.object_body or collision.payload_body != plan.object_body
                or collision.payload_parent != executor.ee_body):
            return False
        scratch = mujoco.MjData(self.model)
        self.synchronize(self.data)
        active = executor._settling_window(2.)
        anchor, started = None, float(self.data.time)
        while active():
            self.synchronize(self.data)
            if not executor.payload_retained():
                return False
            mujoco.mj_copyData(scratch, self.model, self.data)
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_collision(self.model, scratch)
            obj, ee = scratch.body(plan.object_body), scratch.body(executor.ee_body)
            touched = {int(b) for c in scratch.contact if obj.id in self.model.geom_bodyid[c.geom]
                       for b in self.model.geom_bodyid[c.geom] if b != obj.id}
            tips = {self.model.body(n).id for n in ('rubber_tip_left', 'rubber_tip_right')}
            relative = np.eye(4)
            relative[:3, :3] = ee.xmat.reshape(3, 3).T @ obj.xmat.reshape(3, 3)
            relative[:3, 3] = ee.xmat.reshape(3, 3).T @ (obj.xpos - ee.xpos)
            if anchor is None:
                anchor = relative.copy()
                started = float(self.data.time)
            drift = float(np.linalg.norm(relative[:3, 3] - anchor[:3, 3]))
            angle = float(np.arccos(np.clip((np.sum(relative[:3, :3] * anchor[:3, :3])-1)/2, -1, 1)))
            stable = bool(touched == tips and self.model.body(plan.support_body).id not in touched
                          and drift <= .002 and angle <= .01)
            executor.event(phase='loaded_grasp_check', accepted=stable,
                           interval_translation_m=drift, interval_rotation_rad=angle)
            if not stable:
                return False
            if self.data.time - started >= .3:
                previous = collision.payload_transform.copy()
                if rebind:
                    collision.set_payload(self.model, self.data, plan.object_body, executor.ee_body)
                executor._loaded_binding_done = True
                executor.event(phase='loaded_grasp_binding', accepted=True,
                               previous_reference=previous.tolist(), reference=collision.payload_transform.tolist(),
                               reference_changed=rebind, placement_replan_required=True)
                return True
            time.sleep(.05)
        return False

    def _verify(self, plan, *, released, seconds):
        verified = super()._verify(plan, released=released, seconds=seconds)
        if not verified or released or self.payload_binding == 'preserve':
            return verified
        return self._bind_loaded_payload(plan, rebind=self.payload_binding == 'verified_lift')

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
