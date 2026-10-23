# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Planning-only replay of recorded native diagnostics. Never a live robot input."""
from __future__ import annotations

import threading
import uuid

import mujoco
import numpy as np

from emet.controller.manipulation.native_scene import model_signature
from emet.controller.task.tamp.native_adapter import NativeTableAdapter


class _NoActuation:
    def get_emet_session(self):
        return {}

    def __getattr__(self, name):
        raise RuntimeError('offline_replay_cannot_actuate')


class RecordedNativeTablePlanner(NativeTableAdapter):
    """Bind a captured state to private data without forging fresh observations.

    Model signature, dimensions and finite values must match. No server/client is
    constructed; synchronizing means restoring the record, not reading hardware.
    The parent's pure planner is reused, but execution is unconditionally blocked.
    """
    def __init__(self, snapshot: dict, *, model=None):
        if model is None:
            from emet.simulation.mujoco_server_stretch import default_scene_xml_path
            model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
        self.model, self.signature = model, model_signature(model)
        if (snapshot.get('source') != 'simulator_ground_truth'
                or snapshot.get('model_signature') != self.signature):
            raise ValueError('recorded_native_model_mismatch')
        self._q = np.asarray(snapshot.get('qpos'), dtype=float).copy()
        self._v = np.asarray(snapshot.get('qvel'), dtype=float).copy()
        self._time = float(snapshot.get('sim_time', float('nan')))
        if (self._q.shape != (model.nq,) or self._v.shape != (model.nv,)
                or not np.isfinite(self._q).all() or not np.isfinite(self._v).all()
                or not np.isfinite(self._time) or self._time < 0):
            raise ValueError('invalid_recorded_native_state')
        self.data = mujoco.MjData(model)
        self.robot = _NoActuation()
        self.boot, self.namespace = 'offline', 'offline:' + uuid.uuid4().hex
        self.lock = threading.RLock()
        self.tasks, self.plans, self.events, self.planning_rejections = {}, {}, [], {}
        self.planning_ik_calls = 0
        self.synchronize(self.data)

    def synchronize(self, data):
        data.qpos[:], data.qvel[:], data.time = self._q, self._v, self._time
        mujoco.mj_kinematics(self.model, data)

    def _data(self, **extra):
        return {**super()._data(**extra), 'mode': 'offline_planning',
                'inputs': 'recorded_ground_truth', 'execution_available': False,
                'physical_acceptance': None}

    def plan_transfer(self, object_body, support_body, placement_region):
        with self.lock:
            self.tasks['recorded_transfer'] = (object_body, support_body, 'recorded object', 'recorded support')
            return self.plan(task_ref='recorded_transfer', placement_region=placement_region)

    def execute(self, *_):
        raise RuntimeError('offline_replay_cannot_execute')
