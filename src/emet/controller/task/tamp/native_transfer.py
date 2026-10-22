# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Privileged stationary Stretch grounding and bounded native motion refinement."""
from __future__ import annotations

import json
import logging

import mujoco
import numpy as np

from emet.motion.mujoco_collision import body_subtree
from emet.simulation.sim_object_placements import placements_from_mujoco_model

from .native_adapter import NativeTableAdapter
from .task_budget import BudgetExhausted
from .task_contracts import SceneBinding, fingerprint
from .task_session import TaskExecutionResult
from .transfer_provider import RegionTransferProvider, TransferObservation, TransferRegion


def validate_native_task_config(config):
    fields = {'schema_version', 'input_mode', 'robot', 'reset_id', 'objects', 'support_body', 'regions'}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError('invalid_native_task_config')
    if (type(config['schema_version']) is not int or config['schema_version'] != 1
            or config['input_mode'] != 'privileged' or config['robot'] != 'stretch'):
        raise ValueError('unsupported_native_task_config')
    if not isinstance(config['reset_id'], str) or not config['reset_id'].strip():
        raise ValueError('invalid_task_reset_id')
    objects = config['objects']
    if not isinstance(objects, dict) or not objects or any(
        not isinstance(role, str) or not role or role != role.strip().casefold()
        or not isinstance(body, str) or not body for role, body in objects.items()
    ) or len(set(objects.values())) != len(objects):
        raise ValueError('invalid_task_object_roles')
    if not isinstance(config['support_body'], str) or not config['support_body']:
        raise ValueError('invalid_task_support')
    if not isinstance(config['regions'], list):
        raise ValueError('invalid_task_regions')
    regions = []
    for row in config['regions']:
        if not isinstance(row, dict) or set(row) != {'name', 'bounds'}:
            raise ValueError('invalid_task_region')
        regions.append(TransferRegion(row['name'], tuple(tuple(v) for v in row['bounds'])))
    RegionTransferProvider(tuple(regions), observe=lambda: None, transfer=lambda *_: False, input_mode='privileged')
    return config


class NativeTableTransferProvider(RegionTransferProvider):
    def __init__(self, robot, config, *, adapter=None, planning_ik_budget=2048, diagnostic_state=False):
        validate_native_task_config(config)
        if type(planning_ik_budget) is not int or planning_ik_budget <= 0:
            raise ValueError('invalid_planning_ik_budget')
        self.adapter = adapter if adapter is not None else NativeTableAdapter(robot)
        if self.adapter.input_mode != 'privileged':
            raise ValueError('native_transfer_requires_privileged_inputs')
        self.roles, self.support, self.reset_id = dict(config['objects']), config['support_body'], config['reset_id']
        self._revision, self._reference = 0, None
        self.diagnostic_state = bool(diagnostic_state)
        self._profile = fingerprint({'robot': 'stretch', 'interface': 'native_arm_to', 'config': config})
        self._ik_remaining, self.motion_evidence, self._motion_cache = planning_ik_budget, [], {}
        model = self.adapter.model
        self._robot_ids = body_subtree(model, 'base_link')
        self._support_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.support)
        if self._support_id < 0 or int(model.body_jntnum[self._support_id]) != 0:
            raise ValueError('unsupported_task_support')
        self._object_ids = {}
        for role, name in self.roles.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0 or bid in self._robot_ids:
                raise ValueError('invalid_task_object_body')
            joint = int(model.body_jntadr[bid])
            if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError('task_object_requires_free_root')
            self._object_ids[role] = bid
        regions = tuple(TransferRegion(row['name'], tuple(tuple(v) for v in row['bounds'])) for row in config['regions'])
        super().__init__(regions, observe=self._observe_native, transfer=self._transfer_native, input_mode='privileged')

    def _observe_native(self):
        with self.adapter.lock:
            self.adapter.synchronize(self.adapter.data)
            model = self.adapter.model
            scratch = mujoco.MjData(model)
            mujoco.mj_copyData(scratch, model, self.adapter.data)
            mujoco.mj_collision(model, scratch)
            placements = placements_from_mujoco_model(model, scratch)
            surfaces = np.asarray(placements[self.support].get('support_surfaces', []))
            if surfaces.ndim != 3 or surfaces.shape[1:] != (2, 3):
                raise ValueError('task_support_geometry_unavailable')
            for region in self.regions.values():
                box = np.asarray(region.bounds)
                if not any(np.all(box[0] >= p[0, :2]) and np.all(box[1] <= p[1, :2]) for p in surfaces):
                    raise ValueError('task_region_outside_support')
            locations, footprints = [], []
            for role, bid in self._object_ids.items():
                joint = int(model.body_jntadr[bid])
                dof = int(model.jnt_dofadr[joint])
                contacts = {int(other) for c in scratch.contact if bid in model.geom_bodyid[c.geom]
                            for other in model.geom_bodyid[c.geom] if other != bid}
                if (self._support_id not in contacts or contacts & self._robot_ids
                        or np.linalg.norm(scratch.qvel[dof:dof+3]) > .01
                        or np.linalg.norm(scratch.qvel[dof+3:dof+6]) > .1):
                    raise ValueError('task_object_not_settled_on_support')
                bounds = np.asarray(placements[self.roles[role]]['bounds'])[:, :2]
                footprints.append((role, tuple(bounds[1] - bounds[0])))
                containing, intersecting = [], []
                for region in self.regions.values():
                    box = np.asarray(region.bounds)
                    if np.all(bounds[0] >= box[0]) and np.all(bounds[1] <= box[1]):
                        containing.append(region.name)
                    if np.all(bounds[0] < box[1]) and np.all(box[0] < bounds[1]):
                        intersecting.append(region.name)
                if len(containing) > 1 or (intersecting and containing != intersecting):
                    raise ValueError('ambiguous_region_occupancy')
                locations.append((role, containing[0] if containing else None))
            support_z = float(np.max(surfaces[:, 1, 2]))
            height = max(float(np.diff(np.asarray(placements[n]['bounds'])[:, 2])[0]) for n in self.roles.values())
            blocked = set()
            for name, entry in placements.items():
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if name in set(self.roles.values()) | {self.support} or bid in self._robot_ids:
                    continue
                bounds = np.asarray(entry['bounds'])
                if bounds[1, 2] <= support_z+.001 or bounds[0, 2] >= support_z+height:
                    continue
                for region in self.regions.values():
                    box = np.asarray(region.bounds)
                    if np.all(bounds[0, :2] < box[1]) and np.all(box[0] < bounds[1, :2]):
                        blocked.add(region.name)
            if self._reference is None or np.max(np.abs(scratch.qpos-self._reference)) > .005:
                self._revision += 1
                self._reference = scratch.qpos.copy()
                self._motion_cache.clear()
            binding = SceneBinding(self.adapter.signature, self._profile, self.adapter.boot, self.reset_id, self._revision)
            return TransferObservation(binding, tuple(locations), tuple(footprints), frozenset(blocked))

    def _refine(self, role, region, *, retain=False):
        with self.adapter.lock:
            if self._ik_remaining <= 0:
                raise BudgetExhausted("task_motion_budget_exhausted")
            ref = 'task:transfer:' + fingerprint((role, region.name))
            self.adapter.tasks[ref] = (self.roles[role], self.support, role, region.name)
            previous_limit = self.adapter.max_planning_ik_calls
            self.adapter.max_planning_ik_calls = min(previous_limit, self._ik_remaining)
            event_start = len(getattr(self.adapter, "events", []))
            self.adapter.planning_ik_calls = 0
            try:
                planned = self.adapter.plan(task_ref=ref, placement_region=region.bounds)
            finally:
                # Failed/timed-out probes still consume work; never reset the
                # episode allowance merely because no plan was returned.
                count = int(self.adapter.planning_ik_calls)
                self._ik_remaining = max(0, self._ik_remaining - count)
                self.adapter.max_planning_ik_calls = previous_limit
            data = planned.get('data', {})
            evidence = {'object_name': role, 'region': region.name, 'code': planned['code'],
                        'ik_calls': count, 'remaining_planning_ik_calls': self._ik_remaining,
                        'stage': 'execution_refinement' if retain else 'task_search_probe',
                        'rejections': data.get('planning_rejections', {})}
            self.motion_evidence.append(evidence)
            payload = dict(evidence)
            if getattr(self, 'diagnostic_state', False) and planned['status'] != 'ok':
                payload['debug_state'] = {
                    'source': 'simulator_ground_truth', 'model_signature': self.adapter.signature,
                    'qpos': self.adapter.data.qpos.tolist(), 'qvel': self.adapter.data.qvel.tolist(),
                    'sim_time': float(self.adapter.data.time),
                    'ik_rejections': [{'stage': e.get('stage'), 'detail': e.get('detail')}
                                      for e in self.adapter.events[event_start:]
                                      if e.get('phase') == 'planning_rejection'],
                }
            logger = logging.getLogger(__name__)
            (logger.info if planned['status'] == 'ok' else logger.warning)(
                'Native task refinement: %s', json.dumps(payload, allow_nan=False))
            if planned['status'] != 'ok':
                return None
            plan_ref = data['plan_ref']
            if not retain:
                self.adapter.plans.pop(plan_ref, None)
            return plan_ref

    def feasible(self, state, operator):
        if not super().feasible(state, operator):
            return False
        # Probe only measured states. Hypothetical nodes remain task skeletons;
        # do not fake a full-episode motion certificate from current geometry.
        if self._last is not None and state == self._last:
            key = state.binding, operator.ref
            if key not in self._motion_cache:
                self._motion_cache[key] = self._refine(*self._selection(operator)) is not None
            return self._motion_cache[key]
        return True

    def _transfer_native(self, role, region):
        with self.adapter.lock:
            plan_ref = self._refine(role, region, retain=True)
            if plan_ref is None:
                code = 'task_motion_budget_exhausted' if self._ik_remaining <= 0 else 'task_motion_planning_failed'
                return TaskExecutionResult(False, False, code)
            result = self.adapter.execute(plan_ref)
            started = result.get('data', {}).get('execution_started')
            return TaskExecutionResult(result['status'] == 'ok', started if type(started) is bool else True)
