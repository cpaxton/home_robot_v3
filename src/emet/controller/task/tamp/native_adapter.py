# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Stationary Stretch native-debug adapter for the public JSON TAMP tools.

Only the packaged table model is admitted. Motion uses normal client controllers;
GT snapshots are explicit debug inputs. Observed discovery/execution is separate.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature, read_snapshot
from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult, PhysicalPickPlaceExecutor
from emet.controller.task.tamp.api import failure_code, response
from emet.motion.mujoco_collision import body_subtree
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.sim_object_placements import placements_from_mujoco_model


@dataclass
class NativePlan:
    object_body: str
    support_body: str
    initial: np.ndarray
    goal: np.ndarray
    rotation: np.ndarray
    executor: PhysicalPickPlaceExecutor
    boot: str
    placement_region: np.ndarray | None = None
    object_name: str = ""
    receptacle_name: str = ""


class NativeTableAdapter:
    max_planning_ik_calls = 256
    input_mode = 'privileged'

    def __init__(self, robot, *, model=None):
        if model is None:
            from emet.simulation.mujoco_server_stretch import default_scene_xml_path
            model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
        self.robot, self.model = robot, model
        self.signature = model_signature(model)
        self.data = mujoco.MjData(model)
        self.boot, _ = read_snapshot(robot, self.signature)
        self.namespace = uuid.uuid4().hex
        self.tasks, self.plans = {}, {}
        self.lock = threading.RLock()
        self.events = []
        self.planning_rejections = {}
        self.planning_ik_calls = 0
        self.synchronize(self.data)

    def synchronize(self, data):
        boot, snapshot = read_snapshot(self.robot, self.signature)
        if boot != self.boot:
            raise ObservationUnavailable('scene_changed_replan')
        q, v = np.asarray(snapshot['qpos'], dtype=float), np.asarray(snapshot['qvel'], dtype=float)
        t = float(snapshot['sim_time'])
        if q.shape != (self.model.nq,) or v.shape != (self.model.nv,) or not np.isfinite(t):
            raise ObservationUnavailable('invalid_measured_state')
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ObservationUnavailable('invalid_measured_state')
        data.qpos[:], data.qvel[:], data.time = q, v, t
        mujoco.mj_kinematics(self.model, data)

    def _data(self, **extra):
        return {'mode': 'physical', 'inputs': 'privileged', 'geometry_source': 'simulator_ground_truth',
                'assistance': [], 'base_motion': 'stationary',
                'collision_scope': 'sampled_arm_and_payload', **extra}

    def _record_rejection(self, phase, reason, executor):
        code = failure_code(reason)
        key = f'{phase}:{code}'
        self.planning_rejections[key] = self.planning_rejections.get(key, 0) + 1
        self.events.append({'phase': 'planning_rejection', 'stage': phase, 'code': code,
                            'detail': reason, 'contacts': list(executor.collision.last_contacts) if code == 'collision' else []})

    def _failure(self, tool, reason, *, completed=(), phase=None, started=False, identity=None):
        logging.getLogger(__name__).warning("Native TAMP %s failed at %s: %s", tool, phase, reason)
        return response(tool, code=failure_code(reason), status='partial' if completed or started else 'error',
                        data=self._data(**(identity or {}), completed_ops=list(completed), failed_stage=phase, execution_started=started,
                            **({'planning_rejections': dict(self.planning_rejections),
                                'planning_ik_calls': self.planning_ik_calls, 'max_ik_calls': self.max_planning_ik_calls}
                               if tool == 'plan_pick_place' else {})),
                        recovery='inspect' if completed or started else None)

    def discover(self, object_filter='', robot=''):
        with self.lock:
            try:
                self.synchronize(self.data)
                if robot and robot != 'stretch':
                    return self._failure('scene_tasks', 'native_robot_unsupported')
                placements = placements_from_mujoco_model(self.model, self.data)
                labels = (self.robot.get_emet_session() or {}).get('sim_object_placements', {})
                robot_ids = body_subtree(self.model, 'base_link')
                objects, supports = [], []
                for name, entry in placements.items():
                    body = self.model.body(name)
                    joint = int(body.jntadr[0])
                    if body.id in robot_ids:
                        continue
                    if joint >= 0 and self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                        objects.append(name)
                    elif len(entry.get('support_surfaces', [])) and body.jntnum[0] == 0:
                        supports.append(name)
                rows = []
                for obj in objects:
                    label = str(labels.get(obj, {}).get('cat', placements[obj]['cat']))
                    if object_filter.lower() not in label.lower():
                        continue
                    for support in supports:
                        key = (obj, support)
                        ref = next((ref for ref, value in self.tasks.items() if value[:2] == key), None)
                        if ref is None:
                            ref = f'task:{self.namespace}:{len(self.tasks)+1}'
                        target = str(labels.get(support, {}).get('cat', placements[support]['cat']))
                        self.tasks[ref] = (obj, support, label, target)
                        rows.append({'task_ref': ref, 'object_name': label, 'receptacle_name': target,
                                     'access': {'state': 'not_articulated', 'required_action': None,
                                                'available_actions': []}})
                return response('scene_tasks', data=self._data(tasks=rows))
            except (ObservationUnavailable, ValueError) as exc:
                return self._failure('scene_tasks', str(exc))

    def _executor(self, obj):
        from emet.eval.physical_tamp import make_scene_checker
        chain = STRETCH_ARM_CHAIN
        joints = chain.joint_names[:-1]
        checker = make_scene_checker(self.model, {'object_body': obj, 'gripper_bodies': chain.gripper_bodies})
        def command(q):
            # Retain measured head pose; arm_to's default would move unplanned joints.
            self.synchronize(self.data)
            values = dict(zip(joints, q, strict=True))
            cmd = [0, values['joint_lift'], sum(values[f'joint_arm_l{i}'] for i in range(4)),
                   values['joint_wrist_roll'], values['joint_wrist_pitch'], values['joint_wrist_yaw']]
            head = [float(self.data.qpos[self.model.joint(name).qposadr[0]])
                    for name in ('joint_head_pan', 'joint_head_tilt')]
            self.robot.switch_to_manipulation_mode()
            # The physical executor owns measured convergence and retention checks.
            # The client's generic "stopped moving" heuristic rejects loaded grasps
            # even inside the executor's unchanged 0.03 joint tolerance.
            return self.robot.arm_to(cmd, head=head, blocking=False)
        return PhysicalPickPlaceExecutor(self.robot, model=self.model, data=self.data,
            ee_body=chain.ee_body, joint_names=joints, collision=checker,
            synchronize=self.synchronize, command_joints=command,
            coupled_groups=(tuple(f'joint_arm_l{i}' for i in range(4)),),
            gripper_open_configuration=chain.gripper_open_configuration,
            base_body='base_link', input_mode='privileged', endpoint_settle_s=.3,
            use_sim_time=True, position_tolerance_m=.003,
            event=lambda **row: self.events.append({"sim_time": float(self.data.time), **row}))

    def _pinch_center_offset(self):
        """Nominal opposing contact midpoint in the EE frame, from robot geometry.

        The named grasp frame is ahead of the pads. Centering an object on that
        frame (with an extra height offset) pinches its upper edge and torques
        the compliant fingertips during lift. This calibration uses robot
        geometry only; it is not a live contact sensor or an object pose write.
        """
        scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(scratch, self.model, self.data)
        for name in ('joint_gripper_slide', 'joint_gripper_finger_left_open',
                     'joint_gripper_finger_right_open', 'rubber_left_x',
                     'rubber_left_y', 'rubber_right_x', 'rubber_right_y'):
            scratch.qpos[self.model.joint(name).qposadr[0]] = 0.
        mujoco.mj_kinematics(self.model, scratch)
        pads_by_side = []
        centers = []
        for name in ('rubber_tip_left', 'rubber_tip_right'):
            body = self.model.body(name)
            pads = [g for g in range(int(body.geomadr[0]), int(body.geomadr[0] + body.geomnum[0]))
                    if self.model.geom_contype[g] or self.model.geom_conaffinity[g]]
            if len(pads) != 1:
                raise ValueError('native_model_mismatch')
            pads_by_side.append(pads[0])
            centers.append(scratch.geom_xpos[pads[0]])
        # The pads are angled boxes: first contact is on their inward support
        # edge, not at the box center. Center those opposing contact surfaces.
        axis = centers[1] - centers[0]
        axis /= np.linalg.norm(axis)
        contacts = []
        for g, direction in zip(pads_by_side, (axis, -axis), strict=True):
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError('native_model_mismatch')
            rotation = scratch.geom_xmat[g].reshape(3, 3)
            local = rotation.T @ direction
            vertex = np.where(np.abs(local) > .01, np.sign(local), 0.) * self.model.geom_size[g]
            contacts.append(scratch.geom_xpos[g] + rotation @ vertex)
        ee = scratch.body(STRETCH_ARM_CHAIN.ee_body)
        return ee.xmat.reshape(3, 3).T @ (np.mean(contacts, axis=0) - ee.xpos)

    def _place_targets(self, executor, obj, goal, rotation):
        # Compute the measured/predicted payload offset, not a category-specific center assumption.
        placements = placements_from_mujoco_model(self.model, self.data)
        offset = np.asarray(placements[obj]['pos']) - self.data.body(executor.ee_body).xpos
        release = goal - offset
        return [(release + [0, 0, .12], rotation), (release, rotation),
                (release + [0, .10, .12], rotation)]

    def _plan_candidate_pose(self, executor, target, rotation):
        # One shared budget spans grasp alternatives and downstream placement.
        if self.planning_ik_calls >= self.max_planning_ik_calls:
            raise ValueError('no_plan_within_budget')
        self.planning_ik_calls += 1
        return executor.plan_pose(target, rotation)

    def _grasp_candidates(self, executor, before, point, frames, pinch_offset, obj):
        for frame, grasp_height in frames:
            for yaw in (0., -.35, .35, -.7, .7, -1.57, 1.57):
                self.data.qpos[:] = before
                executor.collision.set_payload(self.model, self.data, None)
                mujoco.mj_kinematics(self.model, self.data)
                c, sn = np.cos(yaw), np.sin(yaw)
                rotation = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]) @ frame
                grasp = point - rotation @ pinch_offset + [0, 0, grasp_height]
                executor.grasp_targets = [(grasp + delta, rotation) for delta in
                                          ([0, .08, .04], [0, 0, 0], [0, 0, .17])]
                _, error = executor.plan_gripper_open()
                if error:
                    self._record_rejection('gripper_open', error, executor)
                    return
                paths = []
                for phase, (target, rot) in zip(('pregrasp', 'grasp', 'lift'), executor.grasp_targets, strict=True):
                    if phase == 'lift':
                        executor.collision.set_payload(self.model, self.data, obj, executor.ee_body)
                    path, error = self._plan_candidate_pose(executor, target, rot)
                    if error:
                        self._record_rejection(phase, error, executor)
                        break
                    paths.append(path)
                if len(paths) == 3:
                    yield rotation, paths

    def plan(self, task_ref='', object_name='', receptacle_name='', placement_region=None):
        with self.lock:
            self.planning_rejections = {}
            self.planning_ik_calls = 0
            try:
                region = None
                if placement_region is not None:
                    try:
                        region = np.asarray(placement_region, dtype=float)
                    except (TypeError, ValueError):
                        return self._failure('plan_pick_place', 'invalid_placement_region')
                    if (region.shape != (2, 2) or not np.isfinite(region).all()
                            or not np.all(region[0] < region[1])):
                        return self._failure('plan_pick_place', 'invalid_placement_region')
                    region = region.copy()
                self.synchronize(self.data)
                if task_ref:
                    selected = self.tasks.get(task_ref)
                else:
                    discovery = self.discover(object_name)
                    if discovery['status'] != 'ok':
                        return self._failure('plan_pick_place', discovery['code'])
                    matches = [self.tasks[row['task_ref']] for row in discovery['data']['tasks']
                               if row['receptacle_name'].lower() == receptacle_name.lower()]
                    if len(matches) != 1:
                        return self._failure('plan_pick_place', 'ambiguous_object' if matches else 'task_not_found')
                    selected = matches[0]
                if selected is None:
                    return self._failure('plan_pick_place', 'unknown_task_ref')
                # A handle identifies one task; redundant names must confirm that
                # identity, never silently redirect or contradict the requested move.
                identity = dict(zip(('object_name', 'receptacle_name'), selected[2:], strict=True))
                if task_ref and any(given and given.strip().casefold() != expected.strip().casefold()
                                    for given, expected in zip((object_name, receptacle_name), selected[2:], strict=True)):
                    return self._failure('plan_pick_place', 'task_selector_mismatch', identity=identity)
                obj, support = selected[:2]
                initial = self.data.qpos.copy()
                executor = self._executor(obj)
                placements = placements_from_mujoco_model(self.model, self.data)
                point = np.asarray(placements[obj]['pos'])
                rotation = self.data.body(executor.ee_body).xmat.reshape(3, 3).copy()
                surfaces = np.asarray(placements[support]['support_surfaces']).copy()
                if region is not None:
                    surfaces[:, 0, :2] = np.maximum(surfaces[:, 0, :2], region[0])
                    surfaces[:, 1, :2] = np.minimum(surfaces[:, 1, :2], region[1])
                    surfaces = surfaces[np.all(surfaces[:, 0, :2] < surfaces[:, 1, :2], axis=1)]
                    if len(surfaces) == 0:
                        return self._failure('plan_pick_place', 'placement_region_unavailable')
                before = self.data.qpos.copy()
                try:
                    # The navigation wrist frame is not necessarily a valid grasp frame.
                    # Search bounded wrist yaw alternatives around measured and home frames.
                    home = mujoco.MjData(self.model)
                    mujoco.mj_copyData(home, self.model, self.data)
                    for name in ('wrist_yaw', 'wrist_pitch', 'wrist_roll'):
                        home.qpos[self.model.joint('joint_'+name).qposadr[0]] = self.model.key_ctrl[0, self.model.actuator(name).id]
                    mujoco.mj_kinematics(self.model, home)
                    pinch_offset = self._pinch_center_offset()
                    home_rotation = home.body(executor.ee_body).xmat.reshape(3, 3).copy()
                    # A slight pitched approach avoids rolling rounded payloads
                    # out of the angled pads. Search the same robot-centric
                    # frames for every object, then certify and measure the grasp.
                    frames = []
                    height = float(np.diff(np.asarray(placements[obj]['bounds'])[:, 2])[0])
                    for pitch in (np.pi / 12, np.pi / 4, np.pi / 3):
                        c, sn = np.cos(pitch), np.sin(pitch)
                        frame = home_rotation @ np.array([[c, sn, 0], [-sn, c, 0], [0, 0, 1]])
                        for fraction in ((0.,) if pitch == np.pi / 12 else (0., .25, .4)):
                            frames.append((frame, fraction * height))
                    frames.extend([(home_rotation, 0.), (rotation, 0.)])
                    for rotation, paths in self._grasp_candidates(executor, before, point, frames, pinch_offset, obj):
                        executor.grasp_paths = paths
                        lifted = self.data.qpos.copy()
                        # Reuse the placement stack's obstacle-subtracted support search.
                        from emet.motion.placement_geometry import HeldObject, PlacementScene
                        from emet.motion.placement_surfaces import free_surface_centers
                        lifted_placements = placements_from_mujoco_model(self.model, self.data)
                        robot_ids = body_subtree(self.model, 'base_link')
                        scene_entries = {name: entry for name, entry in lifted_placements.items()
                                         if self.model.body(name).id not in robot_ids}
                        scene = PlacementScene.from_placements(scene_entries, held_object=obj)
                        ee = self.data.body(executor.ee_body)
                        payload = HeldObject.from_world_bounds(lifted_placements[obj]['bounds'],
                            ee_position=ee.xpos, ee_rotation=ee.xmat.reshape(3, 3))
                        # Stationary reach is local: include clipped subpatches near the
                        # grasp, rather than only centers across the whole tabletop.
                        patches = list(surfaces)
                        for radius in (.08, .16, .32):
                            for surface in surfaces:
                                patch = surface.copy()
                                patch[0, :2] = np.maximum(patch[0, :2], point[:2]-radius)
                                patch[1, :2] = np.minimum(patch[1, :2], point[:2]+radius)
                                if np.all(patch[0, :2] < patch[1, :2]):
                                    patches.append(patch)
                        centers = free_surface_centers(patches, scene=scene, payload=payload,
                            ee_rotation=rotation, max_centers=64).centers
                        if not centers:
                            self._record_rejection('surface', 'no_collision_free_placement', executor)
                        centers.sort(key=lambda center: float(np.linalg.norm(center[:2]-point[:2])))
                        for goal in centers:
                            if region is None and np.linalg.norm(goal[:2]-point[:2]) < .04:
                                continue
                            self.data.qpos[:] = lifted
                            mujoco.mj_kinematics(self.model, self.data)
                            executor.place_targets = self._place_targets(executor, obj, goal, rotation)
                            # Pure planning: never call the measured-state replan here, which would reset the hypothetical lift.
                            candidates = []
                            for i, (target, rot) in enumerate(executor.place_targets):
                                if i == 2:
                                    executor.collision.set_payload(self.model, self.data, None)
                                    _, error = executor.plan_gripper_open()
                                    if error:
                                        self._record_rejection('release_open', error, executor)
                                        break
                                path, error = self._plan_candidate_pose(executor, target, rot)
                                if error:
                                    self._record_rejection(('preplace', 'place', 'retreat')[i], error, executor)
                                    break
                                candidates.append(path)
                            if len(candidates) == 3:
                                executor.place_paths = candidates
                                ref = f'plan:{self.namespace}:{uuid.uuid4().hex}'
                                self.plans[ref] = NativePlan(obj, support, initial, goal, rotation, executor, self.boot, region, **identity)
                                return response('plan_pick_place', data=self._data(**identity, plan_ref=ref,
                                    planned_ops=['grasp', 'place'], completed_ops=[], failed_stage=None, measurements=[],
                                    planning_rejections=dict(self.planning_rejections),
                                    planning_ik_calls=self.planning_ik_calls, max_ik_calls=self.max_planning_ik_calls,
                                    placement_region=None if region is None else region.tolist(),
                                    placement_goal={'frame': 'world', 'object_center': goal.tolist()}))
                            executor.collision.set_payload(self.model, self.data, None)
                            self.data.qpos[:] = lifted
                            mujoco.mj_kinematics(self.model, self.data)
                            executor.collision.set_payload(self.model, self.data, obj, executor.ee_body)
                    return self._failure('plan_pick_place', 'no_plan_within_budget')
                finally:
                    self.data.qpos[:] = before
                    executor.collision.set_payload(self.model, self.data, None)
                    mujoco.mj_kinematics(self.model, self.data)
            except (ObservationUnavailable, ValueError, RuntimeError) as exc:
                return self._failure('plan_pick_place', str(exc))

    def _verify(self, plan, *, released, seconds):
        self.synchronize(self.data)
        start = None
        active = plan.executor._settling_window(12.)
        reference = None
        while active():
            self.synchronize(self.data)
            scratch = mujoco.MjData(self.model)
            mujoco.mj_copyData(scratch, self.model, self.data)
            mujoco.mj_collision(self.model, scratch)
            oid = self.model.body(plan.object_body).id
            contacts = {self.model.body(self.model.geom_bodyid[c.geom2 if self.model.geom_bodyid[c.geom1] == oid else c.geom1]).name
                        for c in scratch.contact if oid in self.model.geom_bodyid[c.geom]}
            if released:
                entry = placements_from_mujoco_model(self.model, scratch)[plan.object_body]
                error = np.linalg.norm(np.asarray(entry['pos'])[:2]-plan.goal[:2])
                joint = int(self.model.body(plan.object_body).jntadr[0])
                dof = self.model.jnt_dofadr[joint]
                speed = np.linalg.norm(self.data.qvel[dof:dof+3])
                spin = np.linalg.norm(self.data.qvel[dof+3:dof+6])
                height_error = abs(float(entry['pos'][2]) - (plan.goal[2]-.02))
                valid = (plan.support_body in contacts and not {'rubber_tip_left','rubber_tip_right'} & contacts
                         and error <= .04 and height_error <= .025 and speed <= .01 and spin <= .1)
                if plan.placement_region is not None:
                    bounds = np.asarray(entry['bounds'])[:, :2]
                    valid = valid and bool(np.all(bounds[0] >= plan.placement_region[0] - .005)
                                           and np.all(bounds[1] <= plan.placement_region[1] + .005))
            else:
                ee = self.data.body(plan.executor.ee_body)
                relative = ee.xmat.reshape(3,3).T @ (self.data.body(plan.object_body).xpos-ee.xpos)
                if reference is None:
                    reference = relative.copy()
                valid = {'rubber_tip_left','rubber_tip_right'} <= contacts and plan.support_body not in contacts and np.linalg.norm(relative-reference) <= .025
            if not valid:
                start = None
            elif start is None:
                start = float(self.data.time)
            elif self.data.time-start >= seconds:
                return True
            time.sleep(.05)
        return False

    def execute(self, plan_ref):
        with self.lock:
            plan = self.plans.pop(plan_ref, None)
            if plan is None:
                return self._failure('execute_pick_place_plan', 'unknown_plan')
            identity = {'object_name': plan.object_name, 'receptacle_name': plan.receptacle_name}

            def fail(reason, **kwargs):
                return self._failure('execute_pick_place_plan', reason, identity=identity, **kwargs)

            completed = []
            started = False
            phase = None
            try:
                self.synchronize(self.data)
                if plan.boot != self.boot or np.max(np.abs(self.data.qpos-plan.initial)) > .01:
                    return fail('scene_changed_replan')
                ex = plan.executor
                started = True
                phase = 'grasp'
                result = ex.grasp_only('', object_gt_body=plan.object_body)
                if not result.success:
                    return fail(result.message, started=True, phase=result.phase)
                if not self._verify(plan, released=False, seconds=1):
                    return fail('grasp_verification_failed', started=True, phase='lift')
                completed.append('grasp')
                self.synchronize(self.data)
                ex.place_targets = self._place_targets(ex, plan.object_body, plan.goal, plan.rotation)
                ex.transport = lambda: PhysicalMotionResult(True, 'stationary', 'transport')
                phase = 'place'
                result = ex.place_only('', object_gt_body=plan.object_body, receptacle_gt_body=plan.support_body)
                if not result.success:
                    return fail(result.message, completed=completed, phase=result.phase)
                if not self._verify(plan, released=True, seconds=2):
                    return fail('placement_verify_failed', completed=completed, phase='release')
                completed.append('place')
                return response('execute_pick_place_plan', data=self._data(**identity, plan_ref=plan_ref,
                    planned_ops=['grasp','place'], completed_ops=completed, failed_stage=None,
                    measurements=[{'source':'simulator_ground_truth','bilateral_hold_s':1.,'stable_support_s':2.,
                                   'placement_region_verified': plan.placement_region is not None}]))
            except TimeoutError:
                # The command may already have reached the controller. Preserve
                # partial progress and prohibit reuse of this consumed handle.
                return fail('command_outcome_unknown',
                                     completed=completed, started=started, phase=phase)
            except (ObservationUnavailable, ValueError, RuntimeError) as exc:
                return fail(str(exc), completed=completed,
                                     started=started, phase=phase)


def adapter_for(context, robot):
    """Only called after the observed/privileged input boundary has been checked."""
    state = getattr(robot, '_state', None)
    if not isinstance(state, dict) or not isinstance(state.get('native_tamp_snapshot'), dict):
        return None
    cached = context.get('_native_tamp_adapter')
    boot = (state.get('command_protocol') or {}).get('server_boot_id')
    if cached is None or cached.robot is not robot or cached.boot != boot:
        cached = NativeTableAdapter(robot)
        context['_native_tamp_adapter'] = cached
    return cached
