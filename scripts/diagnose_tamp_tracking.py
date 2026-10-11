#!/usr/bin/env python3
"""Paired contact/no-contact MuJoCo diagnosis from a recorded TAMP attempt.

No renderer, server, or policy; this reconstructs poses, not exact live timing.
Contact-disabled trials are diagnostic ablations, never benchmark successes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from emet.motion.arm_manip_profile import ArmManipProfile
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker
from emet.robots import get_robot_spec
from emet.simulation.teleport_collision import teleport_endpoint_contacts


def attempts(path):
    base = None
    result = []
    for line in Path(path).read_text().splitlines():
        if 'TAMP execute: approach ' in line:
            base = ast.literal_eval(line.split('TAMP execute: approach ')[1])['xyt']
        if 'measured EE: ' in line:
            if base is None:
                raise ValueError('measured attempt without approach')
            result.append(dict(json.loads(line.split('measured EE: ')[1]), base_xyt=base))
    if not result:
        raise ValueError('no measured attempts')
    return result


def run(args):
    profile = ArmManipProfile.for_robot(args.robot)
    spec = get_robot_spec(args.robot)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output.with_suffix('.mjb')
    mujoco.mj_saveModel(model, str(model_path))
    baseline = mujoco.MjData(model)
    server = None
    if args.base_support == 'weld':
        from emet.simulation.robosuite_server import RobosuiteZmqServer
        model.equality('emet_stationary_base')  # refuse silent legacy fallback
        server = object.__new__(RobosuiteZmqServer)
        server._mjmodel, server._spec = model, spec
        server._nav_goal_world = None
        server._passive_base_support = False
    placements = json.loads(args.initial.read_text())['placements']
    for name, pose in placements.items():
        body = model.body(name)
        joint = int(body.jntadr[0])
        if joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            address = int(model.jnt_qposadr[joint])
            baseline.qpos[address:address + 7] = pose['pos'] + pose['quat']
    for name, value in zip(profile.actuator_names, profile.home_cmd, strict=True):
        aid = model.actuator(name).id
        joint = int(model.actuator_trnid[aid, 0])
        baseline.qpos[model.jnt_qposadr[joint]] = value
        baseline.ctrl[aid] = value
    base = model.joint(profile.base_freejoint_name)
    ba, bv = int(base.qposadr[0]), int(base.dofadr[0])
    qa = [int(model.joint(n).qposadr[0]) for n in profile.joint_names]
    va = [int(model.joint(n).dofadr[0]) for n in profile.joint_names]
    aids = [int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(n).id)[0]) for n in profile.joint_names]
    checker = MujocoSceneCollisionChecker(model, robot_body=spec.base_link_name)
    rows = []
    original_flags = int(model.opt.disableflags)
    for index, attempt in enumerate(attempts(args.log)):
        x, y, yaw = attempt['base_xyt']
        baseline.qpos[ba:ba + 7] = [x, y, args.base_z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        model.opt.disableflags = original_flags
        rejected = teleport_endpoint_contacts(model, baseline, spec, attempt['base_xyt'])
        goal = np.asarray(attempt['planned_q'])
        for contacts in (True, False):
            model.opt.disableflags = original_flags if contacts else original_flags | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            data = mujoco.MjData(model)
            mujoco.mj_copyData(data, model, baseline)
            initial_q = data.qpos[qa].copy()
            initial_base = data.qpos[ba:ba + 7].copy()
            if server is not None:
                server._mjdata = data
                server._stationary_base_freejoint_qpos = initial_base
                server._base_freejoint_addrs = lambda: (ba, bv)
            pairs = {}
            peak_speed = 0.0
            peak_constraint = 0.0
            max_force = np.zeros(len(aids))
            for step in range(int(args.duration / model.opt.timestep)):
                fraction = min(1.0, step * model.opt.timestep / args.ramp)
                data.ctrl[aids] = initial_q + fraction * (goal - initial_q)
                # Match this oracle control's stationary full-base hold.
                if server is None:
                    data.qpos[ba:ba + 7] = initial_base
                    data.qvel[bv:bv + 6] = 0
                else:
                    server._hold_stationary_base_freejoint_if_idle()
                mujoco.mj_step(model, data)
                peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel[va]))))
                peak_constraint = max(peak_constraint, float(np.max(np.abs(data.qfrc_constraint[va]))))
                max_force = np.maximum(max_force, np.abs(data.actuator_force[aids]))
                if step % 25 == 0:
                    for c in data.contact:
                        bodies = tuple(int(model.geom_bodyid[g]) for g in c.geom)
                        if not checker.robot_ids.intersection(bodies):
                            continue
                        names = tuple(model.body(b).name for b in bodies)
                        pairs[names] = min(pairs.get(names, 0.0), float(c.dist))
            measured_q = data.qpos[qa].copy()
            mujoco.mj_kinematics(model, data)
            actual_ee = data.body(profile.ee_body).xpos.copy()
            data.qpos[qa] = goal
            mujoco.mj_kinematics(model, data)
            planned_ee = data.body(profile.ee_body).xpos.copy()
            row = {
                'attempt': index, 'contacts_enabled': contacts,
                'base_xyt': attempt['base_xyt'], 'endpoint_rejection': rejected,
                'joint_error_rad': (measured_q - goal).tolist(),
                'ee_tracking_error_m': float(np.linalg.norm(actual_ee - planned_ee)),
                'target_error_m': float(np.linalg.norm(actual_ee - attempt['target_xyz'])),
                'target_tolerance_m': attempt['tolerance_m'],
                'actual_ee_xyz': actual_ee.tolist(), 'planned_ee_xyz': planned_ee.tolist(),
                'peak_joint_speed_rad_s': peak_speed, 'peak_constraint_torque': peak_constraint,
                'peak_actuator_force': max_force.tolist(),
                'contacts': [{'bodies': list(k), 'min_distance_m': v} for k, v in pairs.items()],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        'scope': 'reconstructed diagnostic; no benchmark success claims',
        'mujoco_version': mujoco.__version__, 'base_z': args.base_z,
        'base_support': args.base_support,
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'duration_s': args.duration, 'ramp_s': args.ramp,
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.scene, args.initial, args.log)},
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--robot', default='rby1')
    parser.add_argument('--base-support', choices=['pose_reset', 'weld'], default='pose_reset')
    parser.add_argument('--base-z', type=float, required=True)
    parser.add_argument('--duration', type=float, default=8)
    parser.add_argument('--ramp', type=float, default=2)
    args = parser.parse_args()
    if not 0 < args.ramp < args.duration:
        parser.error('require 0 < ramp < duration')
    run(args)
