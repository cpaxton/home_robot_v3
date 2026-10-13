#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""CPU contact regression through public TAMP tools; privileged, in-process control.

This complements live ZMQ runs; it does not test IPC, cameras, or exploration.
Only fixture initialization writes physics qpos. Execution sets actuator targets.
Launch through ``emet jobs run --no-gpu-exclusive --cpu-safe``.
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco

from emet.agent.tools import get_tools
from emet.controller.manipulation import physical_pick_place
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.task.tamp import native_adapter
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators


class ContactRobot:
    """Normal position-command facade over an independently stepped physics state."""

    def __init__(self, model, object_body, *, cluttered=False):
        self.model, self.data = model, mujoco.MjData(model)
        self.signature = model_signature(model)
        self._state_lock = threading.RLock()
        self.commands = []
        self.data.ctrl[:] = model.key_ctrl[0]
        self.data.ctrl[model.actuator('gripper').id] = .04
        snap_joint_qpos_to_ctrl_for_position_actuators(model, self.data)
        if not cluttered:
            a, b = [model.jnt_qposadr[model.body(n).jntadr[0]] for n in ('object1', 'object2')]
            if object_body == 'object2':
                self.data.qpos[b:b+2] = self.data.qpos[a:a+2].copy()
                self.data.qpos[a] = .3
            else:
                self.data.qpos[b] = .3
        self.step(2.)
        self.refresh()

    def step(self, duration):
        for _ in range(max(1, round(duration / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def refresh(self):
        self._state_received_monotonic = time.monotonic()
        self._state = {'command_protocol': {'server_boot_id': 'cpu-contact'}, 'native_tamp_snapshot': {
            'source': 'simulator_ground_truth', 'model_signature': self.signature,
            'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist(),
            'sim_time': float(self.data.time), 'captured_at': time.time()}}

    def get_emet_session(self):
        return {'is_simulation': True, 'sim_object_placements': {
            'object1': {'cat': 'blue cube'}, 'object2': {'cat': 'red cylinder'}, 'table': {'cat': 'table'}}}

    def switch_to_manipulation_mode(self):
        return True

    def arm_to(self, q, *, head, blocking):
        assert q[0] == 0 and not blocking
        self.commands.append({'sim_time': float(self.data.time), 'arm': list(q), 'head': list(head)})
        for name, value in zip(('lift', 'arm', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'), q[1:], strict=True):
            self.data.ctrl[self.model.actuator(name).id] = value
        for name, value in zip(('head_pan', 'head_tilt'), head, strict=True):
            self.data.ctrl[self.model.actuator(name).id] = value
        return True

    def open_gripper(self, *, blocking):
        self.commands.append({'sim_time': float(self.data.time), 'gripper': 'open'})
        self.data.ctrl[self.model.actuator('gripper').id] = .04
        self.step(1.)
        return abs(self.data.actuator('gripper').length[0] - .04) < .005

    def close_gripper(self, *, blocking):
        assert not blocking
        self.commands.append({'sim_time': float(self.data.time), 'gripper': 'close'})
        # Same normal -0.3 hardware preset and range mapping as the live client.
        self.data.ctrl[self.model.actuator('gripper').id] = (-.3 + .376) / .936 * .06 - .02
        return True

    def get_gripper_position(self):
        return float((self.data.actuator('gripper').length[0] + .02) / .06 * .936 - .376)


def run_case(model, object_body, *, cluttered=False):
    robot = ContactRobot(model, object_body, cluttered=cluttered)
    adapter = native_adapter.NativeTableAdapter(robot, model=model)
    original_sync = adapter.synchronize
    def synchronize(data):
        robot.refresh()
        original_sync(data)
    adapter.synchronize = synchronize
    tools = {t.name: t for t in get_tools({'robot': robot, 'manip_mode': 'physical',
             'tamp_inputs': 'privileged', '_native_tamp_adapter': adapter})}
    physics_clock = SimpleNamespace(monotonic=lambda: float(robot.data.time), sleep=robot.step)
    old_physical, old_native = physical_pick_place.time, native_adapter.time
    responses = []
    try:
        physical_pick_place.time = native_adapter.time = physics_clock
        def call(name, **kwargs):
            result = json.loads(tools[name].func(**kwargs))
            responses.append(result)
            return result
        label = 'blue' if object_body == 'object1' else 'red'
        discovery = call('scene_tasks', object_filter=label)
        rows = discovery['data'].get('tasks', [])
        row = next((r for r in rows if r['receptacle_name'] == 'table'), None)
        if row:
            plan = call('plan_pick_place', task_ref=row['task_ref'])
            if plan['status'] == 'ok':
                call('execute_pick_place_plan', plan_ref=plan['data']['plan_ref'])
    finally:
        physical_pick_place.time, native_adapter.time = old_physical, old_native
    robot.refresh()
    return {'passed': bool(responses and responses[-1]['tool'] == 'execute_pick_place_plan'
                           and responses[-1]['status'] == 'ok'), 'object_body': object_body,
            'responses': responses, 'events': adapter.events, 'commands': robot.commands,
            'snapshot': robot._state['native_tamp_snapshot']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object', choices=['object1', 'object2', 'both'], default='both')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--cluttered', action='store_true')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    objects = ('object1', 'object2') if args.object == 'both' else (args.object,)
    rows = []
    for obj in objects:
        for trial in range(args.trials):
            result = run_case(model, obj, cluttered=args.cluttered)
            path = args.output / f'{obj}-{trial}.json'
            path.write_text(json.dumps(result, indent=2, allow_nan=False))
            row = {'object': obj, 'trial': trial, 'passed': result['passed'], 'artifact': str(path)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {'scope': 'privileged CPU public tools; in-process actuator facade; no ZMQ or exploration',
               'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'cluttered': args.cluttered, 'cases': rows,
               'passed': sum(r['passed'] for r in rows), 'total': len(rows)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return 0 if summary['passed'] == summary['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
