#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Compare grasp-search policies through live native task tools, without an LLM.

Privileged stationary Stretch diagnostic. Native actuation, no cameras or GPU.
Launch via emet jobs. Original agent outcomes are never rescored by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tools import get_tools
from emet.controller.dynamem.constants import DYNAMEM_HEAD_SETTLE_S
from emet.controller.manipulation.native_scene import model_signature, read_snapshot
from emet.controller.task.tamp.native_policy import native_policy
from emet.controller.task.tamp.native_transfer import NativeTableTransferProvider, validate_native_task_config
from emet.controller.task.tamp.task_session import TaskSession
from emet.eval.contact_stability import GRASP_POLICIES, ContactTrace, TracedNativeAdapter
from emet.eval.native_table_live import live_table_robot
from emet.eval.native_table_scoring import goal_predicates, load_placement_regions
from emet.motion.release import ReleaseLimits
from emet.simulation.mujoco_server_stretch import default_scene_xml_path

REPO = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_fixture(output, initial_positions, shift, quaternions=None):
    """Reset-only scene construction; canonical robot/contact/controller unchanged."""
    source = Path(default_scene_xml_path).parent
    fixture = output / 'fixture'
    fixture.mkdir()
    for name in ('scene.xml', 'stretch.xml', 'docking_station.xml', 'scene_environment.xml'):
        shutil.copyfile(source / name, fixture / name)
    for name in ('stretch.xml', 'docking_station.xml'):
        tree = ET.parse(fixture / name)
        tree.find('compiler').set('assetdir', str((source / 'assets').resolve()))
        tree.write(fixture / name)
    tree = ET.parse(fixture / 'scene_environment.xml')
    for name, position in initial_positions.items():
        body = tree.find(f'.//body[@name="{name}"]')
        point = np.asarray(position, dtype=float).copy()
        if body is None or body.find('freejoint') is None or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('invalid_initial_object_position')
        point[0] += shift
        body.set('pos', ' '.join(map(str, point)))
        if quaternions and name in quaternions:
            quat = np.asarray(quaternions[name], dtype=float)
            if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1., atol=1e-6):
                raise ValueError('invalid_initial_object_quaternion')
            body.set('quat', ' '.join(map(str, quat)))
    tree.write(fixture / 'scene_environment.xml')
    return fixture / 'scene.xml'


def prepare(args, pilot, case, out):
    validate_native_task_config(case['task'])
    scene = prepare_fixture(out, case.get('initial_object_positions', pilot['initial_object_positions']), case['initial_x_shift_m'], case.get('initial_object_quaternions'))
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    fixture_model = mujoco.MjModel.from_xml_path(str(scene))
    if model_signature(model) != model_signature(fixture_model):
        raise ValueError('fixture_changed_robot_or_scene_model')
    write_json(out / 'sim.json', {'kind': 'default_mujoco', 'robot': 'stretch', 'headless': True,
               'scene_path': str(scene), 'no_cameras': True, 'port_offset': args.port_offset, 'seed': case['seed']})
    write_json(out / 'regions.json', case['final_regions_object2_then_object1'])
    regions = load_placement_regions(out / 'regions.json')
    write_json(out / 'task.json', case['task'])
    sources = out / 'sources'
    sources.mkdir()
    hashes = {}
    for name in ('scripts/eval_tamp_contact_stability.py', 'src/emet/eval/contact_stability.py',
                 'src/emet/motion/release.py', 'src/emet/controller/task/tamp/native_policy.py',
                 'src/emet/controller/task/tamp/native_transfer.py',
                 'src/emet/simulation/stretch_mujoco/mujoco_server.py',
                 'src/emet/controller/task/tamp/native_adapter.py',
                 'src/emet/controller/manipulation/physical_pick_place.py'):
        path = REPO / name
        shutil.copyfile(path, sources / path.name)
        hashes[name] = sha256(path)
    protocol = {'case_id': case['case_id'], 'instruction': case['instruction'], 'task': case['task'],
                'policy': args.policy, 'native_policy': args.native_policy,
                'scenario': case.get('scenario', 'task'), 'case': case,
                'grasp_yaw_offsets_rad': GRASP_POLICIES.get(args.policy),
                'payload_binding': args.payload_binding,
                'payload_reference_changes_at_verified_lift': args.payload_binding == 'verified_lift',
                'release_policy': args.release_policy,
                'release_limits': {'max_raise_m': args.max_release_raise, 'max_drop_m': args.max_release_drop,
                                   'clearance_m': args.release_clearance,
                                   'offsets_m': list(ReleaseLimits(args.max_release_raise, args.max_release_drop).offsets_m),
                                   'max_adjustments': 2, 'tracking_reserve': 'positive_command_fk_minus_measured_ee_z'} if args.release_policy == 'bounded' else None,
                'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)),
                'source_sha256': hashes, 'case_file_sha256': sha256(args.case_file),
                'fixture_sha256': {p.name: sha256(p) for p in scene.parent.glob('*.xml')},
                'model_signature': model_signature(model), 'mujoco_version': mujoco.__version__,
                'initial_x_shift_m': case['initial_x_shift_m'], 'startup_settle_sim_s': args.startup_settle,
                'execution': 'simulation_live_async', 'inputs': 'privileged', 'agent_score': False,
                'collision_and_retention_thresholds_changed': False, 'recording_changes_timing': True,
                'initial_state_certificate': False, 'hardware_trials': False}
    write_json(out / 'protocol.json', protocol)
    return model, dict(zip(('object2', 'object1'), regions, strict=True))



def execute_case(case, robot, call, out):
    scenario = case.get('scenario', 'task')
    checks = []
    def rejection(name, expected, **kwargs):
        audit = out / 'actuation.jsonl'
        before = audit.read_bytes() if audit.exists() else b''
        receipt = call(name, **kwargs)
        after = audit.read_bytes() if audit.exists() else b''
        checks.append({'tool': name, 'code': receipt['code'], 'status': receipt['status'],
                       'expected_codes': expected, 'no_commands': before == after,
                       'passed': receipt['status'] == 'error' and receipt['code'] in expected and before == after})
        return receipt
    execution = None
    if scenario == 'invalid_then_task':
        rejection('execute_task_plan', ['unknown_plan'], plan_ref='task_plan_invalid')
    discovery = call('scene_tasks')
    if discovery['status'] == 'ok':
        if scenario == 'blocked_placement':
            rejection('plan_pick_place', ['no_collision_free_placement', 'no_plan_within_budget'],
                      object_name='red cylinder', receptacle_name='table', placement_region=case['blocked_region'])
        else:
            plan = call('plan_task', instruction=case['instruction'], strategy='beam')
            if plan['status'] == 'ok':
                if scenario == 'stale_plan':
                    robot.head_to(.15, -.5, blocking=True)
                    robot.step(.5)
                    rejection('execute_task_plan', ['stale_task_plan'], plan_ref=plan['data']['plan_ref'])
                else:
                    execution = call('execute_task_plan', plan_ref=plan['data']['plan_ref'])
                    if scenario == 'reuse_plan' and execution['status'] == 'ok':
                        rejection('execute_task_plan', ['unknown_plan'], plan_ref=plan['data']['plan_ref'])
    return execution, checks

def run(args, pilot, case, out, result):
    model, goals = prepare(args, pilot, case, out)
    if args.prepare_only:
        result.update(run_status='prepared', passed=None)
        return
    os.environ['EMET_PHYSICAL_EXECUTION'] = '1'
    os.environ['EMET_PHYSICAL_AUDIT'] = str(out / 'actuation.jsonl')
    with live_table_robot(out / 'sim.json', out, model, initialize=False) as robot:
        robot.move_to_nav_posture()
        robot.look_front(blocking=True, timeout=3.)
        time.sleep(DYNAMEM_HEAD_SETTLE_S)
        robot.step(args.startup_settle)
        _, initial = read_snapshot(robot, model_signature(model))
        write_json(out / 'initial.json', initial)
        result['initial_predicates'] = goal_predicates(robot, goals)
        trace = ContactTrace(model, out / 'contact.jsonl')
        try:
            adapter = TracedNativeAdapter(robot, model=model, trace=trace, policy=args.policy if args.native_policy is None else None,
                                          native_policy=args.native_policy or 'legacy',
                                          payload_binding=args.payload_binding,
                                          release_limits=ReleaseLimits(args.max_release_raise, args.max_release_drop,
                                              clearance_m=args.release_clearance) if args.release_policy == 'bounded' else None)
            provider = NativeTableTransferProvider(robot, case['task'], adapter=adapter)
            session = TaskSession(provider)
            tools = {t.name: t for t in get_tools({'robot': robot, 'manip_mode': 'physical',
                     'tamp_inputs': 'privileged', '_native_tamp_adapter': adapter, 'tamp_task_session': session})}
            responses = []

            def call(name, **kwargs):
                _, rendered, _ = _dispatch_tool_calls(
                    [{'name': name, 'arguments': kwargs}], tools, SimpleNamespace())
                envelope = json.loads(rendered[0].removeprefix(f'[{name}] '))
                responses.append(envelope)
                with (out / 'responses.jsonl').open('a') as stream:
                    stream.write(json.dumps(envelope) + '\n')
                return envelope

            execution, checks = execute_case(case, robot, call, out)
            result['boundary_checks'] = checks
            result['expected_failure_verified'] = bool(checks) and all(c['passed'] for c in checks)
            stable = True
            for _ in range(40):
                robot.step(.05)
                stable &= all(p['satisfied'] for p in goal_predicates(robot, goals).values())
            result.update(run_status='complete', final_predicates=goal_predicates(robot, goals),
                          responses=responses, payload_residual_maxima=trace.extrema,
                          payload_binding=args.payload_binding,
                          release_policy=args.release_policy,
                          trace_records=trace.records, max_record_wall_s=trace.max_record_wall_s)
            # An independent final task score and an explicit tool completion
            # are both needed; a failed receipt cannot become a reference pass.
            result['passed'] = bool(stable and execution is not None
                                    and execution['status'] == 'ok'
                                    and (not checks or result['expected_failure_verified'])
                                    and not any(p['satisfied'] for p in result['initial_predicates'].values()))
            write_json(out / 'adapter_events.json', adapter.events)
            _, final = read_snapshot(robot, model_signature(model))
            write_json(out / 'final.json', final)
        finally:
            trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, default=REPO / 'configs/eval/tamp_stretch_realism_pilot.json')
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--policy', choices=tuple(GRASP_POLICIES), default='baseline')
    parser.add_argument('--native-policy', choices=('legacy', 'bounded_release'), help='Shared production adapter policy; disables diagnostic overrides')
    parser.add_argument('--payload-binding', choices=('preserve', 'verified_lift_keep', 'verified_lift'), default='preserve',
                        help='Diagnostic only: bind measured payload once after a verified stable lift, before full place replan')
    parser.add_argument('--port-offset', type=int, required=True)
    parser.add_argument('--release-policy', choices=('legacy', 'bounded'), default='legacy')
    parser.add_argument('--max-release-drop', type=float, default=.04, help='Declared maximum release gap in bounded native diagnostic (metres)')
    parser.add_argument('--max-release-raise', type=float, default=.04, help='Maximum arm raise from measured release pose (metres)')
    parser.add_argument('--release-clearance', type=float, default=.003, help='Vertical clearance reserve for opening (metres)')
    parser.add_argument('--startup-settle', type=float, default=6., help='Additional native simulated startup settling seconds')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Freeze inputs and validate model; start no server or controllers')
    args = parser.parse_args()
    if not np.isfinite(args.startup_settle) or args.startup_settle < 0:
        parser.error('--startup-settle must be finite and nonnegative')
    try:
        ReleaseLimits(args.max_release_raise, args.max_release_drop, clearance_m=args.release_clearance)
    except ValueError as error:
        parser.error(str(error))
    if args.native_policy is not None:
        if (args.policy, args.payload_binding, args.release_policy, args.max_release_raise,
                args.max_release_drop, args.release_clearance) != ('baseline', 'preserve', 'legacy', .04, .04, .003):
            parser.error('--native-policy cannot be combined with diagnostic overrides')
        selected = native_policy(args.native_policy)
        args.policy = 'fine_yaw' if args.native_policy == 'bounded_release' else 'baseline'
        args.payload_binding = selected.payload_binding
        args.release_policy = 'bounded' if selected.release_limits else 'legacy'
    pilot = json.loads(args.case_file.read_text())
    matches = [c for c in pilot['cases'] if c['case_id'] == args.case_id]
    if len(matches) != 1:
        parser.error('--case-id must select exactly one declared case')
    out = args.output.resolve()
    if out.exists():
        parser.error('--output must be new; preserve earlier attempts')
    out.mkdir(parents=True)
    result = {'case_id': args.case_id, 'policy': args.policy, 'agent_score': False, 'passed': False}
    started = time.monotonic()
    try:
        run(args, pilot, matches[0], out, result)
    except Exception as error:
        result.update(run_status='infrastructure_error', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    result['wall_time_s'] = time.monotonic() - started
    audit = out / 'actuation.jsonl'
    commands, malformed = [], 0
    for line in audit.read_text().splitlines() if audit.exists() else []:
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError('invalid_audit_row')
            commands.append(command)
        except ValueError:
            malformed += 1
    result.update(audit_commands=len(commands), malformed_audit_rows=malformed,
                  forbidden_commands=sum(bool(r.get('forbidden')) for r in commands))
    result['passed'] = result['passed'] and bool(commands) and not malformed and not result['forbidden_commands']
    write_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] or result['run_status'] == 'prepared' else 1


if __name__ == '__main__':
    raise SystemExit(main())
