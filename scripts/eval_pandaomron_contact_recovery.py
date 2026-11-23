#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Isolate measured contact-recovery handoff on the archived Panda simulator.

Uses the production observed source, native planner and runtime executor. This
checks a short torso motion, not the full sink transfer or model tool selection.
Simulator state is confined to the separately running physics scorer.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np
from eval_pandaomron_sink_agent import CASES, observe

from emet.app.robot_cli import create_robot_client_from_cli
from emet.eval.pandaomron_live_transfer import panda_transfer_factory
from emet.simulation.sim_subprocess import wait_for_sim_tcp_port
from emet.utils.process_tree import popen_session, terminate_process_tree


def run(args):
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    scenes, client = [], None
    env = {**os.environ, 'PYTHONPATH': args.sim_pythonpath, 'EMET_PHYSICAL_EXECUTION': '1',
           'EMET_PHYSICAL_AUDIT': str(output/'actuation.jsonl')}
    command = [sys.executable, str(Path(__file__).with_name('eval_pandaomron_sink_agent.py')),
        str(args.fixture.resolve()), str(output), '--serve-fixture', '--case', 'held-out-shaker',
        '--scene-camera', 'left', '--diagnostic-torso-offset-m', '.002', '--port-offset', str(args.port_offset)]
    result, error, uses_departure, clearance_suffix = None, None, False, False
    before = after = None
    initial_ordinary_collision = False
    retained_clearance = False
    with (output/'server.log').open('w') as server_log:
        server = popen_session(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        def terminate_run(_signum, _frame):
            raise SystemExit(143)

        previous_handler = signal.signal(signal.SIGTERM, terminate_run)
        try:
            wait_for_sim_tcp_port('127.0.0.1', 4401+args.port_offset, proc=server, timeout_sec=60)
            client = create_robot_client_from_cli('pandaomron', '127.0.0.1', port_offset=args.port_offset)
            observe(client, output)
            factory, _ = panda_transfer_factory(args.profile.resolve(), output/'robot-only.xml',
                CASES['held-out-shaker']['object_name'], scenes)
            with (output/'controller.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                adapter = factory(client, None)
                task = next(iter(adapter.tasks.values()))
                snapshot = adapter._observe(task)
                executor = adapter.refine(task, snapshot)
                base_bound = executor.collision._robot_bound

                def buffered_bound(model, data, gid):
                    solid, error = base_bound(model, data, gid)
                    return solid.with_offset(solid.offset_m+args.additional_planning_buffer_m), error

                executor.collision._robot_bound = buffered_bound
                executor.synchronize(executor.data)
                adapter._refresh_checker(executor, task)
                before = executor.data.qpos[executor.qadr].copy()
                goal = before.copy()
                axis = executor.joint_names.index('mobilebase0_joint_torso_height')
                goal[axis] += args.distance_m
                # No posture guide is needed for this isolated straight torso test.
                raw_sync = executor.synchronize

                def synchronize(data):
                    raw_sync(data)
                    adapter._refresh_checker(executor, task, allow_scene_change=False)

                executor.synchronize = synchronize
                executor.approach_joint_routes = ()
                executor.ik_seed_configurations = ()
                with executor.collision.motion_phase('pregrasp'), executor.collision.certify_transfer():
                    with executor.collision.motion_scope(executor.model, executor.data, executor.joint_names, before):
                        initial_ordinary_collision = bool(executor.collision.configuration_collides(executor.model, executor.data))
                    path, error = executor.plan_joint_target(goal)
                if path is not None:
                    uses_departure = getattr(path, 'departure', None) is not None
                    retained_clearance = callable(getattr(path, 'clearance_scope', None))
                    end = getattr(path, 'escape_end_index', None)
                    clearance_suffix = end is not None and end < len(path)-1 and callable(getattr(path, 'clearance_scope', None))
                    adapter._refresh_checker(executor, task)
                    result = executor._execute_path('pregrasp', path)
                executor.synchronize(executor.data)
                after = executor.data.qpos[executor.qadr].copy()
                (output/'controller-result.json').write_text(json.dumps({
                    'planning_error': error, 'result': asdict(result) if result else None,
                    'uses_departure': uses_departure, 'clearance_suffix': clearance_suffix,
                    'initial_ordinary_collision': initial_ordinary_collision, 'retained_clearance': retained_clearance, 'joint_names': list(executor.joint_names),
                    'before': before.tolist(), 'after': after.tolist(), 'goal': goal.tolist(),
                    'departure_diagnostics': executor.collision.last_departure_diagnostics,
                }, indent=2)+'\n')
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            for scene in scenes:
                scene.invalidate()
            if client is not None:
                client.stop()
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate_process_tree(server)
    physics = json.loads((output/'physics.json').read_text())
    audit = output/'actuation.jsonl'
    count = len(audit.read_text().splitlines()) if audit.exists() else 0
    progress = float(after[axis]-before[axis]) if after is not None else 0.
    passed = bool(result and result.success and initial_ordinary_collision and retained_clearance and
        (not uses_departure or clearance_suffix) and count and
        progress >= args.distance_m-.002 and not physics['unintended_contact_frames'] and
        not physics['unintended_self_contact_frames'])
    profile = json.loads(args.profile.read_text())
    if profile.get('simulation_ignore_environment'):
        passed = False
    report = {'passed': passed, 'profile': str(args.profile.resolve()),
              'simulation_settings': {key: profile[key] for key in (
                  'simulation_visual_self_association_m', 'simulation_allow_unknown',
                  'simulation_ignore_environment', 'simulation_clearance_slack_m', 'simulation_best_fit_pose') if key in profile}, 'scope': 'contact_recovery_handoff', 'routing_mode': 'native_executor_probe',
              'full_transfer_test': False, 'initial_ordinary_collision': initial_ordinary_collision,
              'retained_clearance': retained_clearance, 'transition_kind': 'sampled_departure' if uses_departure else 'nominally_clear_start', 'uses_departure': uses_departure, 'clearance_suffix': clearance_suffix,
              'additional_planning_buffer_m': args.additional_planning_buffer_m,
              'requested_distance_m': args.distance_m, 'measured_progress_m': progress,
              'actuation_count': count, 'planning_error': error,
              'execution': asdict(result) if result else None, 'physics': physics}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--sim-pythonpath', required=True)
    parser.add_argument('--port-offset', type=int, default=4520)
    parser.add_argument('--distance-m', type=float, default=.008)
    parser.add_argument('--additional-planning-buffer-m', type=float, default=.005)
    args = parser.parse_args()
    if not np.isfinite(args.distance_m) or not .005 <= args.distance_m <= .03:
        parser.error('distance must be between 5 and 30 mm')
    if not np.isfinite(args.additional_planning_buffer_m) or not 0 <= args.additional_planning_buffer_m <= .05:
        parser.error('additional planning buffer must be between 0 and 50 mm')
    sys.exit(run(args))
