#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Run the actual run-agent CLI against live physics; score outside the agent.

Explicit privileged stationary Stretch diagnostic, no images or local GPU.
Use emet jobs; task success and failure-boundary checks are reported separately.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
from eval_native_table_agent import goal_predicates

from emet.config.sim_launch_config import load_sim_launch_config_from_path
from emet.eval.native_table_live import live_table_robot
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.utils.process_tree import popen_session, terminate_process_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-sim', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--remote-model', required=True)
    parser.add_argument('--placement-regions', type=Path, required=True)
    parser.add_argument('--reject-first', action='store_true', help='Request one invalid-plan call, then a separate authorized task turn')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    config = load_sim_launch_config_from_path(str(args.live_sim))
    if not config.no_cameras:
        parser.error('This diagnostic requires no_cameras: true')
    goals = dict(zip(('object2', 'object1'), json.loads(args.placement_regions.read_text()), strict=True))
    os.environ['EMET_PHYSICAL_EXECUTION'] = '1'
    os.environ['EMET_PHYSICAL_AUDIT'] = str(out / 'actuation.jsonl')
    instruction = (
        'Execute both pick-and-place goals on the table. These are privileged simulator inputs. '
        'Discover tasks, plan with placement_region, execute returned plan_ref, read results, '
        'and rediscover before the next object. Whole footprints must fit the regions. '
        'Return one tool call per round; do not stop at a plan. Stop on any partial or unknown '
        'execution; no held-object recovery is available. Do not drop constraints. Goals: '
        + json.dumps({'red cylinder': goals['object2'], 'blue cube': goals['object1']})
    )
    commands = []
    if args.reject_first:
        commands.append('Diagnostic only: call execute_pick_place_plan once with plan_ref '
                        '"plan:deliberately-invalid". It must be rejected. Report the failure and stop; '
                        'do not plan or move anything in this turn.')
    commands.append(instruction)
    cli = [sys.executable, '-m', 'emet.app.run_agent', '--config', str(repo / 'configs/emet/default.yaml'),
           '--robot', 'stretch', '--robot-ip', '127.0.0.1', '--device', 'cpu', '--memory-backend', 'dynamem',
           '--state-only-sim', '--max-tool-rounds', '12', '--no-discord', '--no-vl-camera',
           '--headless', '--debug-tools', '--port-offset', str(config.port_offset),
           '--llm', f'openai@{args.base_url}#{args.remote_model}',
           '--set', 'agent.manip_mode=physical', '--set', 'agent.tamp_inputs=privileged']
    for command in commands:
        cli += ['--command', command]
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    with live_table_robot(args.live_sim, out, model) as robot:
        initial = goal_predicates(robot, goals)
        protocol = {'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    'cli': cli, 'goals': goals, 'initial_predicates': initial,
                    'scope': 'actual run-agent CLI; live physics; privileged stationary Stretch; no cameras',
                    'cpu_affinity': sorted(os.sched_getaffinity(0)), 'sim_config': args.live_sim.read_text()}
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        with (out / 'agent.log').open('w') as log:
            proc = popen_session(cli, cwd=out, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=900)
            finally:
                terminate_process_tree(proc)
        stable = True
        for _ in range(40):
            robot.step(.05)
            stable &= all(r['satisfied'] for r in goal_predicates(robot, goals).values())
        rows = [json.loads(line) for p in sorted((out / 'logs/chat').glob('*.jsonl'))
                for line in p.read_text().splitlines()]
        outcomes = [row['content'] for row in rows if row['role'] == 'tool']
        audit = [json.loads(line) for line in (out / 'actuation.jsonl').read_text().splitlines()]
        envelopes = []
        for row in outcomes:
            try:
                envelopes.append(json.loads(row.split('] ', 1)[1]))
            except (ValueError, IndexError):
                pass
        rejection_seen = any(e.get('code') == 'unknown_plan' for e in envelopes)
        result = {'task_passed': rc == 0 and stable and not any(r['satisfied'] for r in initial.values()),
                  'cli_exit_code': rc, 'requested_rejection_observed': rejection_seen if args.reject_first else None,
                  'final_predicates': goal_predicates(robot, goals), 'tool_outcomes': envelopes,
                  'audit_commands': len(audit), 'forbidden_commands': sum(bool(r.get('forbidden')) for r in audit)}
        result['passed'] = result['task_passed'] and bool(audit) and result['forbidden_commands'] == 0 and (
            not args.reject_first or rejection_seen)
        (out / 'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
