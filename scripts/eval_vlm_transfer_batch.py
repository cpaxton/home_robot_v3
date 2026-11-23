# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Sequential frozen-fixture expansion screen for runtime VLM manipulation."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from emet.utils.process_tree import popen_session, terminate_process_tree


def classify(row):
    if row.get('task_passed'):
        return 'success'
    diagnostic = row.get('diagnostic', {})
    if not diagnostic.get('events'):
        return 'grounding'
    if 'lift' not in diagnostic['events']:
        return 'grasp_or_approach'
    if not row.get('physical_pick_passed'):
        return 'grasp'
    if not diagnostic.get('completed'):
        return 'motion_or_recovery'
    closed = diagnostic.get('gripper_after_close', {})
    released = diagnostic.get('gripper_before_release', {})
    if closed and released and sum(abs(v) for v in closed.values()) > .002 and sum(abs(v) for v in released.values()) < .001:
        return 'retention_suspected'
    return 'placement_or_verification'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('fixture_root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--sim-pythonpath', required=True)
    parser.add_argument('--profile', type=Path, default=Path('configs/eval/pandaomron_vlm_one_shot_development.json'))
    parser.add_argument('--port-offset', type=int, default=6920)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    # Validate every input before starting any simulation or model request.
    for case in manifest['cases']:
        fixture = args.fixture_root/case['fixture']
        digest = hashlib.sha256((fixture/'stretch.xml').read_bytes()).hexdigest()
        if digest != case['fixture_sha256']:
            raise ValueError('frozen_fixture_changed: '+case['id'])
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    sources = ['scripts/eval_pandaomron_sink_agent.py', 'src/emet/eval/panda_open_loop.py',
               'src/emet/eval/vlm_pick_hint.py', 'src/emet/controller/task/tamp/transfer_grounding.py',
               'scripts/eval_vlm_transfer_batch.py']
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}
    (args.output/'source-hashes.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    for name in sources:
        snapshot = args.output/'source'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(name).read_bytes())
    frozen_profile = args.output/'profile.json'
    frozen_profile.write_bytes(args.profile.read_bytes())
    results = []
    env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'LP_NUM_THREADS': '2', 'PYTHONMALLOC': 'malloc', 'NUMBA_DISABLE_JIT': '1',
           'MUJOCO_GL': 'egl', 'EGL_PLATFORM': 'surfaceless', 'LIBGL_ALWAYS_SOFTWARE': '1'}
    for index, case in enumerate(manifest['cases']):
        if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
               for name, digest in source_hashes.items()):
            raise RuntimeError('evaluation source changed during frozen batch')
        output = args.output/case['id']
        spec = args.output/(case['id']+'-spec.json')
        spec.write_text(json.dumps(case['spec'], indent=2)+'\n')
        command = [sys.executable, 'scripts/eval_pandaomron_sink_agent.py',
            str((args.fixture_root/case['fixture']).resolve()), str(output.resolve()),
            '--case', case['case'], '--case-spec', str(spec.resolve()),
            '--scene-camera', 'right', '--port-offset', str(args.port_offset+index*20),
            '--diagnostic-torso-offset-m', '.002', '--sim-pythonpath', args.sim_pythonpath,
            '--observed-profile', str(frozen_profile.resolve()), '--open-loop-transfer',
            '--open-loop-object-description', case['spec']['object_name'],
            '--open-loop-destination', manifest['destination'], '--model-endpoint', args.endpoint,
            '--model', manifest['model']]
        started = time.monotonic()
        with (args.output/(case['id']+'.stdout.log')).open('w') as log:
            proc = popen_session(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                status = proc.wait(timeout=manifest['wall_limit_s'])
            except subprocess.TimeoutExpired:
                status = 'timeout'
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        terminate_process_tree(proc)
        result = {'id': case['id'], 'case': case['case'], 'exit_status': status,
                  'wall_s': time.monotonic()-started, 'artifact': str(output)}
        score = output/'open-loop-score.json'
        if score.exists():
            row = json.loads(score.read_text())
            attempts = row.get('diagnostic', {}).get('grounding_attempts', [])
            result.update(task_passed=row['task_passed'], category=classify(row),
                physical_pick_passed=row.get('physical_pick_passed', False),
                grounding_attempts=len(attempts),
                first_attempt_success=bool(row['task_passed'] and len(attempts) == 1),
                recovered_success=bool(row['task_passed'] and len(attempts) > 1),
                reason=row.get('diagnostic', {}).get('reason'),
                lift_m=row.get('physics', {}).get('lift_with_two_pads_m'),
                touch_frames=row.get('physics', {}).get('unintended_touch_frames'),
                stages=row.get('diagnostic', {}).get('events', []))
        else:
            result.update(task_passed=False, category='infrastructure_or_unreported',
                          reason='No scored receipt; inspect retained stdout and server logs')
        results.append(result)
        (args.output/'report.json').write_text(json.dumps({'scope': manifest['scope'],
            'completed': len(results), 'total': len(manifest['cases']), 'results': results}, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    return 0 if all(row['task_passed'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
