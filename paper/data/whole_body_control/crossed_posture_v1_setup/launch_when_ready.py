"""Run one source-gated crossed-posture refinement, then expanded final replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from emet.rl.manipulation.mastery import load_suite, read_report, validate_preflight
from emet.rl.regression_gate import verify_regressions

root = Path.cwd()
recovery = root / 'logs/recovery/mastery-crossed-v1'
receipt = recovery / 'regressions.receipt.json'
suite = root / 'configs/benchmarks/mars_tracking_mastery_crossed_v1.json'
teacher = root / 'logs/eval/mars-mastery-crossed-v1-teacher/report.json.gz'
output = root / 'logs/manipulation/mars-mastery-crossed-v1-seed90'
status = {'status': 'waiting_for_gates', 'started_unix': time.time(), 'seed': 90,
          'steps': 65536, 'repeats': False, 'hardware_used': False}

def save():
    target = recovery / 'launch-status.json'
    temporary = target.with_suffix('.pending.json')
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(target)

save()
try:
    for attempt in range(720):
        if receipt.exists() and teacher.exists():
            evidence = verify_regressions(receipt)
            validate_preflight(read_report(teacher), load_suite(suite), ['base'])
            break
        for name in ['regression-driver.log', 'preflight.log']:
            path = recovery / name
            if path.exists() and 'Traceback' in path.read_text():
                raise RuntimeError('Failed gate: ' + name)
        time.sleep(5)
    else:
        raise RuntimeError('Gates did not complete within one hour')
    assert not output.exists(), 'Refusing duplicate or overwritten training'
    command = [sys.executable, '-m', 'emet.rl.manipulation.mastery', 'train',
               '--suite', str(suite), '--stages', 'base', '--steps', '65536',
               '--seed', '90', '--eval-every', '16384', '--learning-rate', '0.00005',
               '--base-std', '0.005', '--arm-std', '0.01', '--command-smoothness-warmup', '0',
               '--initialize-from', str(root / 'logs/manipulation/mars-mastery-unfiltered-v1-seed90/last'),
               '--preflight', str(teacher), '--regression-receipt', str(receipt),
               '--wandb-mode', 'offline', '--output', str(output)]
    status.update(status='training', controller_tests=len(evidence['test_ids']),
                  teacher_cases=len(read_report(teacher)['cases']), command=command,
                  updated_unix=time.time())
    save()
    print('GATE PASS', status['controller_tests'], 'tests,', status['teacher_cases'], 'teacher cases', flush=True)
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
    status.update(status='expanded_final_replay', updated_unix=time.time())
    save()
    subprocess.run([sys.executable, '-m', 'emet.rl.manipulation.mastery', 'evaluate',
                    '--suite', str(suite), '--stages', 'base', '--population', 'admission',
                    '--policy', str(output / 'last'), '--preflight', str(teacher),
                    '--output', str(root / 'logs/eval/mars-mastery-crossed-v1-final-expanded')],
                   check=True, stdin=subprocess.DEVNULL)
    status.update(status='complete', completed_unix=time.time())
    save()
except BaseException as exc:
    status.update(status='failed', error=str(exc), updated_unix=time.time())
    save()
    raise
