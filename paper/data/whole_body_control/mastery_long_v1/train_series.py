"""Sequential CPU-only base mastery experiments; fixed budgets, no promotion."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from emet.rl.manipulation.mastery import load_suite, read_report, validate_preflight
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.regression_gate import verify_regressions

ROOT=Path.cwd()
RECOVERY=ROOT/'logs/recovery/mastery-long-v1'
SERIES=ROOT/'logs/manipulation/mars-mastery-long-v1'
PREFLIGHT=ROOT/'logs/eval/mars-mastery-long-preflight-v2/report.json.gz'
RECEIPT=RECOVERY/'regressions-post-main.receipt.json'
SEEDS=[90,91,92]
STEPS=262144
PERIOD=16384
lock=(RECOVERY/'series.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if SERIES.exists():
    raise RuntimeError('series output exists; refusing overwrite or duplicate training')
verify_regressions(RECEIPT)
validate_preflight(read_report(PREFLIGHT),load_suite(),['base'])
def verify_harness():
    harness=json.loads((RECOVERY/'harness-contracts.json').read_text())
    if not harness['all_pass']: raise RuntimeError('merged harness contracts have not passed')
    for name,expected in harness['source_hashes'].items():
        if Path(name).suffix=='.md': continue
        path=ROOT/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual!=expected: raise RuntimeError('harness source changed: '+name)
verify_harness()
sources=source_hashes()
SERIES.mkdir(parents=True)
status={'format':'mars-mastery-series-v1','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_hashes':sources,'preflight_sha256':hashlib.sha256(PREFLIGHT.read_bytes()).hexdigest(),'regression_receipt_sha256':hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),'seeds':SEEDS,'steps_per_seed':STEPS,'eval_every':PERIOD,'stages':['base'],'max_concurrent_envs':1,'cpus':sorted(os.sched_getaffinity(0)),'hardware_used':False,'safety_promoted':False,'automatic_extension':False,'automatic_stage_advancement':False,'started_unix':time.time(),'runs':[],'status':'starting'}
def save():
    temp=SERIES/'status.pending.json'
    temp.write_text(json.dumps(status,indent=2)+'\n')
    temp.replace(SERIES/'status.json')
save()
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mpl',PYTHONPATH=str(ROOT/'src'),WANDB_MODE='offline',WANDB_DISABLE_CODE='true')
try:
 for seed in SEEDS:
    if sources!=source_hashes(): raise RuntimeError('production sources changed')
    verify_regressions(RECEIPT)
    verify_harness()
    destination=SERIES/f'seed-{seed}'
    command=[sys.executable,'-m','emet.rl.manipulation.mastery','train','--stages','base','--steps',str(STEPS),'--seed',str(seed),'--eval-every',str(PERIOD),'--preflight',str(PREFLIGHT),'--regression-receipt',str(RECEIPT),'--wandb-mode','offline','--output',str(destination)]
    row={'seed':seed,'output':str(destination),'command':command,'started_unix':time.time(),'status':'running'}
    status['runs'].append(row);status['status']='running';status['active_seed']=seed;save()
    print('START',seed,flush=True)
    with (RECOVERY/f'seed-{seed}.log').open('x') as log:
       result=subprocess.run(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=False)
    row['returncode']=result.returncode;row['finished_unix']=time.time()
    if result.returncode!=0: raise RuntimeError(f'seed {seed} failed; inspect log; remaining runs not started')
    completion=json.loads((destination/'complete.json').read_text())
    if completion['steps']!=STEPS: raise RuntimeError('unexpected training budget')
    row['status']='complete';row['mastery']=completion['mastery'];save()
    print('COMPLETE',seed,'all_pass',completion['mastery']['all_pass'],flush=True)
 status['status']='complete';status['active_seed']=None
except BaseException as exc:
 status['status']='failed';status['error']=str(exc)
 raise
finally:
 status['updated_unix']=time.time();save()
