"""Bounded source-bound simulator pipeline; no robot command or deployment."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from emet.rl.regression_gate import ROOT,verify_regressions
REC=ROOT/'logs/recovery/goal-change-curriculum-v1'
STATE=REC/'pipeline.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_state(phase,**extra):
 temporary=STATE.with_suffix('.partial');temporary.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'hardware_used':False,**extra},indent=2)+'\n');os.replace(temporary,STATE)
def port_open():
 with socket.socket() as sock:return sock.connect_ex(('127.0.0.1',8766))==0
def restore_viewer():
 if port_open():return
 out=ROOT/'logs/jobs/goal-change-restored-viewer-v1';out.mkdir(parents=True,exist_ok=False);python=str(ROOT/'.venv/bin/python');env=dict(os.environ,MUJOCO_GL='egl');env.pop('WANDB_API_KEY',None)
 command=[python,'-u',str(ROOT/'logs/recovery/wrapper-reward-training-v1/supervise.py'),'--name','goal-change-restored-viewer-v1','--record-dir',str(out),'--cpu','14','--seconds','21600','--',python,'-u','-m','emet.rl.test_drive','--mode','mars-arm','--port','8766','--policy-root',str(ROOT/'logs/eval/base-goal-reference-native-v1'),'--policy-root',str(ROOT/'logs/eval/scoped-wheel-feedback-v2/candidate')]
 with (out/'job.log').open('x') as log:child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('retained viewer restored',child.pid,flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest-sha256',required=True);args=parser.parse_args()
 if sha(REC/'runner-sha256.json')!=args.manifest_sha256:raise RuntimeError('runner manifest changed before launch')
 expected=json.loads((REC/'runner-sha256.json').read_text())
 def verify_inputs():
  for name,digest in expected.items():
   if sha(REC/name)!=digest:raise RuntimeError('changed experiment input '+name)
 verify_inputs()
 revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
 if revision!=json.loads((REC/'protocol.json').read_text())['runtime_revision']:raise RuntimeError('wrong frozen revision')
 if port_open():raise RuntimeError('pause the owned native viewer before starting this pipeline')
 gate_dir=ROOT/'logs/jobs/goal-change-regressions-v1';gate_dir.mkdir(parents=True,exist_ok=False)
 python=str(ROOT/'.venv/bin/python')
 gate_command=[python,'-u',str(ROOT/'logs/recovery/wrapper-reward-training-v1/supervise.py'),'--name','goal-change-regressions-v1','--record-dir',str(gate_dir),'--cpu','15','--seconds','5400','--',python,'-u','-m','emet.rl.regression_gate','--output',str(ROOT/'logs/eval/goal-change-regressions-v1/full.xml')]
 try:
  with (gate_dir/'job.log').open('x') as log:gate=subprocess.Popen(gate_command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save_state('admission_and_regressions',regression_supervisor_pid=gate.pid)
  subprocess.run([python,'-u',str(REC/'run.py'),'admission'],check=True,timeout=1800)
  save_state('waiting_for_regressions',regression_supervisor_pid=gate.pid)
  deadline=time.monotonic()+5400
  while gate.poll() is None:
   if time.monotonic()>deadline:raise RuntimeError('regression wait deadline exceeded')
   time.sleep(5)
  if gate.returncode:raise RuntimeError('fresh regression job failed; PPO not started')
  verify_inputs();receipt=verify_regressions(ROOT/'logs/eval/goal-change-regressions-v1/full.receipt.json')
  save_state('training',regression_tests=len(receipt['test_ids']))
  subprocess.run([python,'-u',str(REC/'run.py'),'train'],check=True,timeout=10800)
  verify_inputs();subprocess.run([python,'-u',str(REC/'finalize.py')],check=True,timeout=600)
  save_state('complete',regression_tests=len(receipt['test_ids']))
 except BaseException as exc:
  save_state('failed',error=repr(exc));raise
 finally:restore_viewer()
if __name__=='__main__':main()
