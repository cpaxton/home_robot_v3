"""Bounded single-reward-factor experiment, matched old control and viewer restore."""
import hashlib,json,os,socket,subprocess,sys
from emet.rl.regression_gate import ROOT,verify_regressions
REC=ROOT/'logs/recovery/hand-priority-pilot-v1'
def verify_inputs():
 for name,expected in json.loads((REC/'runner-sha256.json').read_text()).items():
  if hashlib.sha256((REC/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('changed input '+name)
def restore_viewer():
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',8766))==0:return
 out=ROOT/'logs/jobs/hand-priority-restored-viewer-v1';out.mkdir(parents=True,exist_ok=False);python=str(ROOT/'.venv/bin/python');env=dict(os.environ,MUJOCO_GL='egl');env.pop('WANDB_API_KEY',None)
 command=[python,'-u',str(ROOT/'logs/recovery/wrapper-reward-training-v1/supervise.py'),'--name','hand-priority-restored-viewer-v1','--record-dir',str(out),'--cpu','14','--seconds','21600','--',python,'-u','-m','emet.rl.test_drive','--mode','mars-arm','--port','8766','--policy-root',str(ROOT/'logs/eval/base-goal-reference-native-v1'),'--policy-root',str(ROOT/'logs/eval/scoped-wheel-feedback-v2/candidate')]
 with (out/'job.log').open('x') as log:p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('retained viewer restored pid',p.pid,flush=True)
def main():
 try:
  verify_inputs();verify_regressions(ROOT/'logs/eval/quiet-reference-regressions-v1/full.receipt.json');subprocess.run([sys.executable,'-u',str(REC/'run.py')],check=True,timeout=3600);verify_inputs();subprocess.run([sys.executable,'-u',str(REC/'finalize.py')],check=True,timeout=600)
 finally:restore_viewer()
if __name__=='__main__':main()
