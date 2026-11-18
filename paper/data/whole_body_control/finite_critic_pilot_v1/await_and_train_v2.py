"""Explicit fresh prerequisites, fixed paired budget and automatic viewer restoration."""
import json,os,socket,subprocess,sys,time
from emet.utils.job_registry import load_job,refresh_job_liveness,update_job
from emet.rl.regression_gate import ROOT,verify_regressions
REC=ROOT/'logs/recovery/finite-critic-pilot-v1'
def restore_viewer():
 deadline=time.monotonic()+3600
 while refresh_job_liveness(load_job('20261117_002841_fc6dd2')).status not in ('done','failed','cancelled'):
  if time.monotonic()>deadline:print('viewer restoration deferred: native admission still owns CPU14',flush=True);return
  time.sleep(5)
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',8766))==0:print('existing viewer preserved',flush=True);return
 out=ROOT/'logs/jobs/finite-critic-restored-viewer-v1';out.mkdir(parents=True,exist_ok=False)
 python=str(ROOT/'.venv/bin/python');supervisor=ROOT/'logs/recovery/wrapper-reward-training-v1/supervise.py'
 command=[python,'-u',str(supervisor),'--name','finite-critic-restored-viewer-v1','--record-dir',str(out),'--cpu','14','--seconds','21600','--',python,'-u','-m','emet.rl.test_drive','--mode','mars-arm','--port','8766','--policy-root',str(ROOT/'logs/eval/base-goal-reference-native-v1'),'--policy-root',str(ROOT/'logs/eval/scoped-wheel-feedback-v2/candidate')]
 environment=dict(os.environ,MUJOCO_GL='egl');environment.pop('WANDB_API_KEY',None)
 with (out/'job.log').open('x') as log:child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('retained viewer restored pid',child.pid,flush=True)
def main():
 jobs=['20261117_002837_d29cc2','20261117_002841_fc6dd2'];deadline=time.monotonic()+7200
 try:
  while True:
   statuses=[]
   for identity in jobs:
    job=load_job(identity)
    if job is None:raise RuntimeError('missing prerequisite '+identity)
    job=refresh_job_liveness(job);statuses.append(job.status)
    if job.status in ('failed','cancelled'):raise RuntimeError('prerequisite '+identity+' '+job.status)
   if all(s=='done' for s in statuses):break
   if time.monotonic()>deadline:raise TimeoutError('prerequisite deadline')
   time.sleep(10)
  verify_regressions(ROOT/'logs/eval/finite-critic-regressions-v1/full.receipt.json');identity=json.loads((ROOT/'logs/jobs/finite-critic-pilot-v2/job.json').read_text())['job_id'];update_job(identity,status='running',phase='finite_joint_then_independent_critic_ppo')
  print('fresh prerequisites verified; starting corrected PPO',flush=True)
  subprocess.run([sys.executable,'-u',str(REC/'run.py'),'train'],check=True)
  subprocess.run([sys.executable,'-u',str(REC/'finalize.py')],check=True)
 finally:restore_viewer()
if __name__=='__main__':main()
