"""Fresh gate and admission, bounded noise pair, automatic retained viewer restore."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
from emet.utils.job_registry import load_job,refresh_job_liveness,update_job
from emet.rl.regression_gate import ROOT,verify_regressions
REC=ROOT/'logs/recovery/quiet-reference-pilot-v1'
JOBS=json.loads((REC/'prerequisite-jobs.json').read_text())
def verify_runners():
 for name,expected in json.loads((REC/'runner-sha256.json').read_text()).items():
  if hashlib.sha256((REC/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('changed experiment input '+name)
def restore_viewer():
 deadline=time.monotonic()+3600
 while refresh_job_liveness(load_job(JOBS['quiet-reference-admission-v1']['job_id'])).status not in ('done','failed','cancelled'):
  if time.monotonic()>deadline:print('viewer restore deferred: native admission still owns CPU14',flush=True);return
  time.sleep(5)
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',8766))==0:print('existing viewer preserved',flush=True);return
 out=ROOT/'logs/jobs/quiet-reference-restored-viewer-v1';out.mkdir(parents=True,exist_ok=False)
 python=str(ROOT/'.venv/bin/python');supervisor=ROOT/'logs/recovery/wrapper-reward-training-v1/supervise.py'
 command=[python,'-u',str(supervisor),'--name','quiet-reference-restored-viewer-v1','--record-dir',str(out),'--cpu','14','--seconds','21600','--',python,'-u','-m','emet.rl.test_drive','--mode','mars-arm','--port','8766','--policy-root',str(ROOT/'logs/eval/base-goal-reference-native-v1'),'--policy-root',str(ROOT/'logs/eval/scoped-wheel-feedback-v2/candidate')]
 environment=dict(os.environ,MUJOCO_GL='egl');environment.pop('WANDB_API_KEY',None)
 with (out/'job.log').open('x') as log:child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('retained viewer restored pid',child.pid,flush=True)
def main():
 deadline=time.monotonic()+7200
 try:
  verify_runners()
  while True:
   statuses=[]
   for entry in JOBS.values():
    identity=entry['job_id'];job=load_job(identity)
    if job is None:raise RuntimeError('missing prerequisite '+identity)
    job=refresh_job_liveness(job);statuses.append(job.status)
    if job.status in ('failed','cancelled'):raise RuntimeError('prerequisite '+identity+' '+job.status)
   if all(s=='done' for s in statuses):break
   if time.monotonic()>deadline:raise TimeoutError('prerequisite deadline')
   time.sleep(10)
  verify_runners();verify_regressions(ROOT/'logs/eval/quiet-reference-regressions-v1/full.receipt.json')
  identity=json.loads((ROOT/'logs/jobs/quiet-reference-pilot-v1/job.json').read_text())['job_id'];update_job(identity,status='running',phase='ordinary_then_quiet_reference_ppo')
  print('fresh prerequisites verified; starting fixed-budget exploration pair',flush=True)
  subprocess.run([sys.executable,'-u',str(REC/'run.py'),'train'],check=True,timeout=5400)
  verify_runners();subprocess.run([sys.executable,'-u',str(REC/'finalize.py')],check=True,timeout=600)
 finally:restore_viewer()
if __name__=='__main__':main()
