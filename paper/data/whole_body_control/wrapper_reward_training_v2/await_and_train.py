"""Launch learning only after both explicit prerequisite jobs succeed."""
import os,subprocess,sys,time
from pathlib import Path
from emet.utils.job_registry import load_job,refresh_job_liveness,update_job
from emet.rl.regression_gate import ROOT,verify_regressions
jobs=['20261116_224138_a8c555','20261116_224836_3e01ed']
deadline=time.monotonic()+7200
while True:
 statuses=[]
 for identity in jobs:
  job=load_job(identity)
  if job is None:raise RuntimeError('missing prerequisite '+identity)
  job=refresh_job_liveness(job);statuses.append(job.status)
  if job.status in ('failed','cancelled'):raise RuntimeError('prerequisite '+identity+' '+job.status)
 if all(s=='done' for s in statuses):break
 if time.monotonic()>deadline:raise TimeoutError('prerequisite deadline exceeded')
 time.sleep(10)
verify_regressions(ROOT/'logs/eval/wrapper-reward-regressions-v1/full.receipt.json')
identity=__import__('json').loads((ROOT/'logs/jobs/wrapper-reward-training-v2/job.json').read_text())['job_id']
update_job(identity,status='running',phase='critic_warmup_then_ppo')
print('prerequisites verified; starting actual paired reward PPO',flush=True)
raise SystemExit(subprocess.call([sys.executable,'-u',str(ROOT/'logs/recovery/wrapper-reward-training-v2/run.py'),'train']))
