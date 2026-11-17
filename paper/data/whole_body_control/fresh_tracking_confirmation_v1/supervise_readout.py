"""Run fresh teacher/source then the fixed final learned actor, without selection."""
import hashlib,json,subprocess,sys,time
from emet.rl.regression_gate import ROOT,verify_regressions
from emet.utils.job_registry import load_job,refresh_job_liveness
REC=ROOT/'logs/recovery/fresh-tracking-confirmation-v1';OUT=ROOT/'logs/eval/fresh-tracking-confirmation-v1'
def verify():
 for name,expected in json.loads((REC/'runner-sha256.json').read_text()).items():
  if hashlib.sha256((REC/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('changed fresh benchmark input '+name)
def main():
 verify();verify_regressions(ROOT/'logs/eval/quiet-reference-regressions-v1/full.receipt.json')
 for controller in ('teacher','ordinary'):subprocess.run([sys.executable,'-u',str(REC/'run.py'),controller],check=True,timeout=1800)
 deadline=time.monotonic()+9000
 while True:
  job=refresh_job_liveness(load_job('20261117_083406_6c3285'))
  if job.status in ('failed','cancelled'):raise RuntimeError('fine-tuning did not finish successfully')
  if job.status=='done':break
  if time.monotonic()>deadline:raise TimeoutError('fixed final checkpoint deadline')
  time.sleep(10)
 verify();subprocess.run([sys.executable,'-u',str(REC/'run.py'),'continued'],check=True,timeout=1800)
 reports={name:json.loads((OUT/(name+'.json')).read_text()) for name in ('teacher','ordinary','continued')}
 for name,r in reports.items():
  if not r['complete']:raise RuntimeError('partial report')
  for a,b in zip(reports['teacher']['cases'],r['cases'],strict=True):
   if a['id']!=b['id']:raise RuntimeError('case IDs differ')
   for key in ('initial_state_sha256','model_fingerprint','configuration_sha256','reference_sha256','second_reference_sha256'):
    if a.get(key)!=b.get(key):raise RuntimeError('unmatched initial state or announced reference '+key)
 summary={'format':'fresh-tracking-confirmation-summary-v1','protocol_sha256':hashlib.sha256((REC/'protocol.json').read_bytes()).hexdigest(),'source_state_digest':reports['teacher']['source_state_digest'],'complete':True,'all_initial_states_and_reference_hashes_matched':True,'controllers':{k:v['summary'] for k,v in reports.items()},'hardware_used':False,'safety_promoted':False}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('FRESH_SUMMARY',json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
