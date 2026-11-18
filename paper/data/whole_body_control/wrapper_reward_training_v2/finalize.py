"""Wait for the fixed paired run, then summarize physical outcomes offline."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from emet.rl.regression_gate import ROOT,verify_regressions
from emet.utils.job_registry import load_job,refresh_job_liveness
from emet.rl.manipulation.carry_benchmark import digest
REC=ROOT/'logs/recovery/wrapper-reward-training-v2';OUT=ROOT/'logs/manipulation/wrapper-reward-training-v2'
deadline=time.monotonic()+15000
while True:
 job=refresh_job_liveness(load_job('20261116_224923_96115d'))
 if job.status=='done':break
 if job.status in ('failed','cancelled'):raise RuntimeError('paired training did not complete: '+job.status)
 if time.monotonic()>deadline:raise TimeoutError('paired training deadline')
 time.sleep(10)
receipt=verify_regressions(ROOT/'logs/eval/wrapper-reward-regressions-v1/full.receipt.json')
read=lambda p:json.loads(Path(p).read_text())
cases=read(REC/'cases.json');splits={c['id']:c['split'] for c in cases};report={'format':'wrapper-reward-physical-comparison-v2','protocol':read(REC/'protocol.json'),'regression_tests':len(receipt['test_ids']),'source_state_digest':read(OUT/'admission.json')['source_state_digest'],'cases':{},'summary':{},'paired':read(OUT/'paired-comparison.json')['initial_state_and_first_actions_identical'],'hardware_used':False,'safety_promoted':False,'qualification':False}
for name in ('new-cases','frozen-coupled'):
 rows={'initial':read(OUT/f'initial-{name}.json')}
 rows.update({arm:read(OUT/f'{arm}-final-{name}.json') for arm in ('contract','legacy')})
 if any([r['id'] for r in rows['initial']]!=[r['id'] for r in rows[arm]] for arm in ('contract','legacy')):raise RuntimeError('case identity mismatch')
 for arm in ('contract','legacy'):
  for before,after in zip(rows['initial'],rows[arm],strict=True):
   if any(before[k]!=after[k] for k in ('initial_state_sha256','model_fingerprint','reference_sha256')):raise RuntimeError('physical comparison is not matched: '+after['id'])
 report['cases'][name]=rows
 for controller,items in rows.items():
  populations={'all':items}
  if name=='new-cases':populations|={split:[r for r in items if splits[r['id']]==split] for split in ('train','audit')}
  for population,group in populations.items():
   metrics={key:(float(np.max(values)) if (values:=[r['metrics'][key] for r in group if not (key.startswith('ee_') or key=='peak_hand_position_error') or r['metrics'].get('hand_target_active')]) else None) for key in ('ee_position_error','ee_orientation_error','peak_hand_position_error','peak_base_linear_jerk_m_s3','peak_base_yaw_jerk_rad_s3','peak_contact_force','peak_hold_base_drift_m')}
   report['summary'][f'{name}/{population}/{controller}']={'successes':sum(r['success'] for r in group),'attempts':len(group),'confirmed_stops':sum(r['stop_confirmed'] for r in group),'unsafe_episodes':sum(any(r['safety_flags'].values()) for r in group),'active_hand_cases':sum(bool(r['metrics'].get('hand_target_active')) for r in group),'worst':metrics}
tamp={'initial':read(OUT/'initial-tamp.json')}|{arm:read(OUT/arm/'final-tamp.json') for arm in ('contract','legacy')};report['cases']['tamp']=tamp
for name,rows in tamp.items():
 report['summary']['tamp/'+name]={'successes':sum(r['success'] for r in rows),'attempts':len(rows),'confirmed_stops':sum(r.get('stop_confirmed') is True for r in rows),'unsafe_episodes':sum(any(r.get('safety_flags', {'execution_error':True}).values()) for r in rows)}
report['complete']=True
(OUT/'physical-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
import wandb
run=wandb.init(project='emet-mars-control',name='wrapper-reward-v2-physical-comparison',mode='offline',dir=str(OUT),config={'protocol_sha256':digest(report['protocol']),'source_state_digest':report['source_state_digest'],'seed':107,'qualification':False},settings=wandb.Settings(disable_git=True,save_code=False))
table=wandb.Table(columns=['cohort','successes','attempts','stops','unsafe_episodes'])
for key,values in report['summary'].items():table.add_data(key,values['successes'],values['attempts'],values['confirmed_stops'],values['unsafe_episodes'])
run.log({'physical_outcomes':table,'matched_initial_state_and_first_actions':True});record={'id':run.id,'mode':'offline','directory':run.dir};run.finish();(OUT/'wandb-comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'summary':report['summary'],'wandb':record},indent=2),flush=True)
