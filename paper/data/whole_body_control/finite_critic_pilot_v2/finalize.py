"""Physical results for finite-contract joint versus independent critic PPO."""
import json
from pathlib import Path
from emet.rl.regression_gate import ROOT,verify_regressions
REC=ROOT/'logs/recovery/finite-critic-pilot-v2';OUT=ROOT/'logs/manipulation/finite-critic-pilot-v2'
read=lambda p:json.loads(Path(p).read_text())
def main():
 receipt=verify_regressions(ROOT/'logs/eval/finite-critic-regressions-v2/full.receipt.json');protocol=read(REC/'protocol.json');arms=protocol['reward_arms'];cases=read(REC/'cases.json');splits={c['id']:c['split'] for c in cases}
 report={'format':'finite-critic-physical-comparison-v1','protocol':protocol,'regression_tests':len(receipt['test_ids']),'source_state_digest':read(OUT/'admission.json')['source_state_digest'],'paired':read(OUT/'paired-comparison.json')['initial_state_and_first_actions_identical'],'cases':{},'summary':{},'complete':False,'hardware_used':False,'safety_promoted':False}
 for cohort in ('new-cases','frozen-coupled'):
  rows={'initial':read(OUT/f'initial-{cohort}.json')}|{arm:read(OUT/f'{arm}-final-{cohort}.json') for arm in arms};report['cases'][cohort]=rows
  for arm in arms:
   if [r['id'] for r in rows[arm]]!=[r['id'] for r in rows['initial']]:raise RuntimeError('case identity mismatch')
   for a,b in zip(rows['initial'],rows[arm],strict=True):
    if any(a[k]!=b[k] for k in ('initial_state_sha256','model_fingerprint','reference_sha256')):raise RuntimeError('unmatched physical comparison')
  for controller,items in rows.items():
   groups={'all':items}
   if cohort=='new-cases':groups|={split:[r for r in items if splits[r['id']]==split] for split in ('train','audit')}
   for population,group in groups.items():
    worst={key:(max(values) if (values:=[r['metrics'][key] for r in group if not key.startswith('ee_') or r['metrics']['hand_target_active']]) else None) for key in ('ee_position_error','ee_orientation_error','peak_base_linear_jerk_m_s3','peak_base_yaw_jerk_rad_s3','peak_contact_force','peak_hold_base_drift_m')}
    report['summary'][f'{cohort}/{population}/{controller}']={'successes':sum(r['success'] for r in group),'attempts':len(group),'confirmed_stops':sum(r['stop_confirmed'] for r in group),'unsafe_episodes':sum(any(r['safety_flags'].values()) for r in group),'worst':worst}
 tamp={'initial':read(OUT/'initial-tamp.json')}|{arm:read(OUT/arm/'final-tamp.json') for arm in arms};report['cases']['tamp']=tamp
 for controller,rows in tamp.items():report['summary']['tamp/'+controller]={'successes':sum(r['success'] for r in rows),'attempts':len(rows),'confirmed_stops':sum(r.get('stop_confirmed') is True for r in rows),'unsafe_episodes':sum(any(r.get('safety_flags',{'execution_error':True}).values()) for r in rows)}
 report['complete']=True;(OUT/'physical-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 import wandb
 run=wandb.init(project='emet-mars-control',name='finite-critic-v1-physical-comparison',mode='offline',dir=str(OUT),config={'source_state_digest':report['source_state_digest'],'paired':report['paired'],'hardware_used':False},settings=wandb.Settings(disable_git=True,save_code=False));table=wandb.Table(columns=['cohort','successes','attempts','stops','unsafe_episodes'])
 for key,v in report['summary'].items():table.add_data(key,v['successes'],v['attempts'],v['confirmed_stops'],v['unsafe_episodes'])
 run.log({'physical_outcomes':table});record={'id':run.id,'directory':run.dir,'mode':'offline'};run.finish();(OUT/'wandb-comparison.json').write_text(json.dumps(record,indent=2)+'\n');print('PHYSICAL_COMPARISON',json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':main()
