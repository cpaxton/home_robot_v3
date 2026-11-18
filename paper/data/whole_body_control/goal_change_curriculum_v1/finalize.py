"""Compare fixed final mean policies; distinguish new goal audits and reused cases."""
import hashlib,json
from pathlib import Path
from emet.rl.regression_gate import ROOT,verify_regressions
OUT=ROOT/'logs/manipulation/goal-change-curriculum-v1';REC=ROOT/'logs/recovery/goal-change-curriculum-v1'
read=lambda p:json.loads(Path(p).read_text())
def main():
 receipt=verify_regressions(ROOT/'logs/eval/goal-change-regressions-v1/full.receipt.json');spec=read(REC/'curriculum.json');splits={c['id']:c['split'] for c in spec['robots']['innate_mars']};report={'format':'goal-change-curriculum-readout-v1','protocol':read(REC/'protocol.json'),'regression_tests':len(receipt['test_ids']),'complete':False,'hardware_used':False,'safety_promoted':False,'summary':{},'cases':{}}
 cohorts={'regular':(read(OUT/'initial-new-cases.json'),read(OUT/'curriculum-final-new-cases.json')),'existing-coupled':(read(OUT/'initial-frozen-coupled.json'),read(OUT/'curriculum-final-frozen-coupled.json')),'tamp':(read(OUT/'initial-tamp.json'),read(OUT/'curriculum/final-tamp.json')),'goal-changes':(read(OUT/'initial-goals.json'),read(OUT/'curriculum-final-goals.json'))}
 for cohort,(source,candidate) in cohorts.items():
  if len(source)!=len(candidate):raise RuntimeError('case count mismatch')
  for a,b in zip(source,candidate,strict=True):
   for key in ('id','seed','initial_state_sha256','model_fingerprint','reference_sha256','compiled_robot_identity'):
    if a.get(key)!=b.get(key):raise RuntimeError(cohort+' mismatched '+key)
   if cohort=='goal-changes' and a.get('goal_change')!=b.get('goal_change'):raise RuntimeError('nominal stage references differ')
  report['cases'][cohort]={'source':source,'curriculum':candidate}
  for name,rows in report['cases'][cohort].items():
   groups={'all':rows}
   if cohort=='goal-changes':groups|={split:[r for r in rows if splits[r['id']]==split] for split in ('train','audit')}
   for split,group in groups.items():
    values={'successes':sum(r['success'] for r in group),'attempts':len(group),'confirmed_stops':sum(r.get('stop_confirmed') is True for r in group),'unsafe_episodes':sum(any(r.get('safety_flags',{'missing':True}).values()) for r in group)}
    if cohort!='tamp':values['worst']={key:max((r['metrics'][key] for r in group if not key.startswith('ee_') or r['metrics']['hand_target_active']),default=None) for key in ('ee_position_error','ee_orientation_error','peak_base_linear_jerk_m_s3','peak_base_yaw_jerk_rad_s3','peak_contact_force')}
    report['summary'][cohort+'/'+split+'/'+name]=values
 report['complete']=True;(OUT/'physical-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('FINAL_COMPARISON',json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':main()
