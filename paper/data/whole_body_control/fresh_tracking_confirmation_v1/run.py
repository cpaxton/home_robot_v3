"""Teacher/source/final actor readouts on a predeclared fresh FK distribution."""
import argparse,copy,hashlib,json,os
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import torch
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv,CoupledReference,admit_reference
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_training import arrays_digest
from emet.rl.regression_gate import ROOT,verify_regressions,source_state
REC=ROOT/'logs/recovery/fresh-tracking-confirmation-v1';OUT=ROOT/'logs/eval/fresh-tracking-confirmation-v1'
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def stop(e):
 e.control.stop();start=float(e.data.time)
 while not e._physics_invalid and not whole_body_stopped(e.control) and float(e.data.time)-start<.75-1e-9:e._tick(np.zeros(e.profile.action_size),score=False)
 return bool(not e._physics_invalid and whole_body_stopped(e.control))
def run_case(case,setting,actor):
 env=CoupledTrackingEnv('innate_mars',[case],suite_path=REC/'suite.json');e=env.unwrapped;e.config=replace(e.config,wheel_feedback_base_only_gains=(0.,.5),initial_heading_range=setting['initial_heading_range_rad']);row={'id':case['id'],'seed':case['seed'],'success':False,'executed':False,'stage_changed':False,'setting':setting}
 try:
  obs,initial=env.reset();first=e.control.task_trajectory
  row.update(initial_state_sha256=arrays_digest({'qpos':e.data.qpos.copy(),'qvel':e.data.qvel.copy()}),model_fingerprint=e.fingerprint,configuration_sha256=digest(asdict(e.config)),reference_sha256=initial['reference_sha256'],initial_base=e.control.base_pose().tolist(),admission=initial['coupled_admission'])
  plans=[first];second=None;switch=None
  if 'second_stage' in setting:
   change=setting['second_stage'];base,joints=first.coordinates(first.end);first.fk(base,joints);proxy=copy.copy(e.control);proxy.data=first.fk.data
   updated=dict(case,travel_m=change['travel_m'],turn_rad=change['turn_rad'],arm_delta=change['arm_delta']);second=CoupledReference(proxy,updated);admission=admit_reference(e,second);row['second_admission']=admission
   if not admission['accepted']:raise ValueError('confirmation_stage_rejected:'+str(admission['reason']))
   for before,after in zip(first.sample(first.end),second.sample(0),strict=True):np.testing.assert_allclose(before,after,atol=1e-9,rtol=0)
   for before,after in zip(first.path.sample(first.end)[1:3],second.path.sample(0)[1:3],strict=True):np.testing.assert_allclose(before,after,atol=1e-8,rtol=0)
   switch=float(np.ceil((first.end+change['after_nominal_end_s'])/e.profile.dt)*e.profile.dt);row['second_reference_sha256']=digest(second.record());row['switch_at_s']=switch;plans.append(second)
  row['goal_samples']=[{'time_s':float(t),'base':plan.sample(t)[0].tolist(),'hand':plan.sample(t)[1].tolist()} for plan in plans for t in np.linspace(0,plan.end,5)]
  origin=float(e.data.time)
  for tick in range(500):
   if second is not None and not row['stage_changed'] and float(e.data.time)-origin>=switch-1e-8:
    if e.control.stopped:raise RuntimeError('cannot rearm stopped control for stage change')
    e.set_task_trajectory(second);obs=env.observe(e.control.observe());row['stage_changed']=True
   if actor is None:action=task_pose_teacher_action(e.control)
   else:
    with torch.no_grad():action=actor({k:torch.from_numpy(obs[k])[None] for k in ('history','preview','motion')})[0].numpy().copy()
   obs,_,done,truncated,info=env.step(action)
   if done or truncated:break
  m=e._info();row.update(executed=True,success=bool(info.get('is_success')),code=info['code'],control_ticks=tick+1,metrics={k:m[k] for k in ('ee_position_error','ee_orientation_error','hand_target_active','settled_seconds','time_to_first_settle','peak_hold_base_drift_m','peak_base_linear_jerk_m_s3','peak_base_yaw_jerk_rad_s3','peak_contact_force')})
 except ValueError as exc:
  if not str(exc).startswith(('coupled_reference_rejected:','coupled_initialization_failed','confirmation_stage_rejected:')):raise
  row['code']=str(exc)
 finally:
  row['stop_confirmed']=stop(e);m=e._info();row['safety_flags']={k:bool(m[k]) for k in ('wheel_lift','unsafe_contact','retention_lost')}|{'invalid_physics':bool(e._physics_invalid)};env.close()
 return row
def main():
 p=argparse.ArgumentParser();p.add_argument('controller',choices=('teacher','ordinary','continued'));a=p.parse_args();torch.set_num_threads(1);receipt=verify_regressions(ROOT/'logs/eval/quiet-reference-regressions-v1/full.receipt.json');cases=json.loads((REC/'cases.json').read_text());protocol=json.loads((REC/'protocol.json').read_text());state=source_state(ROOT)
 if hashlib.sha256((REC/'cases.json').read_bytes()).hexdigest()!=protocol['case_sha256']:raise RuntimeError('case manifest changed')
 actor=None;bundle=None
 if a.controller!='teacher':
  path=ROOT/('logs/manipulation/quiet-reference-pilot-v1/ordinary_noise/checkpoint-16384/actor' if a.controller=='ordinary' else 'logs/manipulation/long-finetune-v1/continued/checkpoint-65536/actor');bundle=WholeBodyBundle(path);actor=bundle.actor
 report={'format':'fresh-tracking-readout-v1','controller':a.controller,'protocol_sha256':hashlib.sha256((REC/'protocol.json').read_bytes()).hexdigest(),'source_state_digest':digest(state),'regression_tests':len(receipt['test_ids']),'actor_sha256':None if bundle is None else bundle.manifest['weights_sha256'],'cases':[],'complete':False,'hardware_used':False,'safety_promoted':False}
 output=OUT/(a.controller+'.json')
 if output.exists():raise RuntimeError('output exists')
 for case in cases:
  row=run_case(case,protocol['settings'][case['id']],actor);report['cases'].append(row);save(output,report);print('FRESH',a.controller,case['id'],row['success'],row['code'],flush=True)
 if state!=source_state(ROOT):raise RuntimeError('source changed')
 report['summary']={'successes':sum(r['success'] for r in report['cases']),'attempts':len(cases),'executed':sum(r['executed'] for r in report['cases']),'confirmed_stops':sum(r['stop_confirmed'] for r in report['cases']),'unsafe_episodes':sum(any(r['safety_flags'].values()) for r in report['cases'])};report['complete']=True;save(output,report)
if __name__=='__main__':main()
