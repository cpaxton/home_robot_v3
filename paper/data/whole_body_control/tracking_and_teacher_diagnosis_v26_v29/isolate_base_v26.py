"""Matched channel interventions after v25; diagnostics, never deployment policies."""
import gzip,hashlib,json
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch
from emet.rl.manipulation.env import ManipulationConfig,ManipulationEnv
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.regression_gate import verify_regressions

torch.set_num_threads(1);verify_regressions(Path('logs/recovery/20261028/failure-replay-regressions-v2.receipt.json'))
base=Path('logs/eval/failure-replay-development-v1-20261028');corpus=json.load(open(base/'corpus.json'));cfg=ManipulationConfig(**corpus['config']);audit=json.loads(gzip.decompress(Path('logs/eval/failure-replay-children-v1-20261028/report.json.gz').read_bytes()))
new=ManipulationPolicy('logs/manipulation/mars-failure-replay-v25/tracking_replay/seed-87/last');old=ManipulationPolicy('logs/manipulation/mars-jerk-tracking-v20/control/seed-86/last')
out=Path('logs/eval/base-isolation-v26-20261028');out.mkdir(exist_ok=False,parents=True)
r={'scope':__doc__,'source_hashes':source_hashes(),'config':asdict(cfg),'cases':[],'complete':False,'hardware_used':False,'safety_promoted':False,'driver_source':Path(__file__).read_text()}
def save():
 raw=(json.dumps(r,allow_nan=False)+'\n').encode();(out/'report.json.gz').write_bytes(gzip.compress(raw,mtime=0));(out/'summary.json').write_text(json.dumps({'complete':r['complete'],'scope':__doc__,'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'cases':[{k:c[k] for k in ['mode','seed','child','code','success','position_error_m','orientation_error_rad','base_heading_error_rad','base_distance_m','retention_lost','wheel_lift']} for c in r['cases'] if 'code' in c]},indent=2)+'\n')
for mode in ['new','zero_wheels','source_wheels','teacher_wheels','teacher_arm']:
 for parent in [c for c in audit['cases'] if c['controller']=='policy' and c['partition']=='validation']:
  env=ManipulationEnv(cfg)
  try:
   obs,info=env.reset(seed=parent['seed'],options={'lift_goal_delta':parent['goal_delta_rad']});assert not info['initialization_failed'];assert env.grasp_lift_admission['initial_state']==parent['admission']['initial_state']
   case={'mode':mode,'seed':parent['seed'],'child':parent['child'],'admission':env.grasp_lift_admission,'trace':[]};r['cases'].append(case)
   for _ in range(round(cfg.horizon/env.profile.dt)+1):
    action=new(obs);original=action.copy();before=env.control.base_pose().copy();source=old(obs);teacher=env.teacher_action()
    if mode=='zero_wheels':action[:2]=0
    if mode=='source_wheels':action[:2]=source[:2]
    if mode=='teacher_wheels':action[:2]=teacher[:2]
    if mode=='teacher_arm':action[2:-1]=teacher[2:-1]
    obs,reward,done,trunc,info=env.step(action)
    case['trace'].append({'time':info['time'],'new_action':original.tolist(),'source_action':source.tolist(),'teacher_action':teacher.tolist(),'applied_action':action.tolist(),'base':env.control.base_pose().tolist(),'hand':env.control.ee_pose().tolist(),'position_error_m':info['ee_position_error'],'orientation_error_rad':info['ee_orientation_error'],'base_distance_m':info['fixed_base_distance_m'],'base_rotation_rad':info['fixed_base_rotation_rad'],'command_arm_velocity':env.control.velocity.copy().tolist(),'measured_arm_velocity':env.control.velocities().tolist(),'reward':reward})
    if done or trunc:break
   case.update(code=info['code'],success=bool(info['is_success']),position_error_m=info['ee_position_error'],orientation_error_rad=info['ee_orientation_error'],base_heading_error_rad=info['base_heading_error_rad'],base_distance_m=info['fixed_base_distance_m'],retention_lost=bool(info['retention_lost']),wheel_lift=bool(info['wheel_lift']),final_info=info);save();print(mode,parent['seed'],parent['child'],case['code'],case['position_error_m'],flush=True)
  finally:env.close()
assert r['source_hashes']==source_hashes();r['complete']=True;save();print('BASE_ISOLATION_COMPLETE',flush=True)
