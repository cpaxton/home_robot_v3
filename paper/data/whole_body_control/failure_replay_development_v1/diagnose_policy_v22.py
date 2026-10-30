"""Repeated-seed channel/runtime diagnostic, not policy qualification; simulator only."""
import gzip, hashlib, json, math, os
from pathlib import Path
from dataclasses import replace, asdict
import numpy as np, torch
from emet.rl.manipulation.env import ManipulationConfig,ManipulationEnv
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.regression_gate import verify_regressions

torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES']==''
verify_regressions(Path('logs/recovery/20261027/jerk-tracking-v2-regressions.receipt.json'))
root=Path('logs/eval/policy-diagnosis-v22-20261028');root.mkdir(exist_ok=False,parents=True)
cfg=replace(ManipulationConfig(**json.load(open('configs/benchmarks/mars_jerk_tracking_curriculum_v1.json'))['environment']),scenario='carry',grasp_lift_probability=1,grasp_lift_phase_weights=None,tracking_precision=4,motion_penalty_scale=1)
actors={n:ManipulationPolicy(f'logs/manipulation/mars-jerk-tracking-v20/{n}/seed-86/last') for n in ['curriculum','control']}
report={'scope':__doc__,'source_hashes':source_hashes(),'config':asdict(cfg),'cases':[],'hardware_used':False,'complete':False,'safety_promoted':False,'driver_source':Path(__file__).read_text()}
def save():
 raw=(json.dumps(report,allow_nan=False)+'\n').encode();(root/'report.json.gz').write_bytes(gzip.compress(raw,mtime=0))
 summary={'complete':report['complete'],'scope':__doc__,'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'cases':[{k:c[k] for k in ['name','seed','physical_wheel_slew','code','success','position_error_m','orientation_error_rad','base_heading_error_rad','retention_lost','wheel_lift','unsafe_contact','yaw_away_fraction','trace_steps']} for c in report['cases'] if 'code' in c]}
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
def decompose(actor,obs):
 normalized={k:np.clip((obs[k]-mean)/np.sqrt(var+actor.manifest['epsilon']),-actor.manifest['clip'],actor.manifest['clip']).astype(np.float32) for k,(mean,var) in actor.normalization.items()}
 with torch.no_grad():
  p=actor.model.policy;t=p.obs_to_tensor(normalized)[0];features=p.extract_features(t);latent=p.mlp_extractor.forward_actor(features);head=p.action_net;context,errors=latent[:,:-head.feedback_width],latent[:,-head.feedback_width:];g=head.gain(context).reshape(-1,head.motion_size,head.feedback_width);terms=(g*errors[:,None,:]/math.sqrt(head.feedback_width))[0].numpy();e=errors[0].numpy()
 groups={'base_current':slice(0,3),'hand_current':slice(3,12),'base_future':slice(12,15),'hand_future':slice(15,24),'arm_velocity':slice(24,29),'base_velocity':slice(29,35)}
 return {'base_yaw_preactivation_terms':{k:float(terms[1,s].sum()) for k,s in groups.items()},'feedback_errors':e.tolist(),'motion_mean':np.tanh(terms.sum(axis=1)).tolist()}
for name in ['curriculum','control','teacher']:
 for slew in [False,True]:
  env=ManipulationEnv(cfg)
  try:
   for seed in range(3202000,3202004):
    env.control.set_physical_wheel_slew(False)
    obs,info=env.reset(seed=seed);assert not info['initialization_failed']
    env.control.set_physical_wheel_slew(slew);assert env.control.physical_wheel_slew_enabled==slew
    case={'name':name,'seed':seed,'physical_wheel_slew':slew,'initial_state':env.grasp_lift_admission['initial_state'],'trace':[]};report['cases'].append(case);away=[]
    for step in range(round(cfg.horizon/env.profile.dt)+1):
     actor=actors.get(name);action=actor(obs) if actor else env.teacher_action();decomp=decompose(actor,obs) if actor and step%10==0 else None
     obs,reward,done,truncated,info=env.step(action);heading=info['base_heading_error_rad'];c=env.control
     signed_error=np.arctan2(np.sin(c.base_goal[2]-c.base_pose()[2]),np.cos(c.base_goal[2]-c.base_pose()[2]))
     if abs(signed_error)>.005 and abs(action[1])>1e-6:away.append(bool(action[1]*signed_error<0))
     case['trace'].append({'step':step,'time':info['time'],'action':action.tolist(),'base_command':c.base_command.copy().tolist(),'applied_base_velocity':c.applied_base_velocity.copy().tolist(),'base':c.base_pose().tolist(),'hand':c.ee_pose().tolist(),'hand_goal':c.hand_goal.copy().tolist(),'ee_position_error':info['ee_position_error'],'ee_orientation_error':info['ee_orientation_error'],'base_heading_error':heading,'arm_velocity':c.velocities().tolist(),'command_arm_velocity':c.velocity.copy().tolist(),'decomposition':decomp,'reward':reward,'code':info['code']})
     if done or truncated:break
    case.update(code=info['code'],success=bool(info['is_success']),position_error_m=info['ee_position_error'],orientation_error_rad=info['ee_orientation_error'],base_heading_error_rad=info['base_heading_error_rad'],retention_lost=bool(info['retention_lost']),wheel_lift=bool(info['wheel_lift']),unsafe_contact=bool(info['unsafe_contact']),yaw_away_fraction=float(np.mean(away)) if away else None,trace_steps=len(case['trace']),final_info=info);save();print(name,slew,seed,case['code'],round(case['position_error_m']*1000,2),case['yaw_away_fraction'],flush=True)
  finally:env.close()
for seed in range(3202000,3202004):
 rows=[c for c in report['cases'] if c['seed']==seed];assert all(c['initial_state']==rows[0]['initial_state'] for c in rows)
assert report['source_hashes']==source_hashes();report['complete']=True;save();print('DIAGNOSIS_COMPLETE',flush=True)
