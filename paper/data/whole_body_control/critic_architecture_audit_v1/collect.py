"""Read-only actor/reward/critic audit on frozen native physics; no policy update."""
import hashlib,json,os
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv,COARSE_SUITE
from emet.rl.manipulation.reward_config import WholeBodyRewardConfig
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_observation import reference_velocity
from emet.rl.manipulation.whole_body_reward import RewardMeter
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.regression_gate import ROOT,source_state,verify_regressions
REC=ROOT/'logs/recovery/critic-architecture-audit-v1';OUT=ROOT/'logs/eval/critic-architecture-audit-v1'
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def features(model,obs):
 with torch.no_grad():
  h,p,m=model.critic_encoder({k:torch.from_numpy(v)[None] for k,v in obs.items()});priv=torch.from_numpy(obs['critic'])[None].clamp(-100,100)
  return torch.cat([h[:,-1],p.flatten(1),m,priv],-1)[0].numpy().copy(),h[0].numpy().copy()
def main():
 torch.set_num_threads(1);verify_regressions(ROOT/'logs/eval/wrapper-reward-regressions-v1/full.receipt.json');bound=source_state(ROOT);OUT.mkdir(parents=True,exist_ok=False)
 cases=json.loads((ROOT/'logs/recovery/wrapper-reward-training-v2/cases.json').read_text());selected=[c for c in cases if (c['split']=='train' and (c['id'].endswith((':hand:0',':hand:1',':hand:3',':hand:4',':base:2')))) or (c['split']=='audit' and c['id'].endswith((':hand:1',':hand:2',':hand:7')))]
 bundle=WholeBodyBundle(ROOT/'logs/eval/scoped-wheel-feedback-v2/candidate/actor');torch.manual_seed(107);model=MultiCriticPolicy(base_rate_feedback=False,base_navigation_feedback=bundle.actor.base_navigation_feedback);model.actor.load_state_dict(bundle.actor.state_dict())
 critics={}
 for name in ('contract','legacy'):
  copy=MultiCriticPolicy(base_rate_feedback=False,base_navigation_feedback=bundle.actor.base_navigation_feedback);copy.load_state_dict(torch.load(ROOT/f'logs/manipulation/wrapper-reward-training-v2/{name}/training.pt',weights_only=True,map_location='cpu')['policy']);critics[name]=copy
 old=WholeBodyRewardConfig();profiles={'legacy':old,'priority_only':replace(old,hand_pose_weight=4),'fine_only':replace(old,position_scale_m=.02,rotation_scale_rad=.2,fine_pose_weight=.75),'progress_only':replace(old,progress_weight=.1),'gating_only':replace(old,pose_gated_velocity=True),'contract':old.contract_tracking()}
 protocol={'source_state_digest':digest(bound),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'actor_weights_sha256':bundle.manifest['weights_sha256'],'profile_records':{k:v.record() for k,v in profiles.items()},'cases':selected,'noise_seeds':[108,109],'gamma':.9975,'actor_updated':False,'scope':'same retained-actor physical traces across reward profiles; trained critics evaluated off-policy; finite-contract MC is not an infinite-horizon value certificate'};save(OUT/'protocol.json',protocol)
 rows=[]
 for case in selected:
  for seed in (108,109):
   env=CoupledTrackingEnv('innate_mars',[case],suite_path=COARSE_SUITE);e=env.unwrapped;e.config=replace(e.config,wheel_feedback_base_only_gains=(0.,.5));rng=np.random.default_rng(seed)
   try:
    obs,init=env.reset();meters={k:RewardMeter(e,base_heading_region=True,config=v) for k,v in profiles.items()};data={k:[] for k in ('features','history','reward_state','diagnostics','predicted_contract','predicted_legacy','next_contract','next_legacy')};rewards={k:[] for k in profiles};states=hashlib.sha256();final={}
    for tick in range(500):
     inp={k:torch.from_numpy(v)[None] for k,v in obs.items()}
     with torch.no_grad():mean=model.actor(inp)[0].numpy();mask=model.actor.action_mask(inp)[0].numpy();pred={k:c.values(inp)[0].numpy().copy() for k,c in critics.items()}
     action=(mean+rng.normal(0,.02,len(mean)))*mask;motion=reference_velocity(e.control);feat,h=features(model,obs);m=meters['legacy'];trajectory=e.control.task_trajectory;remaining=max(0,trajectory.end-e.control.trajectory_elapsed())
     state=np.r_[m.acceleration/np.asarray(e.profile.acceleration),m.base_acceleration/[.5,.5,1.],e.settle_time/2,remaining/e.config.horizon,float(e._reference_complete()),e.steps*e.profile.dt/e.config.horizon]
     for meter in meters.values():meter.before_step()
     data['features'].append(feat);data['history'].append(h);data['reward_state'].append(state);data['diagnostics'].append([float(e.data.time),float(e._info()['ee_position_error']),float(e._info()['ee_orientation_error']),float(e.settle_time),float(np.max(abs(e.control.velocities())))])
     for k in critics:data['predicted_'+k].append(pred[k])
     obs,_,done,truncated,info=env.step(action);states.update(np.r_[e.data.qpos.copy(),e.data.qvel.copy()].tobytes())
     for k,meter in meters.items():
      group=meter.tick(action,motion)
      if done or truncated:group[0]+=meter.terminal_bonus(bool(info.get('is_success')))
      rewards[k].append(group)
     np.testing.assert_array_equal(rewards['legacy'][-1],info['reward_groups'])
     with torch.no_grad():
      inp={k:torch.from_numpy(v)[None] for k,v in obs.items()}
      for k,c in critics.items():data['next_'+k].append(c.values(inp)[0].numpy().copy())
     if done or truncated:break
    arrays={k:np.asarray(v,dtype=np.float32) for k,v in data.items()}
    for k,groups in rewards.items():
     groups=np.asarray(groups,dtype=np.float32);returns=np.zeros_like(groups);following=np.zeros(3)
     for i in reversed(range(len(groups))):following=groups[i]+.9975*following;returns[i]=following
     arrays['rewards_'+k]=groups;arrays['returns_'+k]=returns
    index=len(rows);path=OUT/f'episode-{index:02d}.npz';np.savez_compressed(path,**arrays)
    row={'case':case,'noise_seed':seed,'ticks':tick+1,'terminated':bool(done),'truncated':bool(truncated),'stopped_at_bootstrap':bool(e.control.stopped),'success':bool(info.get('is_success')),'code':info['code'],'state_trace_sha256':states.hexdigest(),'array_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'terminal_values':{k:arrays['next_'+k][-1].tolist() for k in critics},'finite_contract_returns':{k:arrays['returns_'+k][0].tolist() for k in profiles}}
    e.control.stop();start=float(e.data.time)
    while not e._physics_invalid and not whole_body_stopped(e.control) and float(e.data.time)-start<.75-1e-9:e._tick(np.zeros(e.profile.action_size),score=False)
    row['stop_confirmed']=bool(whole_body_stopped(e.control) and not e._physics_invalid);rows.append(row);save(OUT/'episodes.json',rows);print('AUDIT',index,case['id'],seed,row['code'],'truncated',truncated,'bootstrap stopped',row['stopped_at_bootstrap'],'terminal V',[round(row['terminal_values'][k][0],3) for k in critics],flush=True)
   finally:env.close()
 if digest(source_state(ROOT))!=digest(bound):raise RuntimeError('frozen source changed')
 save(OUT/'complete.json',{'complete':True,'episodes':len(rows),'actor_updated':False,'source_state_digest':digest(bound)})
if __name__=='__main__':main()
