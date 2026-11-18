"""Source-bound paired reward pilot; fixed final checkpoints, no hardware motion."""
import argparse,hashlib,json,os
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv,COARSE_SUITE,load_suite
from emet.rl.manipulation.coupled_evaluation import json_value
from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO,PPOConfig,grouped_gae
from emet.rl.manipulation.reward_config import WholeBodyRewardConfig
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle,export_actor
from emet.rl.manipulation.whole_body_env import ControlStepMixture
from emet.rl.manipulation.whole_body_training import collect_rollout,load_development,primitive_rollout,arrays_digest,policy_state_digest
from emet.rl.regression_gate import ROOT,source_state,verify_regressions
REC=ROOT/'logs/recovery/wrapper-reward-training-v2';OUT=ROOT/'logs/manipulation/wrapper-reward-training-v2';RECEIPT=ROOT/'logs/eval/wrapper-reward-regressions-v1/full.receipt.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,default=json_value,allow_nan=False)+'\n');os.replace(tmp,p)
def predict(actor,obs):
 with torch.no_grad():return actor({k:torch.from_numpy(np.asarray(obs[k],dtype=np.float32))[None] for k in ('history','preview','motion')})[0].numpy().copy()
def stop(e):
 e.control.stop();start=float(e.data.time)
 while not e._physics_invalid and not whole_body_stopped(e.control) and float(e.data.time)-start<.75-1e-9:e._tick(np.zeros(e.profile.action_size),score=False)
 return bool(not e._physics_invalid and whole_body_stopped(e.control))
def make_general(cases,cfg):
 e=CoupledTrackingEnv('innate_mars',cases,suite_path=COARSE_SUITE,reward_config=cfg)
 e.unwrapped.config=replace(e.unwrapped.config,wheel_feedback_base_only_gains=(0.,.5))
 return e
def episode(env,case,actor=None):
 obs,initial=env.reset(options={'case_id':case['id']});e=env.unwrapped;identity=arrays_digest({'qpos':e.data.qpos.copy(),'qvel':e.data.qvel.copy()});rewards=np.zeros(3)
 for tick in range(500):
  action=task_pose_teacher_action(e.control) if actor is None else predict(actor,obs)
  obs,_,done,truncated,info=env.step(action);rewards+=info['reward_groups']
  if done or truncated:break
 metrics=e._info();confirmed=stop(e);post_stop=e._info()
 return {'id':case['id'],'seed':case['seed'],'success':bool(info.get('is_success')),'code':info.get('code'),'control_ticks':tick+1,'stop_confirmed':confirmed,'initial_state_sha256':identity,'model_fingerprint':e.fingerprint,'reference_sha256':initial['reference_sha256'],'metrics':metrics,'returns':rewards.tolist(),'safety_flags':{k:bool(post_stop[k]) for k in ('wheel_lift','unsafe_contact','retention_lost')}|{'invalid_physics':bool(e._physics_invalid)}}
def evaluate(actor,cases,label,cfg=None):
 env=make_general(cases,cfg);rows=[]
 try:
  for case in cases:
   rows.append(episode(env,case,actor));save(OUT/f'{label}.json',rows);print('EVAL',label,case['id'],rows[-1]['success'],rows[-1]['code'],flush=True)
 finally:env.close()
 return rows
def warm(learner,rollout):
 cfg=learner.config;p=learner.policy
 _,targets=grouped_gae(rollout['rewards'],rollout['values'],rollout['next_values'],rollout['terminated'],rollout['episode_end'],rollout['ticks'],gamma=cfg.gamma,lam=cfg.gae_lambda)
 obs={k:torch.as_tensor(v,dtype=torch.float32) for k,v in rollout['observations'].items()};targets=torch.from_numpy(targets)
 for _ in range(cfg.epochs):
  for indices in torch.randperm(len(targets)).split(cfg.batch_size):
   loss=.5*((p.values({k:v[indices] for k,v in obs.items()})-targets[indices])**2).mean(0).sum();learner.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.critics.parameters(),cfg.max_grad_norm);learner.optimizer.step()
 return {'critic_only':True,'value_loss':float(loss.detach())}
def admission(protocol,cases,bundle,state):
 OUT.mkdir(parents=True,exist_ok=False)
 report={'protocol_sha256':sha(REC/'protocol.json'),'cases_sha256':sha(REC/'cases.json'),'runner_sha256':sha(__file__),'source_state_digest':digest(state),'initializer_weights_sha256':bundle.manifest['weights_sha256'],'runtime_revision':protocol['runtime_revision'],'ready':False,'complete':False}
 save(OUT/'admission.json',report)
 teacher=evaluate(None,cases,'teacher')
 training=[r for r,c in zip(teacher,cases,strict=True) if c['split']=='train']
 development,_=load_development()
 tamp=primitive_rollout(development['train_cases']+development['audit_cases'],primitive='grasp_pose',planning_scope='unloaded_prefix',wheel_feedback_current_velocity=True,wheel_feedback_base_only_gains=(0.,.5),reward_config=WholeBodyRewardConfig.contract_tracking());save(OUT/'teacher-tamp.json',tamp)
 report.update(teacher_successes=sum(r['success'] for r in teacher),teacher_attempts=len(teacher),training_teacher_successes=sum(r['success'] for r in training),training_teacher_attempts=len(training),teacher_tamp_successes=sum(r['success'] for r in tamp),teacher_tamp_attempts=len(tamp))
 report['ready']=all(r['success'] and r['stop_confirmed'] and not any(r['safety_flags'].values()) for r in training+tamp)
 report['complete']=True;save(OUT/'admission.json',report)
 if not report['ready']:raise RuntimeError('declared teacher admission failed; preserve all cases and do not train')
 baseline=evaluate(bundle.actor,cases,'initial-new-cases')
 old=evaluate(bundle.actor,load_suite(COARSE_SUITE)['robots']['innate_mars'],'initial-frozen-coupled')
 tamp=primitive_rollout(development['train_cases']+development['audit_cases'],policy=bundle.actor,primitive='grasp_pose',planning_scope='unloaded_prefix',wheel_feedback_current_velocity=True,wheel_feedback_base_only_gains=(0.,.5));save(OUT/'initial-tamp.json',tamp)
 report.update(initial_successes=sum(r['success'] for r in baseline),initial_attempts=len(baseline),initial_frozen_successes=sum(r['success'] for r in old),initial_tamp_successes=sum(r['success'] for r in tamp),baseline_complete=True);save(OUT/'admission.json',report)
 if digest(source_state(ROOT))!=digest(state):raise RuntimeError('source changed during admission')
def train_arm(name,protocol,cases,bundle,state):
 receipt=verify_regressions(RECEIPT);admit=json.loads((OUT/'admission.json').read_text())
 if not(admit['ready'] and admit['baseline_complete'] and admit['source_state_digest']==digest(state) and admit['protocol_sha256']==sha(REC/'protocol.json') and admit['cases_sha256']==sha(REC/'cases.json') and admit['runner_sha256']==sha(__file__) and admit['initializer_weights_sha256']==bundle.manifest['weights_sha256']):raise RuntimeError('fresh bound admission required')
 cfg=WholeBodyRewardConfig.contract_tracking() if name=='contract' else WholeBodyRewardConfig()
 target=OUT/name;target.mkdir(exist_ok=False);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
 policy=MultiCriticPolicy(base_rate_feedback=bundle.actor.base_rate_feedback,base_navigation_feedback=bundle.actor.base_navigation_feedback);policy.actor.load_state_dict(bundle.actor.state_dict())
 with torch.no_grad():policy.log_std.fill_(np.log(protocol['initial_action_std']))
 ppo=PPOConfig(learning_rate=protocol['actor_learning_rate'][0],critic_learning_rate=protocol['critic_learning_rate'],epochs=protocol['epochs'],batch_size=protocol['batch_size'],clip=protocol['clip'],target_kl=protocol['target_kl'],entropy_weight=protocol['entropy_weight'])
 learner=MultiCriticPPO(policy,ppo);selected=[c for c in cases if c['split']=='train'];development,_=load_development()
 general=ControlStepMixture({'hand':make_general([c for c in selected if c['mode']!='base_only'],cfg),'base':make_general([c for c in selected if c['mode']=='base_only'],cfg)},protocol['general_hand_base_tick_weights'])
 env=ControlStepMixture({'general':general,'tamp':TampPrimitiveEnv(development['train_cases'],primitive='grasp_pose',planning_scope='unloaded_prefix',wheel_feedback_current_velocity=True,wheel_feedback_base_only_gains=(0.,.5),reward_config=cfg)},protocol['family_control_tick_weights'])
 native=general.environments['hand'].unwrapped
 meta={'format':'wrapper-reward-training-arm-v1','protocol':protocol,'reward_configuration':cfg.record(),'source_state_digest':digest(state),'regression_tests':len(receipt['test_ids']),'regression_receipt_sha256':sha(RECEIPT),'initializer_weights_sha256':bundle.manifest['weights_sha256'],'admission_sha256':sha(OUT/'admission.json'),'runner_sha256':sha(__file__),'initial_policy_state_sha256':policy_state_digest(policy),'complete':False,'phase':'training','decisions':0,'new_ppo_actor_decisions':0,'hardware_used':False,'safety_promoted':False};save(target/'run.json',meta)
 import wandb
 run=wandb.init(project='emet-mars-control',name=f'wrapper-reward-v1-{name}',mode='offline',dir=str(target),config=meta,settings=wandb.Settings(disable_git=True,save_code=False))
 meta['wandb_id']=run.id;save(target/'run.json',meta)
 initial={k:v.clone() for k,v in policy.actor.state_dict().items()};decisions=ticks=0
 def on_episode(info):
  row={k:info.get(k) for k in ('case_id','training_family','code','primitive_success','stop_confirmed','wheel_lift','unsafe_contact','retention_lost','invalid_physics')}
  with (target/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 try:
  observation,_=env.reset(seed=protocol['seed']);meta['initial_observation_sha256']=arrays_digest(observation)
  while decisions<protocol['decisions_per_arm']:
   rollout,observation=collect_rollout(env,policy,observation,protocol['rollout_steps'],episode_callback=on_episode,gamma=ppo.gamma)
   if decisions==0:
    meta['first_rollout']={'observations':arrays_digest(rollout['observations']),'actions':arrays_digest({'a':rollout['actions']}),'reward_groups':arrays_digest({'r':rollout['rewards']})};save(target/'run.json',meta)
   warming=decisions<protocol['critic_only_decisions']
   if warming:
    stats=warm(learner,rollout)
    for k,v in policy.actor.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
   else:
    fraction=(decisions-protocol['critic_only_decisions'])/(protocol['decisions_per_arm']-protocol['critic_only_decisions']);a,b=protocol['actor_learning_rate'];learner.optimizer.param_groups[0]['lr']=a+(b-a)*fraction;stats=learner.update(rollout)
   decisions+=len(rollout['actions']);ticks+=int(rollout['ticks'].sum())
   row={'decisions':decisions,'control_ticks':ticks,'critic_only':warming,'actor_lr':learner.optimizer.param_groups[0]['lr'],'reward_group_means':rollout['rewards'].mean(0).tolist(),'case_ids':sorted(set(rollout['case_ids'].tolist())),'mixture':env.ledger(),'general_mixture':general.ledger(),**stats}
   with (target/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   run.log({'decisions':decisions,'control_ticks':ticks,**{f'reward/{i}':v for i,v in enumerate(row['reward_group_means'])}},step=decisions)
   meta.update(decisions=decisions,control_ticks=ticks,new_ppo_actor_decisions=max(0,decisions-protocol['critic_only_decisions']),mixture=env.ledger(),general_mixture=general.ledger());save(target/'run.json',meta)
   checkpoint=target/'training.pt.partial';torch.save({'policy':policy.state_dict(),'optimizer':learner.optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'decisions':decisions,'run':meta,'environment_state_serialized':False},checkpoint);os.replace(checkpoint,target/'training.pt')
   if decisions%2048==0 or decisions==protocol['decisions_per_arm']:
    directory=target/f'checkpoint-{decisions}';final=export_actor(policy.actor,directory/'actor',native.config,native.fingerprint,steps=decisions,training={'scope':'wrapper_reward_pilot_v1','reward_configuration':cfg.record(),'source_weights_sha256':bundle.manifest['weights_sha256'],'teacher_actions_in_loss':False,'source_state_digest':digest(state)});print('PPO',name,decisions,'ticks',ticks,flush=True)
 finally:env.close();run.finish()
 meta['phase']='final_evaluation';save(target/'run.json',meta)
 results=evaluate(final.actor,cases,f'{name}-final-new-cases',cfg);old=evaluate(final.actor,load_suite(COARSE_SUITE)['robots']['innate_mars'],f'{name}-final-frozen-coupled',cfg)
 tamp=primitive_rollout(development['train_cases']+development['audit_cases'],policy=final.actor,primitive='grasp_pose',planning_scope='unloaded_prefix',wheel_feedback_current_velocity=True,wheel_feedback_base_only_gains=(0.,.5),reward_config=cfg);save(target/'final-tamp.json',tamp)
 verify_regressions(RECEIPT)
 if digest(source_state(ROOT))!=digest(state):raise RuntimeError('source changed during training')
 meta.update(phase='complete',complete=True,new_cases_successes=sum(r['success'] for r in results),new_cases_attempts=len(results),fresh_audit_successes=sum(r['success'] for r,c in zip(results,cases,strict=True) if c['split']=='audit'),fresh_audit_attempts=12,frozen_coupled_successes=sum(r['success'] for r in old),tamp_successes=sum(r['success'] for r in tamp),final_weights_sha256=final.manifest['weights_sha256']);save(target/'run.json',meta);print('COMPLETE',name,{k:v for k,v in meta.items() if k.endswith('successes')},flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['admission','train']);a=p.parse_args();torch.set_num_threads(1)
 protocol=json.loads((REC/'protocol.json').read_text());cases=json.loads((REC/'cases.json').read_text());state=source_state(ROOT);bundle=WholeBodyBundle(ROOT/protocol['initializer'])
 if a.stage=='admission':admission(protocol,cases,bundle,state)
 else:
  for arm in protocol['reward_arms']:train_arm(arm,protocol,cases,bundle,state)
  first,second=[json.loads((OUT/arm/'run.json').read_text()) for arm in protocol['reward_arms']]
  keys=['initial_policy_state_sha256','initial_observation_sha256']
  if any(first[k]!=second[k] for k in keys) or any(first['first_rollout'][k]!=second['first_rollout'][k] for k in ('observations','actions')):raise RuntimeError('reward comparison initial state/first action rollout not paired')
  save(OUT/'paired-comparison.json',{'initial_state_and_first_actions_identical':True,'reward_difference_explicit':True,'arms':{arm:json.loads((OUT/arm/'run.json').read_text()) for arm in protocol['reward_arms']},'complete':True})
if __name__=='__main__':main()
