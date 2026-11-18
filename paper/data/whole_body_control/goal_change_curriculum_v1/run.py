"""Source-bound goal-change curriculum; fixed final checkpoint, no hardware motion."""
import argparse,hashlib,json,os
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import torch
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.goal_change_tracking import GoalChangeTrackingEnv,load_curriculum
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
REC=ROOT/'logs/recovery/goal-change-curriculum-v1';OUT=ROOT/'logs/manipulation/goal-change-curriculum-v1';RECEIPT=ROOT/'logs/eval/goal-change-regressions-v1/full.receipt.json'
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
def make_goals(cases,cfg=None):
 spec=load_curriculum(ROOT/'configs/benchmarks/mars_goal_changes_v1.json');changes={c['id']:spec['goal_changes'][c['id']] for c in cases};e=GoalChangeTrackingEnv('innate_mars',cases,changes=changes,suite_path=ROOT/'configs/benchmarks/mars_goal_changes_v1.json',reward_config=cfg);e.unwrapped.config=replace(e.unwrapped.config,wheel_feedback_base_only_gains=(0.,.5));return e
def episode(env,case,actor=None):
 obs,initial=env.reset(options={'case_id':case['id']});e=env.unwrapped;identity=arrays_digest({'qpos':e.data.qpos.copy(),'qvel':e.data.qvel.copy()});rewards=np.zeros(3)
 for tick in range(500):
  action=task_pose_teacher_action(e.control) if actor is None else predict(actor,obs)
  obs,_,done,truncated,info=env.step(action);rewards+=info['reward_groups']
  if done or truncated:break
 metrics=e._info();confirmed=stop(e);post_stop=e._info()
 return {'id':case['id'],'seed':case['seed'],'success':bool(info.get('is_success')),'code':info.get('code'),'control_ticks':tick+1,'stop_confirmed':confirmed,'initial_state_sha256':identity,'model_fingerprint':e.fingerprint,'reference_sha256':initial['reference_sha256'],'goal_change':initial.get('goal_change'),'goal_change_delivered':info.get('goal_change_delivered'),'metrics':metrics,'returns':rewards.tolist(),'safety_flags':{k:bool(post_stop[k]) for k in ('wheel_lift','unsafe_contact','retention_lost')}|{'invalid_physics':bool(e._physics_invalid)}}
def evaluate(actor,cases,label,cfg=None):
 env=make_general(cases,cfg);rows=[]
 try:
  for case in cases:
   rows.append(episode(env,case,actor));save(OUT/f'{label}.json',rows);print('EVAL',label,case['id'],rows[-1]['success'],rows[-1]['code'],flush=True)
 finally:env.close()
 return rows
def evaluate_goals(actor,cases,label):
 env=make_goals(cases);rows=[]
 try:
  for case in cases:
   try:row=episode(env,case,actor);row['nominal_admitted']=True
   except ValueError as exc:
    if not str(exc).startswith(('goal_change_reference_rejected:','coupled_reference_rejected:','goal_change_exceeds_episode_horizon','coupled_initialization_failed')):raise
    e=env.unwrapped;confirmed=stop(e);info=e._info();row={'id':case['id'],'seed':case['seed'],'success':False,'code':str(exc),'nominal_admitted':False,'stop_confirmed':confirmed,'safety_flags':{k:bool(info[k]) for k in ('wheel_lift','unsafe_contact','retention_lost')}|{'invalid_physics':bool(e._physics_invalid)}}
   rows.append(row);save(OUT/(label+'.json'),rows);print('GOAL_EVAL',label,case['id'],row['success'],row['code'],flush=True)
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
 OUT.mkdir(parents=True,exist_ok=False);spec=load_curriculum(ROOT/protocol['goal_manifest']);goals=spec['robots']['innate_mars'];report={'protocol_sha256':sha(REC/'protocol.json'),'cases_sha256':sha(REC/'cases.json'),'goal_manifest_sha256':sha(ROOT/protocol['goal_manifest']),'runner_sha256':sha(__file__),'source_state_digest':digest(state),'initializer_weights_sha256':bundle.manifest['weights_sha256'],'runtime_revision':protocol['runtime_revision'],'ready':False,'complete':False,'baseline_complete':False};save(OUT/'admission.json',report)
 teacher=evaluate_goals(None,goals,'teacher-goals');source=evaluate_goals(bundle.actor,goals,'initial-goals')
 report.update(teacher_goal_successes=sum(r['success'] for r in teacher),goal_attempts=len(goals),source_goal_successes=sum(r['success'] for r in source),teacher_goal_unsafe_episodes=sum(any(r['safety_flags'].values()) for r in teacher))
 report['ready']=all(r['nominal_admitted'] and r['stop_confirmed'] and not r['safety_flags']['invalid_physics'] for r in teacher+source)
 # Explicit historical source-policy baseline: same actor, ordinary physics unchanged.
 for cohort,baseline in protocol['reused_baselines'].items():
  path=ROOT/baseline['path']
  if sha(path)!=baseline['sha256']:raise RuntimeError('reused baseline changed '+cohort)
  save(OUT/('initial-'+cohort+'.json'),json.loads(path.read_text()))
 report['baseline_reused_from']='long-finetune-v1/continued';report['baseline_complete']=True;report['complete']=True;save(OUT/'admission.json',report)
 if digest(source_state(ROOT))!=digest(state):raise RuntimeError('source changed during admission')
 if not report['ready']:raise RuntimeError('nominal geometry, stop or physics admission failed; retain every case')
def train_arm(name,protocol,cases,bundle,state):
 receipt=verify_regressions(RECEIPT);admit=json.loads((OUT/'admission.json').read_text())
 if not(admit['ready'] and admit['baseline_complete'] and admit['source_state_digest']==digest(state) and admit['protocol_sha256']==sha(REC/'protocol.json') and admit['cases_sha256']==sha(REC/'cases.json') and admit['runner_sha256']==sha(__file__) and admit['initializer_weights_sha256']==bundle.manifest['weights_sha256']):raise RuntimeError('fresh bound admission required')
 cfg=WholeBodyRewardConfig(hand_pose_weight=protocol['hand_pose_weight'])
 target=OUT/name;target.mkdir(exist_ok=False);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
 policy=MultiCriticPolicy(base_rate_feedback=bundle.actor.base_rate_feedback,base_navigation_feedback=bundle.actor.base_navigation_feedback,quiet_reference_std_scale=protocol['quiet_reference_std_scale'] if name=='quiet_reference_noise' else 1.);policy.actor.load_state_dict(bundle.actor.state_dict())
 with torch.no_grad():policy.log_std.fill_(np.log(protocol['initial_action_std']))
 ppo=PPOConfig(critic_epochs=protocol['independent_critic_epochs'],learning_rate=protocol['actor_learning_rate'][0],critic_learning_rate=protocol['critic_learning_rate'],epochs=protocol['epochs'],batch_size=protocol['batch_size'],clip=protocol['clip'],target_kl=protocol['target_kl'],entropy_weight=protocol['entropy_weight'])
 learner=MultiCriticPPO(policy,ppo)
 if sha(ROOT/protocol['source_training_state'])!=protocol['source_training_state_sha256']:raise RuntimeError('source training checkpoint changed')
 saved=torch.load(ROOT/protocol['source_training_state'],map_location='cpu',weights_only=False)
 if saved['decisions']!=protocol['source_decisions']:raise RuntimeError('source decision count differs')
 expected_actor={k:v.clone() for k,v in bundle.actor.state_dict().items()};policy.load_state_dict(saved['policy'],strict=True);learner.optimizer.load_state_dict(saved['optimizer'])
 for k,v in policy.actor.state_dict().items():torch.testing.assert_close(v,expected_actor[k],rtol=0,atol=0)
 for k,v in policy.state_dict().items():torch.testing.assert_close(v,saved['policy'][k],rtol=0,atol=0)
 # Restored Adam moments/critics, explicit new physical reset/RNG stream.
 selected=[c for c in cases if c['split']=='train'];development,_=load_development()
 spec=load_curriculum(ROOT/protocol['goal_manifest']);goal_train=[c for c in spec['robots']['innate_mars'] if c['split']=='train'];general=ControlStepMixture({'hand':make_general([c for c in selected if c['mode']!='base_only'],cfg),'base':make_general([c for c in selected if c['mode']=='base_only'],cfg),'goal_changes':make_goals(goal_train,cfg)},protocol['general_control_tick_weights'])
 env=ControlStepMixture({'general':general,'tamp':TampPrimitiveEnv(development['train_cases'],primitive='grasp_pose',planning_scope='unloaded_prefix',wheel_feedback_current_velocity=True,wheel_feedback_base_only_gains=(0.,.5),reward_config=cfg)},protocol['family_control_tick_weights'])
 native=general.environments['hand'].unwrapped
 meta={'format':'goal-change-curriculum-training-v1','protocol':protocol,'reward_configuration':cfg.record(),'episode_contract':'finite_tracking_deadline_v1','goal_manifest_sha256':sha(ROOT/protocol['goal_manifest']),'source_cumulative_decisions':protocol['source_cumulative_decisions'],'quiet_reference_std_scale':policy.quiet_reference_std_scale,'ppo':asdict(ppo),'source_state_digest':digest(state),'regression_tests':len(receipt['test_ids']),'regression_receipt_sha256':sha(RECEIPT),'initializer_weights_sha256':bundle.manifest['weights_sha256'],'admission_sha256':sha(OUT/'admission.json'),'runner_sha256':sha(__file__),'initial_policy_state_sha256':policy_state_digest(policy),'complete':False,'phase':'training','decisions':0,'new_ppo_actor_decisions':0,'hardware_used':False,'safety_promoted':False};save(target/'run.json',meta)
 import wandb
 run=wandb.init(project='emet-mars-control',name=f'goal-change-curriculum-v1-{name}',mode='offline',dir=str(target),config=meta,settings=wandb.Settings(disable_git=True,save_code=False))
 meta['wandb_id']=run.id;save(target/'run.json',meta)
 initial={k:v.clone() for k,v in policy.actor.state_dict().items()};decisions=ticks=0
 def on_episode(info):
  row={k:info.get(k) for k in ('case_id','training_family','code','primitive_success','stop_confirmed','wheel_lift','unsafe_contact','retention_lost','invalid_physics','native_terminated','native_truncated','episode_contract','goal_change_delivered')}
  with (target/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 try:
  observation,_=env.reset(seed=protocol['seed']);meta['initial_observation_sha256']=arrays_digest(observation)
  while decisions<protocol['decisions_per_arm']:
   rollout,observation=collect_rollout(env,policy,observation,protocol['rollout_steps'],episode_callback=on_episode,gamma=ppo.gamma)
   if decisions==0:
    meta['first_rollout']={'observations':arrays_digest(rollout['observations']),'actions':arrays_digest({'a':rollout['actions']}),'reward_groups':arrays_digest({'r':rollout['rewards']})};save(target/'run.json',meta)
    meta['source_checkpoint_state_verified_before_update']=True;meta['environment_state_restored']=False;meta['source_decisions']=protocol['source_decisions'];save(target/'run.json',meta)
   warming=decisions<protocol['critic_only_decisions']
   if warming:
    stats=warm(learner,rollout)
    for k,v in policy.actor.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
   else:
    fraction=(decisions-protocol['critic_only_decisions'])/(protocol['decisions_per_arm']-protocol['critic_only_decisions']);a,b=protocol['actor_learning_rate'];learner.optimizer.param_groups[0]['lr']=a+(b-a)*fraction;stats=learner.update(rollout)
   decisions+=len(rollout['actions']);ticks+=int(rollout['ticks'].sum())
   hold=policy.quiet_reference_condition({k:torch.from_numpy(v) for k,v in rollout['observations'].items()});row={'quiet_reference_condition_decisions':int(hold.sum()),'quiet_reference_reduced_decisions':int(hold.sum()) if policy.quiet_reference_std_scale<1 else 0,'action_log_std':policy.log_std.detach().numpy().tolist(),'decisions':decisions,'control_ticks':ticks,'critic_only':warming,'actor_lr':learner.optimizer.param_groups[0]['lr'],'reward_group_means':rollout['rewards'].mean(0).tolist(),'case_ids':sorted(set(rollout['case_ids'].tolist())),'mixture':env.ledger(),'general_mixture':general.ledger(),**stats}
   with (target/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   run.log({'decisions':decisions,'control_ticks':ticks,**{f'reward/{i}':v for i,v in enumerate(row['reward_group_means'])}},step=decisions)
   meta.update(decisions=decisions,control_ticks=ticks,new_ppo_actor_decisions=max(0,decisions-protocol['critic_only_decisions']),mixture=env.ledger(),general_mixture=general.ledger());save(target/'run.json',meta)
   checkpoint=target/'training.pt.partial';torch.save({'policy':policy.state_dict(),'optimizer':learner.optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'decisions':decisions,'run':meta,'environment_state_serialized':False},checkpoint);os.replace(checkpoint,target/'training.pt')
   if decisions%protocol['checkpoint_every_decisions']==0 or decisions==protocol['decisions_per_arm']:
    directory=target/f'checkpoint-{decisions}';final=export_actor(policy.actor,directory/'actor',native.config,native.fingerprint,steps=decisions,training={'scope':'goal_change_curriculum_v1','reward_configuration':cfg.record(),'episode_contract':'finite_tracking_deadline_v1','critic_epochs':ppo.critic_epochs,'quiet_reference_std_scale':policy.quiet_reference_std_scale,'source_weights_sha256':bundle.manifest['weights_sha256'],'teacher_actions_in_loss':False,'source_state_digest':digest(state)});print('PPO',name,decisions,'ticks',ticks,flush=True)
 finally:env.close();run.finish()
 meta['phase']='final_evaluation';save(target/'run.json',meta)
 goals=evaluate_goals(final.actor,load_curriculum(ROOT/protocol['goal_manifest'])['robots']['innate_mars'],name+'-final-goals')
 meta['goal_successes']=sum(r['success'] for r in goals);meta['goal_attempts']=len(goals);goal_splits={c['id']:c['split'] for c in load_curriculum(ROOT/protocol['goal_manifest'])['robots']['innate_mars']};meta['goal_train_successes']=sum(r['success'] for r in goals if goal_splits[r['id']]=='train');meta['goal_audit_successes']=sum(r['success'] for r in goals if goal_splits[r['id']]=='audit')
 results=evaluate(final.actor,cases,f'{name}-final-new-cases',cfg);old=evaluate(final.actor,load_suite(COARSE_SUITE)['robots']['innate_mars'],f'{name}-final-frozen-coupled',cfg)
 tamp=primitive_rollout(development['train_cases']+development['audit_cases'],policy=final.actor,primitive='grasp_pose',planning_scope='unloaded_prefix',wheel_feedback_current_velocity=True,wheel_feedback_base_only_gains=(0.,.5),reward_config=cfg);save(target/'final-tamp.json',tamp)
 observed={cid for line in (target/'updates.jsonl').read_text().splitlines() for cid in json.loads(line)['case_ids']};forbidden={c['id'] for c in load_curriculum(ROOT/protocol['goal_manifest'])['robots']['innate_mars'] if c['split']=='audit'}
 if observed & forbidden:raise RuntimeError('audit case entered PPO')
 meta['goal_audit_ids_excluded_from_ppo']=True
 verify_regressions(RECEIPT)
 if digest(source_state(ROOT))!=digest(state):raise RuntimeError('source changed during training')
 meta.update(phase='complete',complete=True,new_cases_successes=sum(r['success'] for r in results),new_cases_attempts=len(results),development_audit_successes=sum(r['success'] for r,c in zip(results,cases,strict=True) if c['split']=='audit'),development_audit_attempts=12,frozen_coupled_successes=sum(r['success'] for r in old),tamp_successes=sum(r['success'] for r in tamp),final_weights_sha256=final.manifest['weights_sha256']);save(target/'run.json',meta);print('COMPLETE',name,{k:v for k,v in meta.items() if k.endswith('successes')},flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('admission','train'));args=parser.parse_args();torch.set_num_threads(1);protocol=json.loads((REC/'protocol.json').read_text());cases=json.loads((REC/'cases.json').read_text());state=source_state(ROOT);bundle=WholeBodyBundle(ROOT/protocol['initializer'])
 if bundle.manifest['weights_sha256']!=protocol['initializer_weights_sha256']:raise RuntimeError('initializer actor changed')
 if sha(ROOT/protocol['goal_manifest'])!=protocol['goal_manifest_sha256'] or sha(REC/'curriculum.json')!=protocol['goal_manifest_sha256']:raise RuntimeError('curriculum manifest changed')
 if args.stage=='admission':admission(protocol,cases,bundle,state);return
 verify_regressions(RECEIPT)
 train_arm('curriculum',protocol,cases,bundle,state)
 control=json.loads((ROOT/protocol['control_run']).read_text());candidate=json.loads((OUT/'curriculum/run.json').read_text())
 if not candidate['source_checkpoint_state_verified_before_update']:raise RuntimeError('source checkpoint not verified')
 save(OUT/'paired-comparison.json',{'initial_policy_and_first_observation_identical':False,'source_checkpoint_state_verified':True,'environment_state_restored':False,'reward_configuration_identical':True,'control_retrained':False,'arms':{'source':control,'curriculum':candidate},'complete':True})
if __name__=='__main__':main()
