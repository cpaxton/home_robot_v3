"""Fixed nearby goals on parent-separated development families; no promotion."""
from pathlib import Path
from dataclasses import replace,asdict
import gzip,hashlib,json
import numpy as np,torch
from emet.rl.manipulation.env import ManipulationConfig,ManipulationEnv
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.carry_benchmark import source_hashes

torch.set_num_threads(1)
base=Path('logs/eval/failure-replay-development-v1-20261028');corpus=json.load(open(base/'corpus.json'));cfg=ManipulationConfig(**corpus['config']);out=Path('logs/eval/failure-replay-children-v1-20261028');out.mkdir(exist_ok=False,parents=True)
actor=ManipulationPolicy('logs/manipulation/mars-jerk-tracking-v20/control/seed-86/last')
families=[f for f in corpus['families'] if f['partition']=='validation']+[f for f in corpus['families'] if f['partition']=='train'][:3]
report={'scope':__doc__,'config':asdict(cfg),'source_hashes':source_hashes(),'corpus_sha256':hashlib.sha256((base/'corpus.json').read_bytes()).hexdigest(),'cases':[],'complete':False,'safety_promoted':False,'hardware_used':False,'driver_source':Path(__file__).read_text()}
def save():
 raw=(json.dumps(report,allow_nan=False)+'\n').encode();(out/'report.json.gz').write_bytes(gzip.compress(raw,mtime=0));(out/'summary.json').write_text(json.dumps({'complete':report['complete'],'scope':__doc__,'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'cases':[{k:c[k] for k in ['controller','seed','child','partition','initialization_failed','code','success','ee_position_error','ee_orientation_error','wheel_lift','retention_lost']} for c in report['cases'] if 'code' in c]},indent=2)+'\n')
for family in families:
 for child in range(2):
  delta=np.random.default_rng(family['seed']*16+child).uniform(-corpus['goal_jitter_rad'],corpus['goal_jitter_rad'],5)
  for selector in ['policy','teacher']:
   env=ManipulationEnv(cfg)
   try:
    obs,info=env.reset(seed=family['seed'],options={'lift_goal_delta':delta.tolist()})
    c={'controller':selector,'seed':family['seed'],'child':child,'partition':family['partition'],'goal_delta_rad':delta.tolist(),'initialization_failed':info['initialization_failed'],'admission':env.grasp_lift_admission,'trace':[]};report['cases'].append(c)
    if not c['initialization_failed']:
     for _ in range(round(cfg.horizon/env.profile.dt)+1):
      action=env.teacher_action() if selector=='teacher' else actor(obs);obs,reward,done,trunc,info=env.step(action)
      c['trace'].append({'time':info['time'],'action':action.tolist(),'hand':env.control.ee_pose().tolist(),'base':env.control.base_pose().tolist(),'goal_hand':env.control.hand_goal.copy().tolist(),'position_error_m':info['ee_position_error'],'orientation_error_rad':info['ee_orientation_error'],'base_heading_error_rad':info['base_heading_error_rad'],'reward':reward,'code':info['code']})
      if done or trunc:break
    c.update(code=info.get('code','initialization_failed'),success=bool(info.get('is_success',False)),ee_position_error=info.get('ee_position_error'),ee_orientation_error=info.get('ee_orientation_error'),wheel_lift=info.get('wheel_lift',False),retention_lost=info.get('retention_lost',False),final_info=info);save();print(selector,family['seed'],child,c['code'],flush=True)
   finally:env.close()
  a,b=report['cases'][-2:];assert a['admission']['initial_state']==b['admission']['initial_state'];assert a['admission'].get('goal')==b['admission'].get('goal')
assert source_hashes()==report['source_hashes'];report['complete']=True;save();print('CHILD_AUDIT_COMPLETE',flush=True)
