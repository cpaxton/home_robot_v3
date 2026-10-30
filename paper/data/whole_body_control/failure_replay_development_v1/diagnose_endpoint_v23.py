"""Goal-only and heading-free lift ablations on repeated diagnostic seeds; no qualification."""
import gzip,hashlib,json,os
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np,torch
from emet.rl.manipulation.env import ManipulationConfig,ManipulationEnv
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.tracking import TaskPoseTrajectory
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.regression_gate import verify_regressions

torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES']==''
verify_regressions(Path('logs/recovery/20261027/jerk-tracking-v2-regressions.receipt.json'))
root=Path('logs/eval/policy-endpoint-v23-20261028');root.mkdir(exist_ok=False,parents=True)
cfg=replace(ManipulationConfig(**json.load(open('configs/benchmarks/mars_jerk_tracking_curriculum_v1.json'))['environment']),scenario='carry',grasp_lift_probability=1,grasp_lift_phase_weights=None,tracking_precision=4,motion_penalty_scale=1)
actors={n:ManipulationPolicy(f'logs/manipulation/mars-jerk-tracking-v20/{n}/seed-86/last') for n in ['curriculum','control']}
matched=json.loads(gzip.decompress(Path('logs/eval/policy-diagnosis-v22-20261028/report.json.gz').read_bytes()))
report={'scope':__doc__,'config':asdict(cfg),'source_hashes':source_hashes(),'cases':[],'hardware_used':False,'complete':False,'safety_promoted':False,'driver_source':Path(__file__).read_text()}
def save():
 raw=(json.dumps(report,allow_nan=False)+'\n').encode();(root/'report.json.gz').write_bytes(gzip.compress(raw,mtime=0));(root/'summary.json').write_text(json.dumps({'scope':__doc__,'complete':report['complete'],'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'cases':[{k:c[k] for k in ['name','mode','seed','success','code','position_error_m','orientation_error_rad','base_heading_error_rad']} for c in report['cases'] if 'code' in c]},indent=2)+'\n')
for name in ['control','curriculum','teacher']:
 for mode in ['endpoint','endpoint_heading_free']:
  env=ManipulationEnv(cfg)
  try:
   for seed in range(3202000,3202004):
    env.control.set_physical_wheel_slew(False);obs,info=env.reset(seed=seed);assert not info['initialization_failed'];env.control.set_physical_wheel_slew(True)
    admission=env.grasp_lift_admission;initial=admission['initial_state'];assert initial==next(c['initial_state'] for c in matched['cases'] if c['seed']==seed)
    base=env.control.base_goal.copy();goal=np.asarray(admission['goal_hand']);qgoal=np.asarray(admission['goal']);grip=env.control.grip_reference
    env.control.set_reference(qgoal,phase='carry');env.set_task_trajectory(TaskPoseTrajectory([0,env.profile.dt],[base,base],[goal,goal],[grip,grip],base_mask=[True,True,mode=='endpoint']))
    obs=env.control.observe();case={'name':name,'mode':mode,'seed':seed,'initial_state':initial,'goal_hand':goal.tolist(),'trace':[]};report['cases'].append(case)
    for _ in range(round(cfg.horizon/env.profile.dt)+1):
     action=env.teacher_action() if name=='teacher' else actors[name](obs);obs,reward,done,trunc,info=env.step(action)
     case['trace'].append({'time':info['time'],'hand':env.control.ee_pose().tolist(),'base':env.control.base_pose().tolist(),'action':action.tolist(),'reward':reward,'code':info['code'],'position_error_m':info['ee_position_error'],'orientation_error_rad':info['ee_orientation_error'],'base_heading_error_rad':info['base_heading_error_rad'],'settled_seconds':info['settled_seconds']})
     if done or trunc:break
    case.update(success=bool(info['is_success']),code=info['code'],position_error_m=info['ee_position_error'],orientation_error_rad=info['ee_orientation_error'],base_heading_error_rad=info['base_heading_error_rad'],final_info=info);save();print(name,mode,seed,case['code'],round(case['position_error_m']*1000,2),flush=True)
  finally:env.close()
assert report['source_hashes']==source_hashes();report['matched_initial_physics']=True;report['complete']=True;save();print('ENDPOINT_COMPLETE',flush=True)
