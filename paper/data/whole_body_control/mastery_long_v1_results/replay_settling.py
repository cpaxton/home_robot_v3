"""Diagnostic replays of final-policy and teacher settling; no interventions."""
from pathlib import Path
import gzip,json
import numpy as np
from emet.rl.manipulation.mastery import MasteryEnv
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.carry_benchmark import source_hashes
out=Path('logs/eval/mars-mastery-long-settling-v1');out.mkdir(exist_ok=False,parents=True)
policy=ManipulationPolicy('logs/manipulation/mars-mastery-long-v1/seed-90/last')
env=MasteryEnv(stages=['base'],split='train');rows=[]
try:
 for cid in ['train:base:hold','train:base:reverse']:
  for role in ['teacher','policy']:
   obs,info=env.reset(options={'case_id':cid});assert not info['initialization_failed'];trace=[]
   for _ in range(240):
    action=task_pose_teacher_action(env.control) if role=='teacher' else policy(obs)
    obs,reward,done,truncated,info=env.step(action)
    base=env.data.qvel[env.control.base_dof:env.control.base_dof+6].copy();arm=env.control.velocities().copy()
    trace.append({'t':info['time'],'endpoint':bool(env._reference_complete()),'base_qvel':base.tolist(),'arm_qvel':arm.tolist(),'command':env.control.base_command.copy().tolist(),'base_error_m':info['base_goal_error'],'heading_error_rad':info['base_heading_error_rad'],'settled_s':info['settled_seconds'],'linear_speed_fail':bool(np.linalg.norm(base[:3])>=0.005),'angular_speed_fail':bool(np.linalg.norm(base[3:])>=0.02),'arm_speed_fail':bool(max(abs(arm))>=0.03),'pose_fail':bool(info['base_goal_error']>=0.005 or info['base_heading_error_rad']>=0.01)})
    if done or truncated:break
   terminal=[x for x in trace if x['endpoint']]
   summary={key:sum(x[key] for x in terminal) for key in ['linear_speed_fail','angular_speed_fail','arm_speed_fail','pose_fail']}
   summary.update(endpoint_samples=len(terminal),code=info['code'],final_hold_s=info['settled_seconds'],max_command=np.max(np.abs([x['command'] for x in terminal]),axis=0).tolist(),max_linear_axis_speed=np.max(np.abs([x['base_qvel'][:3] for x in terminal]),axis=0).tolist())
   rows.append({'case':cid,'controller':role,'initial_state_sha256':env.mastery_admission['initial_state_sha256'],'summary':summary,'trace':trace})
   print(cid,role,json.dumps(summary),flush=True)
finally:env.close()
report={'scope':__doc__,'source_hashes':source_hashes(),'weights_sha256':policy.manifest['weights_sha256'],'hardware_used':False,'cases':rows}
(out/'report.json.gz').write_bytes(gzip.compress(json.dumps(report,allow_nan=False).encode(),mtime=0))
