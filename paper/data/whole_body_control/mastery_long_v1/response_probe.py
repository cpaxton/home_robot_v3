"""Sim-only matched direct-velocity sign and reward probe, with native guards."""
from pathlib import Path
import json
import numpy as np
from emet.rl.manipulation.mastery import MasteryEnv
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
out=Path('logs/eval/mars-mastery-long-response-v3');out.mkdir(exist_ok=False,parents=True)
env=MasteryEnv(stages=['base'],split='train');rows=[]
try:
 for direction in ['forward','reverse','yaw_left','yaw_right']:
  for mode in ['zero','pulse','teacher']:
   obs,info=env.reset(options={'case_id':'train:base:'+direction})
   assert not info['initialization_failed']
   start=env.control.base_pose().copy();trace=[]
   for i in range(240):
    a=np.zeros(8)
    if mode=='teacher': a=task_pose_teacher_action(env.control)
    elif mode=='pulse':
     index=0 if direction in ['forward','reverse'] else 1
     a[index]=(1 if direction in ['forward','yaw_left'] else -1)*0.03
    obs,r,done,truncated,info=env.step(a)
    pose=env.control.base_pose().copy();delta=pose-start
    yaw=start[2];travel=float(delta[:2]@np.array([np.cos(yaw),np.sin(yaw)]))
    trace.append({'t':info['time'],'command':a.tolist(),'slewed_command':env.control.base_command.copy().tolist(),'measured_velocity':info['mastery']['base_velocity_measured'],'travel_m':travel,'yaw_rad':float(delta[2]),'reward':r,'reward_terms':info['reward_terms'],'base_error_m':info['base_goal_error'],'heading_error_rad':info['base_heading_error_rad'],'code':info['code']})
    if done or truncated or (mode=='pulse' and i>=29):break
   row={'case':direction,'mode':mode,'steps':len(trace),'return':sum(t['reward'] for t in trace),'final':trace[-1],'initial_state_sha256':env.mastery_admission['initial_state_sha256'],'trace':trace}
   rows.append(row)
   print(direction,mode,len(trace),round(trace[-1]['travel_m'],5),round(trace[-1]['yaw_rad'],5),info['code'],flush=True)
finally:env.close()
(out/'report.json').write_text(json.dumps({'scope':__doc__,'hardware_used':False,'source_hashes':source_hashes(),'cases':rows},indent=2)+'\n')
