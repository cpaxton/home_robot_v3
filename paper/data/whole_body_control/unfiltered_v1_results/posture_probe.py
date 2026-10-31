"""Diagnostic crossed train postures and yaw signs; no held-out tuning or training."""
from pathlib import Path
from copy import deepcopy
import json
from emet.rl.manipulation.mastery import load_suite,evaluate_suite,save_report
from emet.rl.manipulation.policy import ManipulationPolicy
root=Path.cwd();suite=load_suite(root/'configs/benchmarks/mars_tracking_mastery_unfiltered_v1.json');source={c['id']:c for c in suite['cases']}
cases=[]
for posture in ['yaw_left','yaw_right','hold']:
 for sign in [-1,1]:
  c=deepcopy(source['train:base:'+posture]);c.update(id=f'train:base:probe_{posture}_{sign}',travel_m=0,turn_rad=sign*.04);cases.append(c)
suite['cases']=cases
out=root/'logs/eval/mars-mastery-left-turn-posture-probe-v1';out.mkdir(exist_ok=False,parents=True)
policy=ManipulationPolicy(root/'logs/manipulation/mars-mastery-unfiltered-v1-seed90/last')
teacher=evaluate_suite(suite,stages=['base']);save_report(out/'teacher.json.gz',teacher)
r=evaluate_suite(suite,stages=['base'],policy=policy,teacher_report=teacher);save_report(out/'policy.json.gz',r)
rows=[]
for c in r['cases']:
 row={'id':c['id'],'seed':c['reset_seed'],'turn_rad':c['turn_rad'],'mastered':c['mastered'],'tracking_and_settling':c['tracking_and_settling_pass'],'metrics':c['metrics'],'failures':c['failure_reasons'],'initial_state_sha256':c['admission']['initial_state_sha256']};rows.append(row)
 print(row['id'],'track',row['tracking_and_settling'],'yaw',row['metrics']['base_heading_error_rad'],'fails',row['failures'],flush=True)
(out/'summary.json').write_text(json.dumps({'scope':__doc__,'teacher_passed':sum(c['mastered'] for c in teacher['cases']),'cases':rows,'weights_sha256':policy.manifest['weights_sha256'],'hardware_used':False},indent=2)+'\n')
