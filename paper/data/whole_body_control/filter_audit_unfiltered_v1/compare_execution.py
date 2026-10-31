"""Paired TRAIN-case actuator ablation; not a held-out or hardware qualification."""
from pathlib import Path
from copy import deepcopy
import gzip,json
import numpy as np
from emet.rl.manipulation.mastery import load_suite,evaluate_suite,save_report,summarize
from emet.rl.manipulation.policy import ManipulationPolicy
root=Path.cwd();out=root/'logs/eval/mars-mastery-filter-audit-v1';out.mkdir(exist_ok=False,parents=True)
original=load_suite(root/'configs/benchmarks/mars_tracking_mastery_v2.json')
original['cases']=[c for c in original['cases'] if c['stage']=='base' and c['split']=='train']
policy=ManipulationPolicy(root/'logs/manipulation/mars-mastery-objective-v2-seed90/last');records=[]
variants=[('filtered_feedback05',[.24,1.2],.5),('unfiltered_feedback05',None,.5),('unfiltered_feedback0',None,0),('filtered_feedback0',[.24,1.2],0)]
for name,jerk,gain in variants:
 suite=deepcopy(original);suite['config'].update(wheel_jerk_limits=jerk,wheel_feedback_gain=gain)
 folder=out/name;folder.mkdir()
 (folder/'execution-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
 teacher=evaluate_suite(suite,stages=['base']);save_report(folder/'teacher.json.gz',teacher)
 report=evaluate_suite(suite,stages=['base'],policy=policy,teacher_report=teacher);save_report(folder/'policy.json.gz',report)
 row={'variant':name,'jerk_limits':jerk,'feedback_gain':gain,'teacher':summarize(teacher),'policy':summarize(report),'cases':[]}
 for c in report['cases']:
  trace=c['trace'];dt=.05;commands=np.array([x['action'][:2] for x in trace])*[.12,.5];rates=np.diff(commands,axis=0)/dt
  row['cases'].append({'id':c['id'],'mastered':c['mastered'],'metrics':c['metrics'],'failures':c['failure_reasons'],'rms_raw_command_rate':np.sqrt(np.mean(rates**2,axis=0)).tolist(),'peak_raw_command_acceleration':np.max(abs(rates),axis=0).tolist(),'raw_command_delta_sign_changes':np.sum(np.diff(np.sign(rates),axis=0)!=0,axis=0).tolist(),'initial_state_sha256':c['admission']['initial_state_sha256']})
 records.append(row);(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
 print('VARIANT',name,'teacher',sum(c['mastered'] for c in teacher['cases']),'policy',sum(c['mastered'] for c in report['cases']),'smooth',sum(c['smoothness_pass'] for c in report['cases']),flush=True)
(out/'complete.json').write_text(json.dumps({'scope':__doc__,'weights_sha256':policy.manifest['weights_sha256'],'variants':records,'hardware_used':False,'safety_promoted':False},indent=2)+'\n')
