"""Completed restoring-yaw experiment; architecture intervention and PPO separated."""
from pathlib import Path
import gzip,hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emet.rl.manipulation.mastery import read_report
from emet.rl.regression_gate import verify_regressions
root=Path.cwd();run=root/'logs/manipulation/mars-mastery-restoring-yaw-v1-seed90';out=root/'paper/data/whole_body_control/restoring_yaw_v1_results';out.mkdir(exist_ok=True,parents=True)
receipt=verify_regressions(root/'logs/recovery/restoring-yaw-v1/regressions.receipt.json')
assert json.loads((run/'complete.json').read_text())['steps']==65536
limits=json.loads((root/'paper/data/whole_body_control/filter_audit_unfiltered_v1/common-smoothness-reference.json').read_text())
points=[]
for label in ['step-0','step-16384','step-32768','step-49152','last']:
 report=read_report(run/label/'evaluation.json.gz');manifest=json.loads((run/label/'manifest.json').read_text());rows=[]
 for case in report['cases']:
  row={key:case[key] for key in ['id','split','mastered','tracking_and_settling_pass','smoothness_pass','metrics','failure_reasons']}
  command=np.asarray([tick['action'][:2] for tick in case['trace']])*[.12,.5]
  row['raw_command_jerk_rms']=np.sqrt(np.mean((np.diff(command,n=2,axis=0)/.05**2)**2,axis=0)).tolist()
  row['common_filtered_smoothness_pass']=all(case['metrics'][key]<=value for key,value in limits['limits'][case['id']].items());rows.append(row)
 points.append({'steps':manifest['steps'],'checkpoint':label,'weights_sha256':manifest['weights_sha256'],'cases':rows})
 if label=='step-49152':
  for name in ['evaluation.json.gz','summary.json','manifest.json']:shutil.copy2(run/label/name,out/(label+'-'+name))
(out/'checkpoint-cases.json.gz').write_bytes(gzip.compress(json.dumps(points).encode(),mtime=0))
for name in ['run.json','complete.json']:shutil.copy2(run/name,out/name)
for name in ['evaluation.json.gz','summary.json','manifest.json']:shutil.copy2(run/'last'/name,out/('final-'+name))
for name in ['metrics.jsonl','updates.jsonl']:(out/(name+'.gz')).write_bytes(gzip.compress((run/name).read_bytes(),mtime=0))
for label,path in [('baseline',root/'logs/eval/mars-restoring-yaw-v1-baseline/report.json.gz'),('initial',run/'step-0/admission-evaluation.json.gz'),('final',root/'logs/eval/mars-restoring-yaw-v1-final/report.json.gz')]:shutil.copy2(path,out/(label+'-expanded.json.gz'))
expanded={label:read_report(out/(label+'-expanded.json.gz')) for label in ['baseline','initial','final']}
for label in ['initial','final']:
 for a,b in zip(expanded['baseline']['cases'],expanded[label]['cases'],strict=True):
  assert a['id']==b['id']
  for key in ['initial_state_sha256','reference_sha256']:assert a['admission'][key]==b['admission'][key],a['id']
summary={'steps':65536,'seed':90,'controller_tests':len(receipt['test_ids']),'teacher_passed':50,'matched_initial_states_and_references':True,'canonical_full_mastery':[sum(c['mastered'] for c in p['cases']) for p in points],'canonical_tracking_and_holding':[sum(c['tracking_and_settling_pass'] for c in p['cases']) for p in points],'canonical_common_filtered_smoothness':[sum(c['common_filtered_smoothness_pass'] for c in p['cases']) for p in points],'expanded':{},'hardware_used':False,'weights_uploaded':False,'safety_promoted':False,'further_training_launched':False}
for label,report in expanded.items():
 summary['expanded'][label]={'cases':len(report['cases']),'full_mastery':sum(c['mastered'] for c in report['cases']),'tracking_and_holding':sum(c['tracking_and_settling_pass'] for c in report['cases']),'safety_flags':{key:sum(bool(c['final_info'].get(key)) for c in report['cases']) for key in ['wheel_lift','unsafe_contact','invalid_physics']},'median_peak_jerk':{key:float(np.median([c['metrics'][key] for c in report['cases']])) for key in ['base_linear_jerk','base_yaw_jerk','joint_jerk']}}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
x=[p['steps']/1000 for p in points];fig,axes=plt.subplots(3,2,figsize=(10,9),layout='constrained')
for col,key,title in [(0,'mastered','Full mastery'),(1,'tracking_and_settling_pass','Tracking + 2 s holding')]:
 axes[0,col].plot(x,[sum(c[key] for c in p['cases']) for p in points],marker='o');axes[0,col].set(title=title,ylabel='Canonical cases passed (of 10)',ylim=(-.2,10.2))
for col,(key,name,unit) in enumerate([('base_linear_jerk','Linear','m/s³'),('base_yaw_jerk','Yaw','rad/s³')]):
 axes[1,col].plot(x,[np.median([c['metrics'][key] for c in p['cases']]) for p in points]);axes[1,col].set(title=f'{name}: median peak measured jerk',ylabel=unit)
 axes[2,col].plot(x,[np.median([c['raw_command_jerk_rms'][col] for c in p['cases']]) for p in points]);axes[2,col].set(title=f'{name}: median RMS raw-command jerk',ylabel=unit)
for ax in axes.flat:ax.set_xlabel('PPO steps after architecture intervention (thousands)');ax.grid(alpha=.25)
fig.savefig(out/'learning-curves.svg');plt.close(fig)
shutil.copy2(__file__,out/'archive_complete.py')
(out/'files-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name!='files-sha256.json'},indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
