"""Completed unfiltered comparison and crossed-posture diagnostic; no promotion."""
from pathlib import Path
import gzip,hashlib,json,shutil,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emet.rl.manipulation.mastery import read_report
from emet.rl.regression_gate import verify_regressions
root=Path.cwd();local=root/'logs/manipulation/mars-mastery-unfiltered-v1-seed90';out=root/'paper/data/whole_body_control/unfiltered_v1_results';out.mkdir(exist_ok=False)
receipt=verify_regressions(root/'logs/recovery/mastery-filter-audit/regressions.receipt.json');completion=json.loads((local/'complete.json').read_text());assert completion['steps']==65536
limits=json.loads((root/'paper/data/whole_body_control/filter_audit_unfiltered_v1/common-smoothness-reference.json').read_text())
points=[];trace_hashes={};flags=collections.Counter()
for p in sorted(local.glob('step-*'),key=lambda p:int(p.name.split('-')[1]))+[local/'last']:
 report=read_report(p/'evaluation.json.gz');manifest=json.loads((p/'manifest.json').read_text());rows=[]
 for c in report['cases']:
  cmd=np.array([t['action'][:2] for t in c['trace']])*[.12,.5];accel=np.diff(cmd,axis=0)/.05;jerk=np.diff(accel,axis=0)/.05
  row={k:c[k] for k in ['id','split','mastered','tracking_and_settling_pass','smoothness_pass','metrics','failure_reasons']}
  row['raw_command_acceleration_rms']=np.sqrt(np.mean(accel**2,axis=0)).tolist();row['raw_command_jerk_rms']=np.sqrt(np.mean(jerk**2,axis=0)).tolist();row['fixed_filtered_smoothness_pass']=all(c['metrics'][k]<=v for k,v in limits['limits'][c['id']].items());rows.append(row)
  for key in ['wheel_lift','unsafe_contact','invalid_physics']:flags[key]+=bool(c['final_info'].get(key))
 points.append({'checkpoint':p.name,'steps':manifest['steps'],'weights_sha256':manifest['weights_sha256'],'summary':json.loads((p/'summary.json').read_text()),'fixed_filtered_smoothness_passes':sum(c['fixed_filtered_smoothness_pass'] for c in rows),'cases':rows})
 trace_hashes[str((p/'evaluation.json.gz').relative_to(root))]=hashlib.sha256((p/'evaluation.json.gz').read_bytes()).hexdigest()
(out/'checkpoint-cases.json.gz').write_bytes(gzip.compress(json.dumps(points,allow_nan=False).encode(),mtime=0))
for name in ['complete.json','run.json']:shutil.copy2(local/name,out/name)
for name in ['evaluation.json.gz','summary.json','manifest.json']:shutil.copy2(local/'last'/name,out/('final-'+name))
for name in ['metrics.jsonl','updates.jsonl']:(out/(name+'.gz')).write_bytes(gzip.compress((local/name).read_bytes(),mtime=0))
for name in ['teacher.json.gz','policy.json.gz','summary.json']:shutil.copy2(root/'logs/eval/mars-mastery-left-turn-posture-probe-v1'/name,out/('posture-probe-'+name))
shutil.copy2(root/'logs/recovery/mastery-unfiltered-continuation/posture_probe.py',out/'posture_probe.py')
shutil.copy2(root/'paper/data/whole_body_control/filter_audit_unfiltered_v1/common-smoothness-reference.json',out/'common-smoothness-reference.json')
initial,final=points[0]['cases'],points[-1]['cases'];med=lambda rows,key:np.median([c['metrics'][key] for c in rows])
summary={'scope':__doc__,'source_revision':'f4a4f08e','steps':65536,'seed':90,'cases':10,'final_full_mastery':5,'final_tracking_and_holding':9,'final_paired_smoothness':6,'fixed_filtered_smoothness':points[-1]['fixed_filtered_smoothness_passes'],'controller_tests':len(receipt['test_ids']),'median_peak_linear_jerk_initial':med(initial,'base_linear_jerk'),'median_peak_linear_jerk_final':med(final,'base_linear_jerk'),'median_raw_linear_command_jerk_initial':np.median([c['raw_command_jerk_rms'][0] for c in initial]),'median_raw_linear_command_jerk_final':np.median([c['raw_command_jerk_rms'][0] for c in final]),'checkpoint_safety_flags':dict(flags),'local_trace_sha256':trace_hashes,'next_training_launched':False,'hardware_used':False,'safety_promoted':False,'weights_uploaded':False}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
x=[p['steps']/1000 for p in points];fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
for split,color in [('train','tab:blue'),('validation','tab:orange')]:
 axes[0,0].plot(x,[p['summary']['groups']['base/'+split]['mastered'] for p in points],label=split,color=color)
 axes[0,1].plot(x,[p['summary']['groups']['base/'+split]['tracking_and_settling_passed'] for p in points],label=split,color=color)
for key,label in [('base_linear_jerk','linear (m/s^3)'),('base_yaw_jerk','yaw (rad/s^3)')]:axes[1,0].plot(x,[med(p['cases'],key) for p in points],label=label)
for i,label in enumerate(['raw linear (m/s^3)','raw yaw (rad/s^3)']):axes[1,1].plot(x,[np.median([c['raw_command_jerk_rms'][i] for c in p['cases']]) for p in points],label=label)
axes[0,0].set(title='Full mastery, paired teacher limits',ylabel='Passed cases (of 5)',ylim=(-.2,5.2));axes[0,1].set(title='Tracking and two-second holding',ylabel='Passed cases (of 5)',ylim=(-.2,5.2));axes[1,0].set(title='Median peak measured base jerk',ylabel='Physical units, axes shown in legend');axes[1,1].set(title='Median RMS raw-command jerk',ylabel='Physical units, axes shown in legend')
for a in axes.flat:a.set_xlabel('Additional training steps (thousands)');a.legend(fontsize=8);a.grid(alpha=.2)
fig.savefig(out/'learning-curves.svg');plt.close(fig)
(out/'files-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n');print({k:summary[k] for k in ['final_full_mastery','final_tracking_and_holding','final_paired_smoothness','fixed_filtered_smoothness','checkpoint_safety_flags']},flush=True)
