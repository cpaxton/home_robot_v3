"""Completed finite crossed-posture refinement; retain all failures and baselines."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emet.rl.manipulation.mastery import read_report
from emet.rl.regression_gate import verify_regressions

root=Path.cwd();run=root/'logs/manipulation/mars-mastery-crossed-v1-seed90'
out=root/'paper/data/whole_body_control/crossed_posture_v1_results';out.mkdir(exist_ok=True)
receipt=verify_regressions(root/'logs/recovery/mastery-crossed-v1/regressions.receipt.json')
assert json.loads((run/'complete.json').read_text())['steps']==65536
fixed=json.loads((root/'paper/data/whole_body_control/filter_audit_unfiltered_v1/common-smoothness-reference.json').read_text())
points=[];flags=collections.Counter()
for label in ['step-0','step-16384','step-32768','step-49152','last']:
    report=read_report(run/label/'evaluation.json.gz');manifest=json.loads((run/label/'manifest.json').read_text());rows=[]
    for case in report['cases']:
        row={key:case[key] for key in ['id','split','mastered','tracking_and_settling_pass','smoothness_pass','metrics','failure_reasons']}
        command=np.asarray([tick['action'][:2] for tick in case['trace']])*[.12,.5]
        row['raw_command_jerk_rms']=np.sqrt(np.mean((np.diff(command,n=2,axis=0)/.05**2)**2,axis=0)).tolist()
        row['fixed_filtered_smoothness_pass']=all(case['metrics'][key]<=value for key,value in fixed['limits'][case['id']].items())
        rows.append(row)
        for key in ['wheel_lift','unsafe_contact','invalid_physics']:flags[key]+=bool(case['final_info'].get(key))
    points.append({'checkpoint':label,'steps':manifest['steps'],'weights_sha256':manifest['weights_sha256'],'summary':json.loads((run/label/'summary.json').read_text()),'cases':rows})
(out/'checkpoint-cases.json.gz').write_bytes(gzip.compress(json.dumps(points,allow_nan=False).encode(),mtime=0))
for name in ['run.json','complete.json']:shutil.copy2(run/name,out/name)
for name in ['evaluation.json.gz','summary.json','manifest.json']:shutil.copy2(run/'last'/name,out/('final-'+name))
for name in ['metrics.jsonl','updates.jsonl']:(out/(name+'.gz')).write_bytes(gzip.compress((run/name).read_bytes(),mtime=0))
for population in ['final','baseline']:
    source=root/f'logs/eval/mars-mastery-crossed-v1-{population}-expanded'
    for name in ['report.json.gz','summary.json']:shutil.copy2(source/name,out/(population+'-expanded-'+name))
base=read_report(out/'baseline-expanded-report.json.gz');final=read_report(out/'final-expanded-report.json.gz')
assert [c['id'] for c in base['cases']]==[c['id'] for c in final['cases']]
for a,b in zip(base['cases'],final['cases'],strict=True):
    for key in ['initial_state_sha256','reference_sha256']:assert a['admission'][key]==b['admission'][key],a['id']
summary={'steps':65536,'seed':90,'controller_tests':len(receipt['test_ids']),'canonical_full_mastery':[sum(c['mastered'] for c in p['cases']) for p in points],'canonical_tracking_and_holding':[sum(c['tracking_and_settling_pass'] for c in p['cases']) for p in points],'canonical_paired_smoothness':[sum(c['smoothness_pass'] for c in p['cases']) for p in points],'canonical_common_filtered_smoothness':[sum(c['fixed_filtered_smoothness_pass'] for c in p['cases']) for p in points],'checkpoint_safety_flags':dict(flags),'expanded':{},'hardware_used':False,'safety_promoted':False,'further_training_launched':False,'runtime_revision':'f1c8d127'}
for name,report in [('baseline',base),('final',final)]:
    summary['expanded'][name]={'full_mastery':sum(c['mastered'] for c in report['cases']),'tracking_and_holding':sum(c['tracking_and_settling_pass'] for c in report['cases']),'paired_smoothness':sum(c['smoothness_pass'] for c in report['cases']),'cases':len(report['cases']),'safety_flags':{key:sum(bool(c['final_info'].get(key)) for c in report['cases']) for key in flags}}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
x=[p['steps']/1000 for p in points];fig,axes=plt.subplots(3,2,figsize=(10,9),layout='constrained')
for split,color in [('train','tab:blue'),('validation','tab:orange')]:
    for col,key in [(0,'mastered'),(1,'tracking_and_settling_passed')]:axes[0,col].plot(x,[p['summary']['groups']['base/'+split][key] for p in points],label=split,color=color)
axes[0,0].set(title='Full mastery',ylabel='Cases passed (of 5)',ylim=(-.2,5.2));axes[0,1].set(title='Tracking + 2 s quiet',ylabel='Cases passed (of 5)',ylim=(-.2,5.2))
for col,(key,label,units) in enumerate([('base_linear_jerk','Linear','m/s³'),('base_yaw_jerk','Yaw','rad/s³')]):
    axes[1,col].plot(x,[np.median([c['metrics'][key] for c in p['cases']]) for p in points],label='measured')
    axes[1,col].set(title=f'{label}: median peak measured jerk',ylabel=units)
    axes[2,col].plot(x,[np.median([c['raw_command_jerk_rms'][col] for c in p['cases']]) for p in points],label='raw commands')
    axes[2,col].set(title=f'{label}: median RMS raw-command jerk',ylabel=units)
for ax in axes.flat:ax.set_xlabel('Additional steps (thousands)');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.savefig(out/'learning-curves.svg');plt.close(fig)
shutil.copy2(__file__,out/'archive_complete.py')
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name!='files-sha256.json'}
(out/'files-sha256.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
