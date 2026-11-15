"""Freeze the first crossed-population checkpoint, including regressions."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from emet.rl.manipulation.mastery import read_report

root=Path.cwd()
run=root/'logs/manipulation/mars-mastery-crossed-v1-seed90'
out=root/'paper/data/whole_body_control/crossed_posture_v1_setup'
limits=json.loads((root/'paper/data/whole_body_control/filter_audit_unfiltered_v1/common-smoothness-reference.json').read_text())
report=read_report(run/'step-16384/evaluation.json.gz')
rows=[]
for case in report['cases']:
    row={k:case[k] for k in ['id','split','mastered','tracking_and_settling_pass','smoothness_pass','metrics','failure_reasons']}
    command=np.asarray([tick['action'][:2] for tick in case['trace']])*[.12,.5]
    row['raw_command_jerk_rms']=np.sqrt(np.mean((np.diff(command,n=2,axis=0)/.05**2)**2,axis=0)).tolist()
    row['fixed_filtered_smoothness_pass']=all(case['metrics'][key]<=value for key,value in limits['limits'][case['id']].items())
    rows.append(row)
metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
metrics=[row for row in metrics if row['timesteps']<=16384]
visits=metrics[-1]['sampling/episode_starts']
summary={
    'steps':16384,'canonical_cases':10,'mastered':sum(c['mastered'] for c in rows),
    'tracking_and_holding':sum(c['tracking_and_settling_pass'] for c in rows),
    'paired_smoothness':sum(c['smoothness_pass'] for c in rows),
    'fixed_filtered_smoothness':sum(c['fixed_filtered_smoothness_pass'] for c in rows),
    'safety_flags':{key:sum(bool(c['final_info'].get(key)) for c in report['cases']) for key in ['wheel_lift','unsafe_contact','invalid_physics']},
    'training_pairs_visited':len(visits),'min_episode_starts':min(visits.values()),'max_episode_starts':max(visits.values()),
    'cases':rows,'hardware_used':False,'safety_promoted':False,
    'interpretation':'Early full-mastery regression from5/10 to3/10; tracking9/10 retained; no demonstrated benefit yet; bounded run continues unchanged',
}
(out/'step-16384-cases.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ['evaluation.json.gz','summary.json','manifest.json']:
    shutil.copy2(run/'step-16384'/name,out/('step-16384-'+name))
(out/'metrics-through-16384.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(row) for row in metrics)+'\n').encode(),mtime=0))
shutil.copy2(__file__,out/'archive_interim.py')
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name!='files-sha256.json'}
(out/'files-sha256.json').write_text(json.dumps(checks,indent=2)+'\n')
print({key:value for key,value in summary.items() if key!='cases'},flush=True)
