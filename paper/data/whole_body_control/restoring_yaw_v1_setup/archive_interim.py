"""Retain all available scheduled post-update results; no validation selection."""
from pathlib import Path
import gzip,hashlib,json,shutil
import numpy as np
root=Path.cwd();run=root/'logs/manipulation/mars-mastery-restoring-yaw-v1-seed90';out=root/'paper/data/whole_body_control/restoring_yaw_v1_setup'
limits=json.loads((root/'paper/data/whole_body_control/filter_audit_unfiltered_v1/common-smoothness-reference.json').read_text())
points=[]
for step in [0,16384,32768]:
 path=run/f'step-{step}'
 with gzip.open(path/'evaluation.json.gz','rt') as f:report=json.load(f)
 rows=[]
 for case in report['cases']:
  row={key:case[key] for key in ['id','split','mastered','tracking_and_settling_pass','smoothness_pass','metrics','failure_reasons']}
  command=np.asarray([tick['action'][:2] for tick in case['trace']])*[.12,.5]
  row['raw_command_jerk_rms']=np.sqrt(np.mean((np.diff(command,n=2,axis=0)/.05**2)**2,axis=0)).tolist()
  row['common_filtered_smoothness_pass']=all(case['metrics'][key]<=value for key,value in limits['limits'][case['id']].items())
  rows.append(row)
 point={'steps':step,'cases':rows,'summary':json.loads((path/'summary.json').read_text())};points.append(point)
 if step:
  for name in ['evaluation.json.gz','manifest.json','summary.json']:shutil.copy2(path/name,out/(f'step-{step}-'+name))
(out/'checkpoint-cases-through-32768.json.gz').write_bytes(gzip.compress(json.dumps(points).encode(),mtime=0))
for name in ['metrics.jsonl','updates.jsonl']:
 rows=[json.loads(line) for line in (run/name).read_text().splitlines()];rows=[row for row in rows if row['timesteps']<=32768]
 (out/(name+'-through-32768.gz')).write_bytes(gzip.compress(('\n'.join(json.dumps(row) for row in rows)+'\n').encode(),mtime=0))
shutil.copy2(__file__,out/'archive_interim.py')
(out/'files-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name!='files-sha256.json'},indent=2)+'\n')
print('Archived0/16k/32k metrics, full post-update traces and optimizer streams',flush=True)
