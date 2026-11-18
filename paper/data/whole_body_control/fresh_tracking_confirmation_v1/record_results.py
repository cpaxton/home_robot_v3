"""Preserve every fresh attempt and one copy of each matched nominal target set."""
import hashlib,json
from pathlib import Path
ROOT=Path.cwd();RAW=ROOT/'logs/eval/fresh-tracking-confirmation-v1';OUT=ROOT/'paper/data/whole_body_control/fresh_tracking_confirmation_v1'
r=json.loads((RAW/'summary.json').read_text());r['cases']={};r['targets']={};r['raw_sha256']={};cases=json.loads((OUT/'cases.json').read_text());kinds={c['id']:('goal_change' if int(c['id'].rsplit(':',1)[1])>=20 else c['mode']) for c in cases};r['by_use_case']={}
for name in ('teacher','ordinary','continued'):
 p=RAW/(name+'.json');report=json.loads(p.read_text());r['raw_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest();r['cases'][name]=[]
 for row in report['cases']:
  compact={k:v for k,v in row.items() if k!='goal_samples'};r['cases'][name].append(compact)
  if name=='teacher':r['targets'][row['id']]=row.get('goal_samples')
  elif row.get('goal_samples')!=r['targets'][row['id']]:raise RuntimeError('target samples differ')
 groups={}
 for kind in sorted(set(kinds.values())):
  selected=[x for x in report['cases'] if kinds[x['id']]==kind];groups[kind]={'successes':sum(x['success'] for x in selected),'attempts':len(selected),'unsafe_episodes':sum(any(x['safety_flags'].values()) for x in selected)}
 r['by_use_case'][name]=groups
(OUT/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
(OUT/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
