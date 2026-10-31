"""Freeze completed single-seed evidence; no policy promotion or weight upload."""
from pathlib import Path
import collections,gzip,hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emet.rl.regression_gate import verify_regressions
from emet.rl.manipulation.mastery import read_report

root=Path.cwd();local=root/'logs/manipulation/mars-mastery-long-v1/seed-90';out=root/'paper/data/whole_body_control/mastery_long_v1_results';out.mkdir(exist_ok=False)
verify_regressions(root/'logs/recovery/mastery-long-v1/regressions-post-main.receipt.json')
status=json.loads((local.parent/'status.json').read_text());assert status['status']=='complete' and status['seeds']==[90]
teacher=read_report(root/'logs/eval/mars-mastery-long-preflight-v2/report.json.gz');paired={c['id']:c for c in teacher['cases']}
paths=sorted(local.glob('step-*'),key=lambda p:int(p.name.split('-')[1]))+[local/'last'];checkpoints=[];raw_hashes={};codes=collections.Counter();flags=collections.Counter()
for p in paths:
 report=read_report(p/'evaluation.json.gz');manifest=json.loads((p/'manifest.json').read_text());assert report['source_hashes']==teacher['source_hashes']
 rows=[]
 for c in report['cases']:
  t=paired[c['id']]
  for key in ['initial_state_sha256','reference_sha256']:assert c['admission'][key]==t['admission'][key]
  rows.append({k:c[k] for k in ['id','split','stage','admitted','code','mastered','tracking_and_settling_pass','smoothness_pass','failure_reasons','metrics']})
  codes[c['code']]+=1
  for k in ['wheel_lift','unsafe_contact','invalid_physics']:flags[k]+=bool(c['final_info'].get(k))
 checkpoints.append({'checkpoint':p.name,'steps':manifest['steps'],'weights_sha256':manifest['weights_sha256'],'summary':json.loads((p/'summary.json').read_text()),'cases':rows})
 for name in ['evaluation.json.gz','manifest.json']:
  raw_hashes[str((p/name).relative_to(root))]=hashlib.sha256((p/name).read_bytes()).hexdigest()
(out/'checkpoint-cases.json.gz').write_bytes(gzip.compress(json.dumps(checkpoints,allow_nan=False).encode(),mtime=0))
for name in ['metrics.jsonl','updates.jsonl']:
 (out/(name+'.gz')).write_bytes(gzip.compress((local/name).read_bytes(),mtime=0))
for name in ['run.json','complete.json']:shutil.copy2(local/name,out/name)
shutil.copy2(local.parent/'status.json',out/'series-complete.json')
for name in ['evaluation.json.gz','summary.json','manifest.json']:shutil.copy2(local/'last'/name,out/('final-'+name))
shutil.copy2(root/'logs/eval/mars-mastery-long-settling-v1/report.json.gz',out/'settling-replay.json.gz')
shutil.copy2(root/'logs/recovery/mastery-long-v1/replay_settling.py',out/'replay_settling.py')
rows=[json.loads(line) for line in (local/'metrics.jsonl').read_text().splitlines()];training_codes=collections.Counter()
for r in rows:
 for c in r['rollout/cases'].values():training_codes.update(c['terminal_codes'])
summary={'scope':__doc__,'steps':262144,'seed':90,'cancelled_seeds':[91,92],'checkpoints':len(checkpoints),'final_mastery':0,'final_tracking_and_settling':4,'final_cases':10,'teacher_base_mastered':10,'training_terminal_codes':dict(training_codes),'checkpoint_eval_codes':dict(codes),'checkpoint_eval_safety_flags':dict(flags),'coverage_steps':rows[-1]['sampling/steps'],'coverage_episode_starts':rows[-1]['sampling/episode_starts'],'source_hashes':teacher['source_hashes'],'raw_local_trace_sha256':raw_hashes,'weights_uploaded':False,'safety_promoted':False,'hardware_used':False,'source_unchanged_receipt_verified':True}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
x=np.array([c['steps'] for c in checkpoints])/1000
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for split,color in [('train','tab:blue'),('validation','tab:orange')]:
 axes[0,0].plot(x,[c['summary']['groups']['base/'+split]['tracking_and_settling_passed'] for c in checkpoints],label=split,color=color)
 axes[0,0].plot(x,[c['summary']['groups']['base/'+split]['mastered'] for c in checkpoints],ls=':',color=color,label=split+' full mastery')
 for direction,style in [('forward','-'),('reverse','--')]:
  cid=split+':base:'+direction
  axes[0,1].plot(x,[next(r for r in c['cases'] if r['id']==cid)['metrics']['base_error_m']*1000 for c in checkpoints],ls=style,color=color,label=split+' '+direction)
 for direction,style in [('yaw_left','-'),('yaw_right','--')]:
  cid=split+':base:'+direction
  axes[1,0].plot(x,[next(r for r in c['cases'] if r['id']==cid)['metrics']['base_heading_error_rad'] for c in checkpoints],ls=style,color=color,label=split+' '+direction)
ratio=[]
for c in checkpoints:
 ratio.append([r['metrics']['base_linear_jerk']/max(.1,1.25*paired[r['id']]['metrics']['base_linear_jerk']) for r in c['cases']])
axes[1,1].plot(x,np.median(ratio,axis=1),label='median across ten cases')
axes[1,1].plot(x,np.max(ratio,axis=1),label='maximum across ten cases')
axes[1,1].axhline(1,color='black',ls=':',label='paired jerk limit')
axes[1,0].axhline(.01,color='black',ls=':',label='heading tolerance')
axes[0,0].set(ylabel='Cases passed (of 5 per split)',ylim=(-.2,5.2),title='Tracking/2s hold improves; full mastery remains poor')
axes[0,1].set(ylabel='Endpoint XY error (mm)',title='Forward/reverse translation only')
axes[1,0].set(ylabel='Endpoint yaw error (rad)',title='Turning remains inconsistent')
axes[1,1].set(ylabel='Peak base-linear jerk / paired limit',title='Smoothness degrades during learning')
for ax in axes.flat:
 ax.set_xlabel('Environment steps (thousands)');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.savefig(out/'learning-curves.svg');plt.close(fig)
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
(out/'files-sha256.json').write_text(json.dumps(files,indent=2)+'\n');print('Archived completed run:',{k:summary[k] for k in ['steps','checkpoints','final_mastery','final_tracking_and_settling','checkpoint_eval_safety_flags']},flush=True)
