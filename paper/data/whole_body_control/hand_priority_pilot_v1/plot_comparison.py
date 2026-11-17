"""Paired active-hand endpoints and base jerk, with all contact failures marked."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path.cwd();raw=ROOT/'logs/manipulation/hand-priority-pilot-v1/physical-comparison.json';out=ROOT/'paper/data/whole_body_control/hand_priority_pilot_v1';r=json.loads(raw.read_text());rows=[]
for cohort,controllers in r['cases'].items():
 if cohort=='tamp':continue
 for a,b in zip(controllers['ordinary_noise'],controllers['hand_priority_only'],strict=True):
  if a['id']!=b['id']:raise RuntimeError('unmatched case')
  rows.append({'id':a['id'],'hand_active':bool(a['metrics']['hand_target_active']),'control_position_mm':1000*a['metrics']['ee_position_error'],'candidate_position_mm':1000*b['metrics']['ee_position_error'],'control_linear_jerk':a['metrics']['peak_base_linear_jerk_m_s3'],'candidate_linear_jerk':b['metrics']['peak_base_linear_jerk_m_s3'],'control_unsafe':any(a['safety_flags'].values()),'candidate_unsafe':any(b['safety_flags'].values())})
plt.rcParams.update({'font.size':9,'svg.fonttype':'none'});fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
for ax,x,y,title in zip(axes,('control_position_mm','control_linear_jerk'),('candidate_position_mm','candidate_linear_jerk'),('Active-hand endpoint error (mm)','Peak base linear jerk (m/s³)'),strict=True):
 group=[row for row in rows if x!='control_position_mm' or row['hand_active']]
 for row in group:ax.scatter(row[x],row[y],s=24,c='#cc5522' if row['candidate_unsafe'] else '#2676ad',marker='x' if row['candidate_unsafe'] else 'o',alpha=.75)
 limit=max(max(row[x],row[y]) for row in group)*1.08;ax.plot([0,limit],[0,limit],c='.5',lw=.8,label='equal outcome');ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Ordinary control',ylabel='Hand-pose weight 4',title=title);ax.grid(alpha=.18)
 if x=='control_position_mm':ax.axhline(3,c='#777',ls=':',lw=.8);ax.axvline(3,c='#777',ls=':',lw=.8)
fig.suptitle('Fixed final policies — points below diagonal improve; orange × = candidate safety failure',fontsize=9)
fig.savefig(out/'paired-outcomes.svg',metadata={'Date':None});plt.close(fig)
(out/'plot-data.json').write_text(json.dumps({'raw_report_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'scope':'All general/coupled cases for jerk; endpoint plot requires active hand. No task thresholds changed.','rows':rows},indent=2)+'\n')
