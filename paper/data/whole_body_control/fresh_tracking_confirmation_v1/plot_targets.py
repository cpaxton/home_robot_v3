"""Show nominal active targets, using five saved samples per stage for inspection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'paper/data/whole_body_control/fresh_tracking_confirmation_v1';r=json.loads((OUT/'results.json').read_text());cases=json.loads((OUT/'cases.json').read_text());plt.rcParams.update({'svg.fonttype':'none','font.size':8});fig=plt.figure(figsize=(10,4.3),layout='constrained');base=fig.add_subplot(121);hand=fig.add_subplot(122,projection='3d');seen=set()
for c in cases:
 i=int(c['id'].rsplit(':',1)[1]);samples=r['targets'][c['id']]
 if not samples:continue
 kind='goal change' if i>=20 else c['mode'];color={'goal change':'#7a4da2','coupled':'#2676ad','hand_free_base':'#d08a23','base_only':'#555555'}[kind];label=kind if kind not in seen else None;seen.add(kind);b=np.array([x['base'] for x in samples]);p=np.array([x['hand'] for x in samples])
 if any(c['base_mask'][:2]):
  base.plot(b[:,0],b[:,1],color=color,lw=1,label=label);base.text(b[-1,0],b[-1,1],str(i),fontsize=6)
  if c['base_mask'][2]:base.arrow(b[-1,0],b[-1,1],.02*np.cos(b[-1,2]),.02*np.sin(b[-1,2]),color=color,width=.0005,length_includes_head=True)
 if c['mode']!='base_only':
  hand.plot(p[:,0],p[:,1],p[:,2],color=color,lw=1);hand.text(*p[-1,:3],str(i),fontsize=6);cols=p[-1,3:].reshape(3,2);rot=np.column_stack([cols,np.cross(cols[:,0],cols[:,1])])
  for axis,axiscolor in zip(rot.T,('r','g','b'),strict=True):
   q=p[-1,:3]+.015*axis;hand.plot([p[-1,0],q[0]],[p[-1,1],q[1]],[p[-1,2],q[2]],c=axiscolor,lw=.6,alpha=.7)
base.set(title='Nominal base goals: active masks only',xlabel='World x (m)',ylabel='World y (m)',aspect='equal');base.grid(alpha=.15);base.legend(loc='upper left',fontsize=7);hand.set(title='Hand targets: end-pose RGB axes',xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)');fig.suptitle('Fresh case IDs 0–23: five samples/stage for inspection, not execution',fontsize=10);fig.savefig(OUT/'target-gallery.svg',metadata={'Date':None});plt.close(fig)
