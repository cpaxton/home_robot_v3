"""Plot fixed final physical outcomes; no checkpoint selection or confidence claims."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'
from emet.rl.regression_gate import ROOT
raw=ROOT/'logs/manipulation/wrapper-reward-training-v2/physical-comparison.json'
r=json.loads(raw.read_text());rows=r['cases']['new-cases'];controllers=['initial','contract','legacy'];labels=['Retained','New reward PPO','Legacy reward PPO'];colors=['#555555','#007a87','#ba5c13']
active=[x['id'] for x in rows['initial'] if x['metrics']['hand_target_active']]
fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
for name,label,color in zip(controllers,labels,colors,strict=True):
 by_id={x['id']:x for x in rows[name]};values=[1000*by_id[i]['metrics']['ee_position_error'] for i in active]
 axes[0].plot(np.arange(len(active)),values,marker='o',color=color,label=label,linewidth=1)
 for index,identity in enumerate(active):
  if any(by_id[identity]['safety_flags'].values()):axes[0].scatter(index,values[index],marker='x',s=80,color=color,zorder=4)
axes[0].axhline(3,color='#aa3333',linestyle='--',label='3 mm contract');axes[0].set_yscale('log');axes[0].set_xticks(np.arange(len(active)),[('T' if ':train:' in i else 'A')+i.rsplit(':',1)[1] for i in active]);axes[0].set_ylabel('Terminal active-hand position error (mm)');axes[0].set_xlabel('Training (T) and fresh audit (A) cases; x = safety stop');axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
cohorts=[('new-cases/train','New train'),('new-cases/audit','Fresh audit'),('frozen-coupled/all','Existing coupled'),('tamp','TAMP prefix')];positions=np.arange(len(cohorts));width=.25
for offset,(name,label,color) in enumerate(zip(controllers,labels,colors,strict=True)):
 items=[r['summary'][f'{key}/{name}'] for key,_ in cohorts];heights=[100*x['successes']/x['attempts'] for x in items];bars=axes[1].bar(positions+(offset-1)*width,heights,width,color=color,label=label)
 for bar,item in zip(bars,items,strict=True):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{item['successes']}/{item['attempts']}",ha='center',va='bottom',fontsize=8)
axes[1].set_xticks(positions,[label for _,label in cohorts]);axes[1].set_ylabel('Contract success (%)');axes[1].set_ylim(0,115);axes[1].legend(fontsize=8,loc='lower right');axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
fig.suptitle('Fixed 16,384-decision reward pilots: matched cases, one training seed, no promotion')
out=ROOT/'logs/manipulation/wrapper-reward-training-v2/physical-comparison.svg';fig.savefig(out);print(out)
