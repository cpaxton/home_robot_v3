"""Plot completed checkpoint summaries without combining physical dimensions."""
from pathlib import Path
import gzip,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('paper/data/whole_body_control/unfiltered_v1_results')
points=json.loads(gzip.decompress((out/'checkpoint-cases.json.gz').read_bytes()))
x=[p['steps']/1000 for p in points]
fig,axes=plt.subplots(3,2,figsize=(10,9),layout='constrained')
for split,color in [('train','tab:blue'),('validation','tab:orange')]:
 axes[0,0].plot(x,[p['summary']['groups']['base/'+split]['mastered'] for p in points],label=split,color=color)
 axes[0,1].plot(x,[p['summary']['groups']['base/'+split]['tracking_and_settling_passed'] for p in points],label=split,color=color)
axes[0,0].set(title='Full mastery, paired teacher limits',ylabel='Passed cases (of 5)',ylim=(-.2,5.2))
axes[0,1].set(title='Tracking and two-second holding',ylabel='Passed cases (of 5)',ylim=(-.2,5.2))
for ax,key,title,unit in [(axes[1,0],'base_linear_jerk','Median peak measured linear jerk','m/s^3'),(axes[1,1],'base_yaw_jerk','Median peak measured yaw jerk','rad/s^3')]:
 ax.plot(x,[np.median([c['metrics'][key] for c in p['cases']]) for p in points]);ax.set(title=title,ylabel=unit)
for i,title,unit in [(0,'Median RMS raw linear-command jerk','m/s^3'),(1,'Median RMS raw yaw-command jerk','rad/s^3')]:
 ax=axes[2,i];ax.plot(x,[np.median([c['raw_command_jerk_rms'][i] for c in p['cases']]) for p in points]);ax.set(title=title,ylabel=unit)
for ax in axes.flat:
 ax.set_xlabel('Additional training steps (thousands)');ax.grid(alpha=.2)
for ax in axes[0]:ax.legend(fontsize=8)
fig.savefig(out/'learning-curves.svg');plt.close(fig)
