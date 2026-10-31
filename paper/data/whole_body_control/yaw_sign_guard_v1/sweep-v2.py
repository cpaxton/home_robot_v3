"""One CPU worker: matched direction/scales with/without guard, then teacher admission."""
import json,subprocess,sys
from pathlib import Path
root=Path.cwd()
status={'status':'running','hardware_used':False,'training_started':False}
p=root/'logs/recovery/yaw-sign-20261030/STATE.json'
def save():p.write_text(json.dumps(status,indent=2)+'\n')
try:
 for name,extra in [('unguarded',[]),('guarded',['--guard'])]:
  status['stage']=name;save()
  subprocess.run([sys.executable,'-m','emet.rl.manipulation.yaw_audit','--policy',str(root/'logs/manipulation/mars-mastery-crossed-v1-seed90/last'),'--output',str(root/'logs/eval/mars-yaw-sign-20261030-v2'/name),*extra],check=True)
 status['stage']='teacher-admission50';save()
 subprocess.run([sys.executable,'-m','emet.rl.manipulation.mastery','preflight','--suite',str(root/'configs/benchmarks/mars_tracking_mastery_yaw_curriculum_v1.json'),'--stages','base','--output',str(root/'logs/eval/mars-yaw-sign-20261030-v2/teacher-admission')],check=True)
 status['status']='complete';save()
except BaseException as exc:
 status.update(status='failed',error=str(exc));save();raise
