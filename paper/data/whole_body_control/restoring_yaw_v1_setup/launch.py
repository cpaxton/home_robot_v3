"""One source-gated CPU run; current baseline, structural intervention, PPO, final replay."""
import json,subprocess,sys,time
from pathlib import Path
from emet.rl.regression_gate import verify_regressions
from emet.rl.manipulation.mastery import load_suite,read_report,validate_preflight
root=Path.cwd();rec=root/'logs/recovery/restoring-yaw-v1'
suite=root/'configs/benchmarks/mars_tracking_mastery_restoring_yaw_v1.json'
teacher=root/'logs/eval/mars-restoring-yaw-v1-teacher/report.json.gz'
receipt=rec/'regressions.receipt.json'
output=root/'logs/manipulation/mars-mastery-restoring-yaw-v1-seed90'
status={'status':'waiting_for_gates','started_unix':time.time(),'steps':65536,'seed':90,'repeats':False,'hardware_used':False}
def save():
 p=rec/'STATE.json';tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(p)
save()
try:
 for _ in range(720):
  if teacher.exists() and receipt.exists():
   evidence=verify_regressions(receipt);validate_preflight(read_report(teacher),load_suite(suite),['base']);break
  for name in ['preflight.log','regression-driver.log']:
   p=rec/name
   if p.exists() and 'Traceback' in p.read_text():raise RuntimeError('Failed gate '+name)
  time.sleep(5)
 else:raise RuntimeError('Gate timeout')
 status.update(status='baseline_expanded',tests=len(evidence['test_ids']),teacher_cases=len(read_report(teacher)['cases']));save()
 subprocess.run([sys.executable,'-m','emet.rl.manipulation.mastery','evaluate','--suite',str(suite),'--stages','base','--population','admission','--policy',str(root/'logs/manipulation/mars-mastery-crossed-v1-seed90/last'),'--preflight',str(teacher),'--output',str(root/'logs/eval/mars-restoring-yaw-v1-baseline')],check=True,stdin=subprocess.DEVNULL)
 assert not output.exists(),'Refusing overwritten/repeated training'
 command=[sys.executable,'-m','emet.rl.manipulation.mastery','train','--suite',str(suite),'--stages','base','--steps','65536','--seed','90','--eval-every','16384','--learning-rate','0.00005','--base-std','0.005','--arm-std','0.01','--initialize-from',str(root/'logs/manipulation/mars-mastery-crossed-v1-seed90/last'),'--preflight',str(teacher),'--regression-receipt',str(receipt),'--audit-initial-admission','--wandb-mode','offline','--output',str(output)]
 status.update(status='training',command=command,updated_unix=time.time());save()
 subprocess.run(command,check=True,stdin=subprocess.DEVNULL)
 status.update(status='final_expanded_replay',updated_unix=time.time());save()
 subprocess.run([sys.executable,'-m','emet.rl.manipulation.mastery','evaluate','--suite',str(suite),'--stages','base','--population','admission','--policy',str(output/'last'),'--preflight',str(teacher),'--output',str(root/'logs/eval/mars-restoring-yaw-v1-final')],check=True,stdin=subprocess.DEVNULL)
 status.update(status='complete',completed_unix=time.time());save()
except BaseException as exc:
 status.update(status='failed',error=str(exc));save();raise
