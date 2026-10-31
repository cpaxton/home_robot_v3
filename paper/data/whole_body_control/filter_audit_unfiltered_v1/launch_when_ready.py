"""Wait for fresh gates then exec one bounded unfiltered training comparison."""
import json,os,sys,time
from pathlib import Path
from emet.rl.regression_gate import verify_regressions
from emet.rl.manipulation.mastery import load_suite,read_report,validate_preflight
root=Path.cwd();recovery=root/'logs/recovery/mastery-filter-audit';receipt=recovery/'regressions.receipt.json';output=root/'logs/manipulation/mars-mastery-unfiltered-v1-seed90'
status={'status':'waiting_for_regression_and_teacher','started_unix':time.time(),'seed':90,'steps':65536,'repeats':False,'hardware_used':False}
def save():
 p=recovery/'launch-status.json';q=p.with_suffix('.pending.json');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
save()
for attempt in range(720):
 if receipt.exists():
  try:evidence=verify_regressions(receipt)
  except json.JSONDecodeError:time.sleep(.1);continue
  suite=root/'configs/benchmarks/mars_tracking_mastery_unfiltered_v1.json';preflight=root/'logs/eval/mars-mastery-unfiltered-v1-teacher/report.json.gz'
  if not preflight.exists():time.sleep(5);continue
  validate_preflight(read_report(preflight),load_suite(suite),['base'])
  assert not output.exists(),'refusing duplicate/overwritten training'
  command=[sys.executable,'-m','emet.rl.manipulation.mastery','train','--suite',str(suite),'--stages','base','--steps','65536','--seed','90','--eval-every','16384','--learning-rate','0.00005','--base-std','0.005','--arm-std','0.01','--command-smoothness-warmup','16384','--initialize-from',str(root/'logs/manipulation/mars-mastery-objective-v2-seed90/last'),'--preflight',str(preflight),'--regression-receipt',str(receipt),'--wandb-mode','offline','--output',str(output)]
  status.update(status='starting',controller_tests=len(evidence['test_ids']),command=command,updated_unix=time.time());save();print('GATE PASS',len(evidence['test_ids']),'single unfiltered comparison',flush=True)
  os.execv(sys.executable,command)
 log=recovery/'regression-driver.log'
 if log.exists() and 'Traceback' in log.read_text():
  status['status']='failed_regression_gate';save();raise RuntimeError('failed regressions; no training started')
 time.sleep(5)
status['status']='gate_timeout';save();raise RuntimeError('no passing gate within one hour')
