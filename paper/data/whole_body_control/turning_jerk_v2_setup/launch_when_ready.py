"""Wait for current-source regression evidence, then start one bounded refinement."""
import json,os,sys,time
from pathlib import Path
from emet.rl.regression_gate import verify_regressions
from emet.rl.manipulation.mastery import load_suite,read_report,validate_preflight
root=Path.cwd();recovery=root/'logs/recovery/mastery-objective-v2';receipt=recovery/'regressions.receipt.json';output=root/'logs/manipulation/mars-mastery-objective-v2-seed90'
status={'status':'waiting_for_regressions','started_unix':time.time(),'seed':90,'steps':131072,'automatic_extension':False,'repeated_seeds':False,'hardware_used':False}
def save():
 p=recovery/'launch-status.json';q=p.with_suffix('.pending.json');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
save()
for attempt in range(720):
 if receipt.exists():
  try:evidence=verify_regressions(receipt)
  except json.JSONDecodeError:time.sleep(.1);continue
  suite=root/'configs/benchmarks/mars_tracking_mastery_v2.json';preflight=root/'logs/eval/mars-mastery-objective-v2-teacher/report.json.gz'
  validate_preflight(read_report(preflight),load_suite(suite),['base'])
  assert not output.exists(),'refusing duplicate or overwritten training'
  command=[sys.executable,'-m','emet.rl.manipulation.mastery','train','--suite',str(suite),'--stages','base','--steps','131072','--seed','90','--eval-every','16384','--learning-rate','0.00005','--base-std','0.01','--arm-std','0.03','--initialize-from',str(root/'logs/manipulation/mars-mastery-long-v1/seed-90/last'),'--preflight',str(preflight),'--regression-receipt',str(receipt),'--wandb-mode','offline','--output',str(output)]
  status.update(status='starting',controller_tests=len(evidence['test_ids']),command=command,updated_unix=time.time());save();print('GATE PASS',len(evidence['test_ids']),'starting single refinement',flush=True)
  os.execv(sys.executable,command)
 driver=recovery/'regression-driver.log'
 if driver.exists() and 'Traceback' in driver.read_text():
  status.update(status='failed_regression_gate',updated_unix=time.time());save();raise RuntimeError('regressions failed; no training started')
 time.sleep(5)
status['status']='gate_timeout';save();raise RuntimeError('no complete passing receipt within one hour')
