"""Wait for verified current-source regression evidence, then exec the series."""
from pathlib import Path
import json,os,sys,time
from emet.rl.regression_gate import verify_regressions
root=Path.cwd();recovery=root/'logs/recovery/mastery-long-v1'
receipt=recovery/'regressions-post-main.receipt.json'
log=recovery/'regression-post-main-driver.log'
status={'status':'waiting_for_full_controller_regressions','started_unix':time.time(),'hardware_used':False}
def save():
 p=recovery/'launch-status.json';pending=p.with_suffix('.pending.json');pending.write_text(json.dumps(status,indent=2)+'\n');pending.replace(p)
save();print('WAITING for current-source regression receipt',flush=True)
for attempt in range(720):
 if receipt.exists():
  try:r=verify_regressions(receipt)
  except json.JSONDecodeError:
   time.sleep(0.1);continue
  status.update(status='gate_passed_starting_series',controller_tests=len(r['test_ids']),updated_unix=time.time());save()
  print('GATE PASS',len(r['test_ids']),'starting series',flush=True)
  os.execv(sys.executable,[sys.executable,str(recovery/'train_series.py')])
 if log.exists() and 'Traceback' in log.read_text():
  status.update(status='blocked_by_failed_controller_gate',updated_unix=time.time());save()
  raise RuntimeError('controller regression failed; no PPO run started')
 time.sleep(5)
status.update(status='gate_wait_timeout',updated_unix=time.time());save()
raise RuntimeError('no complete regression receipt within one hour; training not started')
