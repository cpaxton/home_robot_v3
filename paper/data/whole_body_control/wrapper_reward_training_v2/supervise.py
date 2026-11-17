"""Detached, CPU-pinned managed experiment; never authorizes hardware motion."""
import argparse,json,os,shlex,subprocess,traceback
from pathlib import Path
from emet.utils.job_registry import register_job,update_job
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--record-dir',type=Path,required=True);p.add_argument('--cpu',required=True);p.add_argument('--seconds',default='14400');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
command=['taskset','-c',a.cpu,'timeout',a.seconds,*command]
a.record_dir.mkdir(parents=True,exist_ok=True)
job=register_job(name=a.name,cmd=shlex.join(command),repo=Path.cwd(),out_dir=a.record_dir,log_path=a.record_dir/'job.log',pid=os.getpid(),status='running',meta={'hardware_used':False,'cpu':a.cpu,'immutable_runtime':str(Path.cwd())})
(a.record_dir/'job.json').write_text(json.dumps({'job_id':job.id,'pid':os.getpid(),'command':command},indent=2)+'\n')
try:
 result=subprocess.call(command)
 update_job(job.id,status='done' if result==0 else 'failed',error='' if result==0 else f'exit {result}')
except BaseException as exc:
 update_job(job.id,status='failed',error=repr(exc));traceback.print_exc();raise
raise SystemExit(result)
