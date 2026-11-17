"""Managed, bounded passive operator view. No actuator command endpoints."""
import json,os,shlex,subprocess
from pathlib import Path
from emet.utils.job_registry import register_job,update_job
root=Path.cwd();out=root/'logs/jobs/herman-operator-console-v3';out.mkdir(parents=True,exist_ok=False)
command=['taskset','-c','12','timeout','--foreground','21900',str(root/'.venv/bin/python'),'-u','-m','emet.visualization.operator_console','--ssh','jetson1@192.168.1.43','--port','8770','--seconds','21600','--fps','5']
job=register_job(name='herman-operator-console-v3',cmd=shlex.join(command),repo=root,out_dir=out,log_path=out/'job.log',pid=os.getpid(),status='running',meta={'hardware_motion_enabled':False,'passive_ros_subscriptions_only':True,'operator_targets':'local_preview_only','cpu':12})
(out/'job.json').write_text(json.dumps({'job_id':job.id,'pid':os.getpid(),'command':command},indent=2)+'\n')
try:
 result=subprocess.call(command);update_job(job.id,status='done' if result==0 else 'failed',error='' if result==0 else f'exit {result}')
except BaseException as exc:
 update_job(job.id,status='failed',error=repr(exc));raise
raise SystemExit(result)
