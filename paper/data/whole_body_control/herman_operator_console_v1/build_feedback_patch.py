"""Compile in a temporary robot workspace; never install into active ROS or run arm."""
import json,shlex,subprocess
from pathlib import Path
root=Path.cwd();patch=root/'docs/patches/innate_os_feedback_units_herman_20261117.patch';meta=json.loads(patch.with_suffix('.meta.json').read_text())
remote=r'''import hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
request=json.load(__import__('sys').stdin);home=Path.home();upstream=home/'innate-os';relative=Path('ros2_ws/src/maurice_bot/maurice_arm');source=upstream/relative
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
if revision!=request['metadata']['upstream_revision']:raise RuntimeError('upstream revision changed')
for name,digest in request['metadata']['before_sha256'].items():
 if hashlib.sha256((upstream/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('audited file changed '+name)
def hashes():return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
before=hashes();workspace=Path(tempfile.mkdtemp(prefix='emet-feedback-build-20261117-'));destination=workspace/relative;destination.parent.mkdir(parents=True);shutil.copytree(source,destination)
patch=workspace/'feedback.patch';patch.write_text(request['patch'])
subprocess.run(['git','apply','--check',str(patch)],cwd=workspace,check=True,capture_output=True,text=True)
subprocess.run(['git','apply',str(patch)],cwd=workspace,check=True,capture_output=True,text=True)
env=dict(os.environ,MAKEFLAGS='-j1',CMAKE_BUILD_PARALLEL_LEVEL='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env.pop('PYTHONPATH',None)
setup='source /opt/ros/humble/setup.bash >/dev/null\nsource ~/innate-os/ros2_ws/install/setup.bash >/dev/null\n'
command=setup+'exec timeout 900 nice -n 10 colcon build --executor sequential --packages-select maurice_arm --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF'
started=time.time();log=workspace/'build.log'
with log.open('w') as output:result=subprocess.run(['bash','-c',command],cwd=workspace/'ros2_ws',env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=930)
unchanged=before==hashes();binary=workspace/'ros2_ws/install/maurice_arm/lib/maurice_arm/arm'
report={'format':'herman-isolated-ros-build-v1','upstream_revision':revision,'patch_sha256':hashlib.sha256(request['patch'].encode()).hexdigest(),'temporary_workspace':str(workspace),'returncode':result.returncode,'elapsed_s':time.time()-started,'original_package_unchanged':unchanged,'original_package_files':len(before),'active_install_modified':False,'driver_binary_executed':False,'robot_commands_sent':0,'hardware_stopping_validated':False,'build_log':log.read_text()[-16000:]}
if binary.exists():report['binary']={'path':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'bytes':binary.stat().st_size,'file':subprocess.check_output(['file',str(binary)],text=True).strip()}
print(json.dumps(report))
if not unchanged:raise RuntimeError('original source changed during isolated build')
'''
payload=json.dumps({'metadata':meta,'patch':patch.read_text()})
result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','jetson1@192.168.1.43','python3 -c '+shlex.quote(remote)],input=payload,capture_output=True,text=True,timeout=960)
if result.returncode:raise RuntimeError(result.stderr[-3000:])
report=json.loads(result.stdout);out=root/'logs/recovery/operator-console-v1';(out/'isolated-ros-build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='build_log'},indent=2));print(report['build_log'][-1200:])
