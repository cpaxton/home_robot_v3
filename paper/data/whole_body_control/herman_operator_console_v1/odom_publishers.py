"""Inspect only current /odom publisher identities under the existing ROS RMW."""
import json,shlex,subprocess
from pathlib import Path
code=r'''import os,subprocess
from pathlib import Path
env=dict(os.environ);allowed={"ROS_DOMAIN_ID","ROS_LOCALHOST_ONLY","RMW_IMPLEMENTATION","ZENOH_SESSION_CONFIG_URI","ZENOH_ROUTER_CONFIG_URI","ZENOH_CONFIG_OVERRIDE"}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if not (p/'comm').read_text().startswith('robot_state_pub'):continue
  values=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\x00') if '=' in x);env.update({k:v for k,v in values.items() if k in allowed});break
 except OSError:pass
setup='source /opt/ros/humble/setup.bash >/dev/null\nsource ~/innate-os/ros2_ws/install/setup.bash >/dev/null\n'
subprocess.run(['bash','-c',setup+'ros2 topic info /odom -v'],env=env,check=True,timeout=12)
'''
result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','jetson1@192.168.1.43','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=22)
Path('logs/recovery/operator-console-v1/odom-publishers.txt').write_text(result.stdout+'\n'+result.stderr)
print(result.stdout[:5000]);print('exit',result.returncode)
