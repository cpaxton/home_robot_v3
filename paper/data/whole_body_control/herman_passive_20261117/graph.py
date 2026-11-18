"""Read ROS graph using the running publisher's middleware/domain settings."""
import json,shlex,subprocess
from pathlib import Path
code=r'''import json,os,subprocess
from pathlib import Path
env=dict(os.environ);allowed={"ROS_DOMAIN_ID","ROS_LOCALHOST_ONLY","RMW_IMPLEMENTATION","ZENOH_SESSION_CONFIG_URI","ZENOH_ROUTER_CONFIG_URI","ZENOH_CONFIG_OVERRIDE"};selected={}
for proc in Path("/proc").iterdir():
 if not proc.name.isdigit():continue
 try:
  if not (proc/"comm").read_text().startswith("robot_state_pub"):continue
  values=dict(x.split("=",1) for x in (proc/"environ").read_bytes().decode(errors="replace").split("\x00") if "=" in x);selected={k:v for k,v in values.items() if k in allowed};break
 except OSError:pass
env.update(selected);setup="source /opt/ros/humble/setup.bash >/dev/null\nsource ~/innate-os/ros2_ws/install/setup.bash >/dev/null\n"
reports={"middleware":selected.get("RMW_IMPLEMENTATION"),"domain":selected.get("ROS_DOMAIN_ID"),"copied_setting_names":sorted(selected),"motion_commands_sent":False}
for name,command in (("nodes","ros2 node list --no-daemon --spin-time 3"),("topics","ros2 topic list -t --no-daemon --spin-time 3")):
 r=subprocess.run(["bash","-c",setup+command],env=env,capture_output=True,text=True,timeout=12);reports[name]={"exit":r.returncode,"output":r.stdout[:16000],"stderr":r.stderr[:2000]}
print(json.dumps(reports))'''
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=yes','jetson1@192.168.1.43','python3 -c '+shlex.quote(code)]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
if r.returncode:raise RuntimeError(r.stderr[:1500])
report=json.loads(r.stdout);Path('logs/recovery/herman-passive-20261117/graph.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
