"""Read source/configuration and process ROS domains; never access a motor bus."""
import json,shlex,subprocess
from pathlib import Path
code=r'''import hashlib,json,os,subprocess
from pathlib import Path
root=Path.home()/"innate-os";package=root/"ros2_ws/src/maurice_bot/maurice_arm";report={"files":{},"ros_process_domains":[],"configuration_lines":[],"installed_arm_files":[],"motion_commands_sent":False,"drivers_started":False,"software_deployed":False}
for relative in ("maurice_arm/arm_control.cpp","maurice_arm/arm.cpp","maurice_arm/robot.cpp","include/maurice_arm/feedback_units.hpp","CMakeLists.txt"):
 p=package/relative
 if not p.exists():report["files"][relative]={"present":False};continue
 data=p.read_bytes();lines=data.decode(errors="replace").splitlines();keywords=("4096","velocities_rad.push_back","dynamixelVelocity","COMM_SUCCESS","isAvailable","return {z","vector<int>(servo_ids_","add_executable","arm_control.cpp")
 report["files"][relative]={"present":True,"sha256":hashlib.sha256(data).hexdigest(),"selected_lines":[{"line":i,"text":line.strip()} for i,line in enumerate(lines,1) if any(k in line for k in keywords)][:24]}
for p in (package/"config").glob("*.yaml"):
 for i,line in enumerate(p.read_text(errors="replace").splitlines(),1):
  if any(k in line for k in ("control_mode:","current_limit:","watchdog:","profile_velocity:","profile_acceleration:")):report["configuration_lines"].append({"path":str(p.relative_to(root)),"line":i,"text":line.strip()})
for proc in Path("/proc").iterdir():
 if not proc.name.isdigit():continue
 try:
  name=(proc/"comm").read_text().strip()
  if not any(k in name.lower() for k in ("robot_state","arm_control","maurice","dynamixel","nav2","controller")):continue
  env=dict(x.split("=",1) for x in (proc/"environ").read_bytes().decode(errors="replace").split("\x00") if "=" in x)
  report["ros_process_domains"].append({"name":name,"pid":int(proc.name),"ros_domain_id":env.get("ROS_DOMAIN_ID"),"rmw":env.get("RMW_IMPLEMENTATION"),"ros_localhost_only":env.get("ROS_LOCALHOST_ONLY")})
 except OSError:pass
installed=root/"ros2_ws/install/maurice_arm/lib/maurice_arm"
if installed.exists():report["installed_arm_files"]=[p.name for p in installed.iterdir() if p.is_file()]
print(json.dumps(report))'''
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=yes','jetson1@192.168.1.43','python3 -c '+shlex.quote(code)]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
if r.returncode:raise RuntimeError(r.stderr[:1500])
report=json.loads(r.stdout);out=Path('logs/recovery/herman-passive-20261117/source-audit.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
