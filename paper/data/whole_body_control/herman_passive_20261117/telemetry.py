"""Subscribe to existing robot state/odom topics; never create command publishers."""
import json,shlex,subprocess
from pathlib import Path
listener=r'''import json,math,time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from rosidl_runtime_py.convert import message_to_ordereddict
rclpy.init();node=rclpy.create_node("emet_passive_feedback_audit");records={"arm":[],"commanded_arm":[],"odom":[]};subscriptions=[]
def callback(name,msg):
 wall=time.time();stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
 if name=="odom":
  p=msg.pose.pose.position;q=msg.pose.pose.orientation;v=msg.twist.twist
  value={"position":[p.x,p.y,p.z],"quaternion_xyzw":[q.x,q.y,q.z,q.w],"twist":[v.linear.x,v.linear.y,v.linear.z,v.angular.x,v.angular.y,v.angular.z]}
 else:value={"names":list(msg.name),"position":list(msg.position),"velocity":list(msg.velocity),"effort":list(msg.effort)}
 records[name].append({"arrival_monotonic":time.monotonic(),"stamp_s":stamp,"age_s":wall-stamp if stamp>0 else None,"value":value})
for name,topic,kind in (("arm","/mars/arm/state",JointState),("commanded_arm","/mars/arm/command_state",JointState),("odom","/odom",Odometry)):
 subscriptions.append(node.create_subscription(kind,topic,lambda msg,n=name:callback(n,msg),qos_profile_sensor_data))
deadline=time.monotonic()+6
while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
result={"scope":"passive subscription to existing publishers; values stationary or moving are observed, not commanded","motion_commands_sent":False,"command_publishers_created":False,"effort_units":"uncalibrated driver current/load proxies, not certified Nm","topics":{}}
def quantile(values,q):
 a=sorted(values);return a[min(len(a)-1,int(q*(len(a)-1)))] if a else None
for name,rows in records.items():
 intervals=[b["arrival_monotonic"]-a["arrival_monotonic"] for a,b in zip(rows,rows[1:])];ages=[x["age_s"] for x in rows if x["age_s"] is not None];result["topics"][name]={"samples":len(rows),"interval_median_s":quantile(intervals,.5),"interval_p95_s":quantile(intervals,.95),"maximum_interval_s":max(intervals) if intervals else None,"stamp_age_median_s":quantile(ages,.5),"stamp_age_max_s":max(ages) if ages else None,"zero_stamp_samples":sum(x["stamp_s"]==0 for x in rows),"first":rows[0]["value"] if rows else None,"last":rows[-1]["value"] if rows else None}
node.destroy_node();rclpy.shutdown();print(json.dumps(result))'''
remote=r'''import json,os,shlex,subprocess
from pathlib import Path
env=dict(os.environ);allowed={"ROS_DOMAIN_ID","ROS_LOCALHOST_ONLY","RMW_IMPLEMENTATION","ZENOH_SESSION_CONFIG_URI","ZENOH_ROUTER_CONFIG_URI","ZENOH_CONFIG_OVERRIDE"}
for p in Path("/proc").iterdir():
 if not p.name.isdigit():continue
 try:
  if not (p/"comm").read_text().startswith("robot_state_pub"):continue
  values=dict(x.split("=",1) for x in (p/"environ").read_bytes().decode(errors="replace").split("\x00") if "=" in x);env.update({k:v for k,v in values.items() if k in allowed});break
 except OSError:pass
setup="source /opt/ros/humble/setup.bash >/dev/null\nsource ~/innate-os/ros2_ws/install/setup.bash >/dev/null\n"
r=subprocess.run(["bash","-c",setup+"python3 -c "+shlex.quote(LISTENER)],env=env,capture_output=True,text=True,timeout=14)
if r.returncode:raise RuntimeError(r.stderr[:2000])
for line in reversed(r.stdout.splitlines()):
 if line.startswith("{"):print(line);break
else:raise RuntimeError("no telemetry JSON")'''.replace('LISTENER',repr(listener))
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=yes','jetson1@192.168.1.43','python3 -c '+shlex.quote(remote)];r=subprocess.run(cmd,capture_output=True,text=True,timeout=22)
if r.returncode:raise RuntimeError(r.stderr[:2000])
report=json.loads(r.stdout);Path('logs/recovery/herman-passive-20261117/telemetry.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
