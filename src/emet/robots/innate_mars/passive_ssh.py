"""Construct SSH invocations for two repository-owned ROS subscriber programs."""

import re
import shlex
from pathlib import Path


def subscriber_command(target, *, program, arguments):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", target) or target.startswith("-"):
        raise ValueError("explicit user@host SSH destination required")
    if program not in {"passive_relay", "rehearsal_capture_ros"}:
        raise ValueError("only repository-owned subscriber programs are supported")
    source = Path(__file__).with_name(program + ".py").read_text()
    remote = shlex.join(["python3", "-u", "-c", source, *arguments])
    # Reuse only ROS/Zenoh settings from the running state publisher. Never dump
    # arbitrary process environments or secrets into logs/browser responses.
    bootstrap = """import os
from pathlib import Path
env=dict(os.environ)
allowed={"ROS_DOMAIN_ID","ROS_LOCALHOST_ONLY","RMW_IMPLEMENTATION","ZENOH_SESSION_CONFIG_URI","ZENOH_ROUTER_CONFIG_URI","ZENOH_CONFIG_OVERRIDE"}
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if not (proc/'comm').read_text().startswith('robot_state_pub'):continue
  values=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode(errors='replace').split('\\x00') if '=' in x)
  env.update({k:v for k,v in values.items() if k in allowed});break
 except OSError:pass
setup='source /opt/ros/humble/setup.bash >/dev/null\\nsource ~/innate-os/ros2_ws/install/setup.bash >/dev/null\\n'
command=setup+'exec '+REMOTE_COMMAND
os.execvpe('bash',['bash','-c',command],env)
""".replace("REMOTE_COMMAND", repr(remote))
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=2",
        target,
        "python3 -c " + shlex.quote(bootstrap),
    ]


def relay_command(target, *, seconds=3600, fps=5):
    if not 0 < seconds <= 21600 or not 0 < fps <= 10:
        raise ValueError("bounded relay duration and rate required")
    return subscriber_command(target, program="passive_relay", arguments=["--seconds", str(seconds), "--fps", str(fps)])
