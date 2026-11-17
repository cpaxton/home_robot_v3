"""Construct an SSH-only, subscriber-only camera/feedback relay."""

import re
import shlex
from pathlib import Path


def relay_command(target, *, seconds=3600, fps=5):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", target) or target.startswith("-"):
        raise ValueError("explicit user@host SSH destination required")
    if not 0 < seconds <= 21600 or not 0 < fps <= 10:
        raise ValueError("bounded relay duration and rate required")
    source = Path(__file__).with_name("passive_relay.py").read_text()
    # Reuse only ROS/Zenoh settings from the running state publisher. Never dump
    # arbitrary process environments or secrets into logs/browser responses.
    bootstrap = (
        """import os,shlex
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
command=setup+'exec python3 -u -c '+shlex.quote(SOURCE)+' --seconds '+str(SECONDS)+' --fps '+str(FPS)
os.execvpe('bash',['bash','-c',command],env)
""".replace("SOURCE", repr(source))
        .replace("SECONDS", repr(seconds))
        .replace("FPS", repr(fps))
    )
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
