# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Start/stop innate Mars ZMQ bridge on a robot (innate-os + Zenoh)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from emet.deploy import _ssh_run
from emet.utils.connection import get_active_connection, get_connection, save_connection

DEFAULT_INNATE_WORKSPACE = "~/innate-os/ros2_ws"
DEFAULT_EMET_DIR = "~/emet"
TMUX_SESSION = "ros_nodes"
TMUX_WINDOW = "emet-bridge"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_mars_target(
    *,
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    connection_name: str | None = None,
) -> tuple[str, str, str | None, str, str]:
    """Return (host, user, password, workspace, emet_dir)."""
    conn = None
    if connection_name:
        conn = get_connection(connection_name)
    elif not host:
        conn = get_active_connection()
    if conn:
        host = host or conn.get("host")
        user = user or conn.get("user", "root")
        password = password if password is not None else conn.get("password")
        workspace = str(conn.get("workspace") or DEFAULT_INNATE_WORKSPACE)
        emet_dir = str(conn.get("emet_dir") or DEFAULT_EMET_DIR)
    else:
        workspace = DEFAULT_INNATE_WORKSPACE
        emet_dir = DEFAULT_EMET_DIR
    if not host:
        raise SystemExit(
            "No robot host. Use: emet mars start --ip <host> --username <user>, "
            "or emet connect save <host> --user <user>"
        )
    user = user or "root"
    password = password or os.environ.get("EMET_ROBOT_PASSWORD")
    return host.strip(), user.strip(), password, workspace.rstrip("/"), emet_dir.rstrip("/")


def _remote_bridge_launch_cmd(*, workspace: str, emet_dir: str) -> str:
    emet_core = f"{emet_dir}/emet_core"
    return (
        f"source {emet_dir}/bridge_env.sh 2>/dev/null || "
        f"export PYTHONPATH={emet_core}:$PYTHONPATH; "
        f"source ~/innate-os/dds/setup_dds.zsh && "
        f"source {workspace}/install/setup.zsh && "
        f"export PYTHONPATH={emet_core}:$PYTHONPATH && "
        f"ros2 launch innate_mars_bridge server.launch.py"
    )


def _kill_bridge_remote() -> str:
    """Free ZMQ ports (avoids pkill -f matching the SSH shell command line)."""
    return "fuser -k 4401/tcp 4402/tcp 4403/tcp 4404/tcp 2>/dev/null || true"


def start_bridge_on_robot(
    host: str,
    user: str,
    password: str | None,
    *,
    workspace: str = DEFAULT_INNATE_WORKSPACE,
    emet_dir: str = DEFAULT_EMET_DIR,
) -> None:
    """Start innate_mars_bridge inside innate-os tmux (Zenoh DDS env)."""
    launch_line = _remote_bridge_launch_cmd(workspace=workspace, emet_dir=emet_dir)
    remote = (
        f"{_kill_bridge_remote()}; "
        "sleep 1; "
        f"if ! tmux has-session -t {TMUX_SESSION} 2>/dev/null; then "
        f"echo 'ERROR: tmux session {TMUX_SESSION} not found. "
        "On the robot run: cd ~/innate-os && innate service start'; "
        "exit 1; "
        "fi; "
        f"tmux kill-window -t {TMUX_SESSION}:{TMUX_WINDOW} 2>/dev/null || true; "
        f"tmux new-window -t {TMUX_SESSION} -n {TMUX_WINDOW}; "
        f"tmux send-keys -t {TMUX_SESSION}:{TMUX_WINDOW} {launch_line!r} C-m"
    )
    print(f"Starting innate Mars bridge on {user}@{host} (tmux {TMUX_SESSION}:{TMUX_WINDOW})...")
    _ssh_run(host, user, password, remote, check=True)


def stop_bridge_on_robot(host: str, user: str, password: str | None) -> None:
    remote = (
        f"{_kill_bridge_remote()}; "
        f"tmux kill-window -t {TMUX_SESSION}:{TMUX_WINDOW} 2>/dev/null || true; "
        "echo bridge stopped"
    )
    print(f"Stopping innate Mars bridge on {user}@{host}...")
    _ssh_run(host, user, password, remote, check=False)


def bridge_status_on_robot(host: str, user: str, password: str | None) -> None:
    remote = (
        "echo '--- bridge process ---'; "
        "pgrep -af 'innate_mars_zmq_server' || echo '(not running)'; "
        "echo '--- zmq ports ---'; "
        "ss -tln 2>/dev/null | grep -E ':440[1-4]' || echo '(4401-4404 not listening)'; "
        f"echo '--- tmux ---'; "
        f"tmux has-session -t {TMUX_SESSION} 2>/dev/null && "
        f"tmux capture-pane -t {TMUX_SESSION}:{TMUX_WINDOW} -p 2>/dev/null | tail -6 "
        f"|| echo '({TMUX_SESSION}:{TMUX_WINDOW} unavailable)'"
    )
    _ssh_run(host, user, password, remote, check=False)


def mars_start(
    *,
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    connection_name: str | None = None,
    save_profile: bool = True,
    deploy: bool = False,
    preview: bool = False,
    wait_s: float = 20.0,
) -> None:
    host, user, password, workspace, emet_dir = resolve_mars_target(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
    )

    if save_profile and host:
        name = save_connection(
            host=host,
            user=user,
            password=password,
            name=host,
            set_active=True,
            workspace=workspace,
            emet_dir=emet_dir,
            robot="innate_mars",
        )
        print(f"Saved connection profile '{name}'.")

    if deploy:
        from emet.deploy import deploy as deploy_impl

        deploy_impl(
            host=host,
            user=user,
            password=password,
            workspace=workspace,
            emet_dir=emet_dir,
            start_bridge=False,
            root=_project_root(),
        )

    start_bridge_on_robot(host, user, password, workspace=workspace, emet_dir=emet_dir)

    if wait_s > 0:
        print(f"Waiting {wait_s:.0f}s for bridge startup...")
        time.sleep(wait_s)

    bridge_status_on_robot(host, user, password)

    if preview:
        print("Running camera preview...")
        cmd = [
            sys.executable,
            "-m",
            "emet.app.preview_robot_cameras",
            "--source",
            "zmq",
            "--robot",
            "innate_mars",
            "--robot-ip",
            host,
        ]
        subprocess.run(cmd, check=False)

    print(f"Bridge ZMQ: tcp://{host}:4401 (obs) 4403 (state)")
    print(f"Preview: uv run emet preview-cameras --source zmq --robot innate_mars --robot-ip {host}")
