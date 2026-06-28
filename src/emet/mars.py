# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Start/stop innate Mars ZMQ bridge on a robot (innate-os + Zenoh)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from emet.deploy import _ssh_capture, _ssh_run
from emet.utils.connection import get_active_connection, get_connection, save_connection
from emet.utils.terminal_ui import note, style

DEFAULT_INNATE_WORKSPACE = "~/innate-os/ros2_ws"
DEFAULT_EMET_DIR = "~/emet"
TMUX_SESSION = "ros_nodes"
TMUX_WINDOW = "emet-bridge"

MARS_ZMQ_PORTS: dict[int, str] = {
    4401: "observations",
    4402: "actions",
    4403: "state",
    4404: "servo",
}


@dataclass
class MarsBridgeStatus:
    host: str
    user: str
    pid: int | None = None
    process_cmd: str | None = None
    listening_ports: set[int] = field(default_factory=set)
    tmux_available: bool = False
    ros_log_lines: list[str] = field(default_factory=list)
    ssh_exit_code: int = 0

    @property
    def process_running(self) -> bool:
        return self.pid is not None

    @property
    def all_ports_listening(self) -> bool:
        return set(MARS_ZMQ_PORTS).issubset(self.listening_ports)

    @property
    def ready_for_stream(self) -> bool:
        return self.process_running and 4401 in self.listening_ports and 4403 in self.listening_ports

    def headline_log(self) -> str | None:
        for line in reversed(self.ros_log_lines):
            low = line.lower()
            if "innate_mars" in low or "waiting for cameras" in low or "error" in low:
                return line.strip()
        if self.ros_log_lines:
            return self.ros_log_lines[-1].strip()
        return None


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


def _remote_bridge_launch_cmd(*, workspace: str, emet_dir: str, onboard_da3: bool = False) -> str:
    emet_core = f"{emet_dir}/emet_core"
    emet_src = f"{emet_dir}/src"
    da3_env = "export EMET_MARS_ONBOARD_DA3=1; " if onboard_da3 else ""
    py_paths = f"{emet_core}:{emet_src}"
    return (
        f"{da3_env}"
        f"source {emet_dir}/bridge_env.sh 2>/dev/null || "
        f"export PYTHONPATH={py_paths}:$PYTHONPATH; "
        f"source ~/innate-os/dds/setup_dds.zsh && "
        f"source {workspace}/install/setup.zsh && "
        f"export PYTHONPATH={py_paths}:$PYTHONPATH && "
        f"ros2 launch innate_mars_bridge server.launch.py"
    )


def _kill_bridge_remote() -> str:
    """Free ZMQ ports (avoids pkill -f matching the SSH shell command line)."""
    return "fuser -k 4401/tcp 4402/tcp 4403/tcp 4404/tcp 2>/dev/null || true"


def _remote_status_cmd() -> str:
    return (
        "pgrep -af 'innate_mars_zmq_server' 2>/dev/null || true; "
        "echo '---'; "
        "ss -tlnH 2>/dev/null | grep -E ':440[1-4] ' || true; "
        "echo '---'; "
        f"tmux has-session -t {TMUX_SESSION} 2>/dev/null && "
        f"tmux capture-pane -t {TMUX_SESSION}:{TMUX_WINDOW} -p 2>/dev/null | tail -12 "
        f"|| echo '---tmux-unavailable---'"
    )


def parse_bridge_status_output(host: str, user: str, raw: str, *, exit_code: int = 0) -> MarsBridgeStatus:
    """Parse SSH status probe output into a structured snapshot."""
    status = MarsBridgeStatus(host=host, user=user, ssh_exit_code=exit_code)
    sections = raw.split("---")
    proc_blob = sections[0] if sections else raw
    port_blob = sections[1] if len(sections) > 1 else ""
    tmux_blob = sections[2] if len(sections) > 2 else ""

    for line in proc_blob.splitlines():
        line = line.strip()
        if not line or "pgrep" in line:
            continue
        m = re.match(r"^(\d+)\s+(.+)$", line)
        if m and "innate_mars_zmq_server" in line:
            status.pid = int(m.group(1))
            status.process_cmd = m.group(2).strip()
            break

    for line in port_blob.splitlines():
        for port in MARS_ZMQ_PORTS:
            if f":{port}" in line:
                status.listening_ports.add(port)

    tmux_text = tmux_blob.strip()
    if tmux_text == "---tmux-unavailable---":
        status.tmux_available = False
    elif tmux_text:
        status.tmux_available = True
        for line in tmux_text.splitlines():
            s = line.strip()
            if not s:
                continue
            if "hard error" in s.lower() and "shm" in s.lower():
                continue
            if s.startswith("[server-1]") or "innate_mars" in s.lower() or "waiting" in s.lower():
                status.ros_log_lines.append(s)
    return status


def fetch_bridge_status(host: str, user: str, password: str | None) -> MarsBridgeStatus:
    code, out, err = _ssh_capture(host, user, password, _remote_status_cmd())
    if err.strip():
        out = out + "\n" + err
    return parse_bridge_status_output(host, user, out, exit_code=code)


def _short_ros_message(line: str) -> str:
    if "]: " in line:
        return line.rsplit("]: ", 1)[-1].strip()
    line = line.strip()
    return line if len(line) <= 72 else line[:69] + "…"


def _compact_ports(status: MarsBridgeStatus) -> str:
    expected = set(MARS_ZMQ_PORTS)
    up = status.listening_ports & expected
    n = len(up)
    if n == len(expected):
        return style("4401–4404", fg="green")
    if n == 0:
        return style("down", fg="red")
    missing = ",".join(str(p) for p in sorted(expected - up))
    return style(f"{n}/4 (missing {missing})", fg="yellow")


def print_bridge_status(
    status: MarsBridgeStatus,
    *,
    profile: str | None = None,
    onboard_da3: bool = False,
    show_next_steps: bool = True,
) -> None:
    """Pretty-print bridge health for ``emet mars start`` / ``emet mars status``."""
    conn_ref = profile or status.host
    target = conn_ref if profile else f"{status.user}@{status.host}"

    if status.ready_for_stream:
        state = style("ready", fg="green", bold=True)
        glyph = style("●", fg="green")
    elif status.process_running:
        state = style("starting", fg="yellow", bold=True)
        glyph = style("●", fg="yellow")
    else:
        state = style("down", fg="red", bold=True)
        glyph = style("●", fg="red")

    parts = [f"{glyph} Mars bridge {state}", target]
    if status.pid:
        parts.append(f"pid {status.pid}")
    parts.append(f"ZMQ {_compact_ports(status)}")
    if onboard_da3:
        parts.append(style("DA3", fg="cyan"))

    log_line = status.headline_log()
    show_log = bool(
        log_line
        and (
            not status.ready_for_stream
            or "waiting" in log_line.lower()
            or "error" in log_line.lower()
        )
    )
    if show_log and log_line:
        msg = _short_ros_message(log_line)
        tone = "err" if "error" in log_line.lower() else "warn" if "waiting" in log_line.lower() else None
        if tone == "err":
            parts.append(style(msg, fg="red"))
        elif tone == "warn":
            parts.append(style(msg, fg="yellow"))
        else:
            parts.append(style(msg, dim=True))

    print(" · ".join(parts))

    if show_next_steps and status.ready_for_stream:
        print(style(f"  → emet stream --connection {conn_ref}", dim=True))
    elif show_next_steps and status.process_running:
        print(style(f"  → emet preview-cameras --source zmq --connection {conn_ref}", dim=True))


def start_bridge_on_robot(
    host: str,
    user: str,
    password: str | None,
    *,
    workspace: str = DEFAULT_INNATE_WORKSPACE,
    emet_dir: str = DEFAULT_EMET_DIR,
    onboard_da3: bool = False,
) -> None:
    """Start innate_mars_bridge inside innate-os tmux (Zenoh DDS env)."""
    launch_line = _remote_bridge_launch_cmd(workspace=workspace, emet_dir=emet_dir, onboard_da3=onboard_da3)
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
    note(f"Starting bridge on {user}@{host} ({TMUX_SESSION}:{TMUX_WINDOW})…")
    _ssh_run(host, user, password, remote, check=True)


def stop_bridge_on_robot(host: str, user: str, password: str | None) -> None:
    remote = (
        f"{_kill_bridge_remote()}; "
        f"tmux kill-window -t {TMUX_SESSION}:{TMUX_WINDOW} 2>/dev/null || true; "
        "echo bridge stopped"
    )
    note(f"Stopping bridge on {user}@{host}…")
    _ssh_run(host, user, password, remote, check=False)
    print(style("Bridge stopped.", fg="green"))


def bridge_status_on_robot(
    host: str,
    user: str,
    password: str | None,
    *,
    profile: str | None = None,
    onboard_da3: bool = False,
    show_next_steps: bool = True,
) -> MarsBridgeStatus:
    status = fetch_bridge_status(host, user, password)
    print_bridge_status(
        status,
        profile=profile,
        onboard_da3=onboard_da3,
        show_next_steps=show_next_steps,
    )
    return status


def mars_start(
    *,
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    connection_name: str | None = None,
    save_profile: bool = True,
    deploy: bool = False,
    preview: bool = False,
    onboard_da3: bool = False,
    wait_s: float = 20.0,
) -> None:
    host, user, password, workspace, emet_dir = resolve_mars_target(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
    )

    profile_name: str | None = None
    if save_profile and host:
        profile_name = save_connection(
            host=host,
            user=user,
            password=password,
            name=host,
            set_active=True,
            workspace=workspace,
            emet_dir=emet_dir,
            robot="innate_mars",
        )
        print(style(f"Saved profile '{profile_name}'.", fg="green", dim=True))

    if deploy:
        from emet.deploy import deploy as deploy_impl

        print()
        note(f"Deploying to {user}@{host}…")
        deploy_impl(
            host=host,
            user=user,
            password=password,
            workspace=workspace,
            emet_dir=emet_dir,
            start_bridge=False,
            with_da3=onboard_da3,
            root=_project_root(),
        )

    start_bridge_on_robot(host, user, password, workspace=workspace, emet_dir=emet_dir, onboard_da3=onboard_da3)

    if wait_s > 0:
        note(f"Waiting {wait_s:.0f}s for bridge startup…")
        time.sleep(wait_s)

    bridge_status_on_robot(
        host,
        user,
        password,
        profile=profile_name or connection_name or host,
        onboard_da3=onboard_da3,
        show_next_steps=not preview,
    )

    if preview:
        note("Opening camera preview…")
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
