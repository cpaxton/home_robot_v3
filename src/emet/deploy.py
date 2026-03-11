# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Deploy emet_core and innate_mars_bridge to a robot via rsync and SSH."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from emet.utils.connection import get_active_connection, get_connection


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, env=env or os.environ)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r


def _ssh_target(host: str, user: str, _password: Optional[str]) -> str:
    return f"{user}@{host}"


def _has_sshpass() -> bool:
    return shutil.which("sshpass") is not None


def _rsync_to_robot(
    local_path: Path,
    remote_path: str,
    host: str,
    user: str,
    password: Optional[str],
    use_paramiko: bool,
) -> None:
    if use_paramiko:
        _paramiko_upload(host, user, password, local_path, remote_path)
        return
    target = _ssh_target(host, user, password)
    dest = f"{target}:{remote_path}"
    _ssh_run(host, user, password, f"mkdir -p {remote_path}", use_paramiko=False)
    cmd = ["rsync", "-az", "--delete", str(local_path) + "/", dest + "/"]
    if password:
        cmd = ["sshpass", "-p", password] + cmd
    _run(cmd)


def _paramiko_upload(
    host: str,
    user: str,
    password: Optional[str],
    local_path: Path,
    remote_path: str,
) -> None:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password or None, timeout=30)
    try:
        sftp = client.open_sftp()
        try:
            _sftp_put_r(sftp, local_path, remote_path)
        finally:
            sftp.close()
    finally:
        client.close()


def _sftp_makedirs(sftp: Any, remote_path: str) -> None:
    """Create remote directory and all parents (like mkdir -p)."""
    remote_path = remote_path.rstrip("/")
    if not remote_path:
        return
    parts = [p for p in remote_path.split("/") if p and p != "."]
    if not parts:
        return
    prefix = "/" + parts[0] if remote_path.startswith("/") else parts[0]
    for p in parts[1:]:
        try:
            sftp.stat(prefix)
        except FileNotFoundError:
            try:
                sftp.mkdir(prefix)
            except OSError:
                pass
        prefix = f"{prefix}/{p}"
    try:
        sftp.stat(prefix)
    except FileNotFoundError:
        try:
            sftp.mkdir(prefix)
        except OSError:
            pass


def _sftp_put_r(sftp: Any, local: Path, remote: str) -> None:
    _sftp_makedirs(sftp, remote)
    for entry in local.iterdir():
        local_entry = local / entry.name
        remote_entry = f"{remote.rstrip('/')}/{entry.name}"
        if local_entry.is_dir():
            _sftp_put_r(sftp, local_entry, remote_entry)
        else:
            _sftp_makedirs(sftp, os.path.dirname(remote_entry))
            sftp.put(str(local_entry), remote_entry)


def _ssh_run(
    host: str,
    user: str,
    password: Optional[str],
    remote_cmd: str,
    use_paramiko: Optional[bool] = None,
) -> None:
    if use_paramiko is None:
        use_paramiko = bool(password and not _has_sshpass())
    if use_paramiko:
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=user, password=password or None, timeout=30)
        try:
            _, stdout, stderr = client.exec_command(remote_cmd)
            exit_status = stdout.channel.recv_exit_status()
            out = stdout.read().decode()
            err = stderr.read().decode()
            if out:
                print(out, end="")
            if err:
                print(err, end="", file=sys.stderr)
            if exit_status != 0:
                sys.exit(exit_status)
        finally:
            client.close()
        return
    target = _ssh_target(host, user, password)
    full_cmd = ["ssh", target, remote_cmd]
    if password:
        full_cmd = ["sshpass", "-p", password] + full_cmd
    _run(full_cmd)


def deploy(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    connection_name: Optional[str] = None,
    workspace: str = "~/ament_ws",
    emet_dir: str = "~/emet",
    start_bridge: bool = False,
    root: Optional[Path] = None,
) -> None:
    """Sync emet_core and innate_mars_bridge to the robot and install.

    Args:
        host: Robot host (IP or hostname). If None, use active connection.
        user: SSH user. If None, use active connection or "root".
        password: SSH password. If None, use connection or EMET_ROBOT_PASSWORD or SSH key.
        connection_name: Use this saved connection by name instead of active.
        workspace: Remote path to ROS2 workspace (e.g. ~/ament_ws).
        emet_dir: Remote path for emet_core (e.g. ~/emet).
        start_bridge: If True, run ros2 launch innate_mars_bridge server.launch.py after deploy.
        root: Project root (default: repo root containing src/emet_core and src/innate_mars_bridge).
    """
    root = root or _project_root()
    conn = None
    if connection_name:
        conn = get_connection(connection_name)
    elif not host:
        conn = get_active_connection()
    if conn:
        host = host or conn.get("host")
        user = user or conn.get("user", "root")
        password = password if password is not None else conn.get("password")
    if not host:
        raise SystemExit("No host. Use --host, or run: emet connect save <host> [--user USER]")
    user = user or "root"
    password = password or os.environ.get("EMET_ROBOT_PASSWORD")
    use_paramiko = bool(password and not _has_sshpass())

    emet_core_src = root / "src" / "emet_core"
    bridge_src = root / "src" / "innate_mars_bridge"
    if not emet_core_src.is_dir():
        raise SystemExit(f"emet_core not found at {emet_core_src}")
    if not bridge_src.is_dir():
        raise SystemExit(f"innate_mars_bridge not found at {bridge_src}")

    remote_emet = os.path.expanduser(emet_dir)
    remote_ws = os.path.expanduser(workspace)
    remote_emet_core = os.path.join(remote_emet, "emet_core")
    remote_ws_src = os.path.join(remote_ws, "src")

    print("Syncing emet_core to robot...")
    _rsync_to_robot(emet_core_src, remote_emet_core, host, user, password, use_paramiko)
    print("Syncing innate_mars_bridge to robot...")
    _ssh_run(host, user, password, f"mkdir -p {remote_ws_src}", use_paramiko)
    _rsync_to_robot(bridge_src, os.path.join(remote_ws_src, "innate_mars_bridge"), host, user, password, use_paramiko)

    print("Installing emet_core on robot (pip install -e)...")
    _ssh_run(host, user, password, f"pip install -e {remote_emet_core}", use_paramiko)

    print("Building ROS2 workspace on robot (colcon build)...")
    _ssh_run(host, user, password, f"cd {remote_ws} && colcon build --packages-select innate_mars_bridge", use_paramiko)

    if start_bridge:
        print("Starting bridge on robot in background...")
        _ssh_run(
            host,
            user,
            password,
            f"cd {remote_ws} && source install/setup.bash && nohup ros2 launch innate_mars_bridge server.launch.py > /tmp/innate_mars_bridge.log 2>&1 &",
            use_paramiko,
        )
        print("Bridge started. To view: emet view-bridge")
    else:
        print("To start the bridge on the robot, SSH in and run:")
        print(f"  cd {remote_ws} && source install/setup.bash && ros2 launch innate_mars_bridge server.launch.py")
