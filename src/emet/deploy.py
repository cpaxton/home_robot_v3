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

"""Deploy emet_core and innate_mars_bridge to a robot via rsync and SSH."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from emet.utils.connection import get_active_connection, get_connection


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, env=env or os.environ)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r


def _ssh_target(host: str, user: str, _password: str | None) -> str:
    return f"{user}@{host}"


def _has_sshpass() -> bool:
    return shutil.which("sshpass") is not None


def _rsync_to_robot(
    local_path: Path,
    remote_path: str,
    host: str,
    user: str,
    password: str | None,
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
    password: str | None,
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


def build_remote_bridge_import_verify_cmd(*, remote_emet: str, remote_ws: str) -> str:
    """Return an SSH remote command that smoke-tests bridge + emet_core imports."""
    py_paths = f"{remote_emet.rstrip('/')}/emet_core:{remote_emet.rstrip('/')}/src"
    ros_setup = f"source /opt/ros/humble/setup.bash && source {remote_ws.rstrip('/')}/install/setup.bash"
    py_snippet = "import innate_mars_bridge.ros.camera; import emet.utils.image; import emet.core.server"
    # Outer bash -lc uses single quotes; pass -c argument in double quotes (no nested '…').
    return f"bash -lc '{ros_setup} && export PYTHONPATH={py_paths}:$PYTHONPATH && python3 -c \"{py_snippet}\"'"


def _ssh_run(
    host: str,
    user: str,
    password: str | None,
    remote_cmd: str,
    use_paramiko: bool | None = None,
    *,
    check: bool = True,
) -> int:
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
            if exit_status != 0 and check:
                sys.exit(exit_status)
            return exit_status
        finally:
            client.close()
        return 0
    target = _ssh_target(host, user, password)
    full_cmd = ["ssh", target, remote_cmd]
    if password:
        full_cmd = ["sshpass", "-p", password] + full_cmd
    r = subprocess.run(full_cmd)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode


def _ssh_capture(
    host: str,
    user: str,
    password: str | None,
    remote_cmd: str,
    use_paramiko: bool | None = None,
) -> tuple[int, str, str]:
    """Run remote command and return (exit_code, stdout, stderr) without printing."""
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
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            return exit_status, out, err
        finally:
            client.close()
    target = _ssh_target(host, user, password)
    full_cmd = ["ssh", target, remote_cmd]
    if password:
        full_cmd = ["sshpass", "-p", password] + full_cmd
    r = subprocess.run(full_cmd, capture_output=True, text=True)
    return r.returncode, r.stdout or "", r.stderr or ""


def deploy(
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    connection_name: str | None = None,
    workspace: str = "~/ament_ws",
    emet_dir: str = "~/emet",
    start_bridge: bool = False,
    with_da3: bool = False,
    root: Path | None = None,
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
        with_da3: If True, sync emet perception (DA3) to the robot and install torch + depth-anything-3.
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
        if workspace == "~/ament_ws" and conn.get("workspace"):
            workspace = str(conn["workspace"])
        if emet_dir == "~/emet" and conn.get("emet_dir"):
            emet_dir = str(conn["emet_dir"])
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

    remote_emet = emet_dir.rstrip("/")
    remote_ws = workspace.rstrip("/")
    remote_emet_core = f"{remote_emet}/emet_core"
    remote_ws_src = f"{remote_ws}/src"
    ros_setup = f"source /opt/ros/humble/setup.bash && source {remote_ws}/install/setup.bash"
    robot_reqs = root / "configs" / "robots" / "innate_mars_robot_requirements.txt"
    da3_reqs = root / "configs" / "robots" / "innate_mars_da3_requirements.txt"
    emet_src = root / "src" / "emet"

    print(f"Syncing emet_core → {host}:{remote_emet_core} (rsync --delete)...")
    _rsync_to_robot(emet_core_src, remote_emet_core, host, user, password, use_paramiko)
    print(f"Syncing innate_mars_bridge → {host}:{remote_ws_src}/innate_mars_bridge...")
    _ssh_run(host, user, password, f"mkdir -p {remote_ws_src}", use_paramiko)
    _rsync_to_robot(bridge_src, f"{remote_ws_src}/innate_mars_bridge", host, user, password, use_paramiko)

    if robot_reqs.is_file():
        print("Installing bridge Python deps on robot...")
        remote_reqs = f"{remote_emet}/innate_mars_robot_requirements.txt"
        target = _ssh_target(host, user, password)
        req_cmd = ["rsync", "-az", str(robot_reqs), f"{target}:{remote_reqs}"]
        if password:
            req_cmd = ["sshpass", "-p", password] + req_cmd
        _run(req_cmd)
        _ssh_run(
            host,
            user,
            password,
            f"bash -lc 'python3 -m pip install --user -r {remote_reqs}'",
            use_paramiko,
        )

    if with_da3:
        if not emet_src.is_dir():
            raise SystemExit(f"emet package not found at {emet_src} (required for --with-da3)")
        print("Syncing emet perception (onboard DA3) to robot...")
        remote_emet_src = f"{remote_emet}/src/emet"
        for sub in ("perception", "utils"):
            local_sub = emet_src / sub
            if local_sub.is_dir():
                _rsync_to_robot(local_sub, f"{remote_emet_src}/{sub}", host, user, password, use_paramiko)
        init_py = emet_src / "__init__.py"
        if init_py.is_file():
            _ssh_run(host, user, password, f"mkdir -p {remote_emet_src}", use_paramiko)
            _rsync_to_robot(init_py, remote_emet_src, host, user, password, use_paramiko)
        if da3_reqs.is_file():
            print("Installing onboard DA3 Python deps on robot (torch + depth-anything-3; may take a while)...")
            remote_da3_reqs = f"{remote_emet}/innate_mars_da3_requirements.txt"
            target = _ssh_target(host, user, password)
            req_cmd = ["rsync", "-az", str(da3_reqs), f"{target}:{remote_da3_reqs}"]
            if password:
                req_cmd = ["sshpass", "-p", password] + req_cmd
            _run(req_cmd)
            _ssh_run(
                host,
                user,
                password,
                f"bash -lc 'python3 -m pip install --user -r {remote_da3_reqs}'",
                use_paramiko,
            )

    py_paths = f"{remote_emet_core}:{remote_emet}/src"
    _ssh_run(
        host,
        user,
        password,
        (
            f"bash -lc 'mkdir -p {remote_emet} && "
            f'printf %s\\n "export PYTHONPATH={py_paths}:\\$PYTHONPATH" '
            f"> {remote_emet}/bridge_env.sh'"
        ),
        use_paramiko,
    )

    print("Building ROS2 workspace on robot (colcon build)...")
    build_cmd = f"bash -lc '{ros_setup} && cd {remote_ws} && colcon build --packages-select innate_mars_bridge'"
    _ssh_run(host, user, password, build_cmd, use_paramiko)

    print("Verifying bridge imports emet_core on robot...")
    verify_cmd = build_remote_bridge_import_verify_cmd(remote_emet=remote_emet, remote_ws=remote_ws)
    _ssh_run(host, user, password, verify_cmd, use_paramiko)

    if start_bridge:
        from emet.mars import start_bridge_on_robot

        start_bridge_on_robot(
            host,
            user,
            password,
            workspace=remote_ws,
            emet_dir=remote_emet,
        )
        print("Bridge started in tmux ros_nodes:emet-bridge. View: emet view-bridge")
    else:
        print("To start the bridge on the robot, SSH in and run:")
        print(
            f"  source {remote_emet}/bridge_env.sh && cd {remote_ws} && "
            f"source /opt/ros/humble/setup.bash && source install/setup.bash && "
            f"ros2 launch innate_mars_bridge server.launch.py"
        )
