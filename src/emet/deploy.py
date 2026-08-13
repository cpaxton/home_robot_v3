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

"""Deploy emet_core + robot ROS2 bridge (Stretch or Innate Mars) via rsync and SSH."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from emet.utils.connection import get_active_connection, get_connection

DEFAULT_STRETCH_WORKSPACE = "~/ament_ws"
DEFAULT_MARS_WORKSPACE = "~/innate-os/ros2_ws"
DEFAULT_EMET_DIR = "~/emet"
DEFAULT_STRETCH_USER = "hello-robot"
DEFAULT_MARS_USER = "jetson1"

StartVia = Literal["mars_tmux", "nohup"]


@dataclass(frozen=True)
class DeployRobotSpec:
    """Per-robot deploy layout (bridge package, workspace, launch)."""

    robot_id: str
    bridge_pkg: str
    default_workspace: str
    default_user: str
    launch_file: str
    requirements_basename: str | None
    verify_imports: tuple[str, ...]
    start_via: StartVia


_DEPLOY_SPECS: dict[str, DeployRobotSpec] = {
    "stretch": DeployRobotSpec(
        robot_id="stretch",
        bridge_pkg="stretch_ros2_bridge",
        default_workspace=DEFAULT_STRETCH_WORKSPACE,
        default_user=DEFAULT_STRETCH_USER,
        launch_file="server.launch.py",
        requirements_basename="stretch_robot_requirements.txt",
        verify_imports=(
            "import stretch_ros2_bridge.ros.camera",
            "import emet.utils.image",
            "import emet.core.server",
        ),
        start_via="nohup",
    ),
    "innate_mars": DeployRobotSpec(
        robot_id="innate_mars",
        bridge_pkg="innate_mars_bridge",
        default_workspace=DEFAULT_MARS_WORKSPACE,
        default_user=DEFAULT_MARS_USER,
        launch_file="server.launch.py",
        requirements_basename="innate_mars_robot_requirements.txt",
        verify_imports=(
            "import innate_mars_bridge.ros.camera",
            "import emet.utils.image",
            "import emet.core.server",
        ),
        start_via="mars_tmux",
    ),
}

_ROBOT_ALIASES = {
    "stretch": "stretch",
    "hello_stretch": "stretch",
    "hello-stretch": "stretch",
    "innate_mars": "innate_mars",
    "mars": "innate_mars",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def normalize_deploy_robot(robot: str | None) -> str | None:
    """Map aliases to ``stretch`` / ``innate_mars``; return None if unset."""
    if robot is None:
        return None
    key = str(robot).strip().lower().replace("-", "_")
    if not key:
        return None
    if key in _ROBOT_ALIASES:
        return _ROBOT_ALIASES[key]
    if key in _DEPLOY_SPECS:
        return key
    raise SystemExit(
        f"Unsupported deploy robot {robot!r}. Use: stretch, innate_mars "
        f"(aliases: hello_stretch, mars)."
    )


def get_deploy_spec(robot: str) -> DeployRobotSpec:
    rid = normalize_deploy_robot(robot)
    if rid is None or rid not in _DEPLOY_SPECS:
        raise SystemExit(f"Unsupported deploy robot {robot!r}")
    return _DEPLOY_SPECS[rid]


def resolve_deploy_robot(
    robot: str | None = None,
    *,
    connection_name: str | None = None,
    host: str | None = None,
    workspace: str | None = None,
) -> str:
    """Resolve deploy target robot id from flag, connection profile, or workspace hint.

    Never guess Stretch for a saved connection that lacks ``robot:`` (legacy Mars
    profiles). When ``--host`` differs from the active profile host, ``--robot``
    is required so we do not push the wrong bridge package.
    """
    explicit = normalize_deploy_robot(robot)
    if explicit:
        return explicit
    conn = get_connection(connection_name) if connection_name else get_active_connection()
    if host and conn and not connection_name:
        conn_host = str(conn.get("host") or "").strip()
        if conn_host and str(host).strip() != conn_host:
            raise SystemExit(
                f"--host {host!r} differs from active connection host {conn_host!r}; "
                "pass --robot stretch|innate_mars or --connection NAME."
            )
    if conn:
        from_conn = normalize_deploy_robot(conn.get("robot"))
        if from_conn:
            return from_conn
        ws = str(conn.get("workspace") or workspace or "")
        if "innate" in ws.lower():
            return "innate_mars"
        if "ament" in ws.lower():
            return "stretch"
        raise SystemExit(
            "Connection profile has no robot: field. "
            "Pass --robot stretch|innate_mars or re-save with "
            "`emet connect save HOST --user USER --robot stretch|innate_mars`."
        )
    ws_hint = workspace or ""
    if "innate" in ws_hint.lower():
        return "innate_mars"
    # No connection: default matches Stretch ament_ws layout (CLI --workspace default).
    return "stretch"


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


def build_remote_bridge_import_verify_cmd(
    *,
    remote_emet: str,
    remote_ws: str,
    robot: str,
) -> str:
    """Return an SSH remote command that smoke-tests bridge + emet_core imports."""
    spec = get_deploy_spec(robot)
    py_paths = f"{remote_emet.rstrip('/')}/emet_core:{remote_emet.rstrip('/')}/src"
    ros_setup = f"source /opt/ros/humble/setup.bash && source {remote_ws.rstrip('/')}/install/setup.bash"
    py_snippet = "; ".join(spec.verify_imports)
    # Outer bash -lc uses single quotes; pass -c argument in double quotes (no nested '…').
    return f"bash -lc '{ros_setup} && export PYTHONPATH={py_paths}:$PYTHONPATH && python3 -c \"{py_snippet}\"'"


def build_stretch_bridge_start_remote_cmd(
    *,
    workspace: str = DEFAULT_STRETCH_WORKSPACE,
    emet_dir: str = DEFAULT_EMET_DIR,
    launch_file: str = "server.launch.py",
) -> str:
    """Remote shell to free ZMQ ports and nohup-start stretch_ros2_bridge.

    Uses ``fuser`` on ports (not ``pkill -f``) so the SSH command line cannot
    match itself — same pattern as Mars ``_kill_bridge_remote``.
    """
    remote_ws = workspace.rstrip("/")
    remote_emet = emet_dir.rstrip("/")
    py_paths = f"{remote_emet}/emet_core:{remote_emet}/src"
    launch = (
        f"source {remote_emet}/bridge_env.sh 2>/dev/null || "
        f"export PYTHONPATH={py_paths}:$PYTHONPATH; "
        f"cd {remote_ws} && source /opt/ros/humble/setup.bash && source install/setup.bash && "
        f"export PYTHONPATH={py_paths}:$PYTHONPATH && "
        f"ros2 launch stretch_ros2_bridge {launch_file}"
    )
    kill = "fuser -k 4401/tcp 4402/tcp 4403/tcp 4404/tcp 2>/dev/null || true"
    return (
        f"{kill}; "
        "sleep 1; "
        f"nohup bash -lc {launch!r} > /tmp/emet-stretch-bridge.log 2>&1 & "
        "echo stretch bridge started; "
        "sleep 1; "
        "pgrep -af 'stretch_ros2_bridge' || true"
    )


def start_stretch_bridge_on_robot(
    host: str,
    user: str,
    password: str | None,
    *,
    workspace: str = DEFAULT_STRETCH_WORKSPACE,
    emet_dir: str = DEFAULT_EMET_DIR,
    launch_file: str = "server.launch.py",
) -> None:
    """Start stretch_ros2_bridge via nohup (native ament_ws path; not Docker)."""
    remote = build_stretch_bridge_start_remote_cmd(
        workspace=workspace,
        emet_dir=emet_dir,
        launch_file=launch_file,
    )
    print(f"Starting stretch_ros2_bridge on {user}@{host} (nohup → /tmp/emet-stretch-bridge.log)…")
    _ssh_run(host, user, password, remote, check=True)


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
    workspace: str = DEFAULT_STRETCH_WORKSPACE,
    emet_dir: str = DEFAULT_EMET_DIR,
    start_bridge: bool = False,
    with_da3: bool = False,
    robot: str | None = None,
    root: Path | None = None,
) -> None:
    """Sync emet_core and the robot's ROS2 bridge, then colcon build.

    Supports **stretch** (``stretch_ros2_bridge`` → ``~/ament_ws``) and
    **innate_mars** (``innate_mars_bridge`` → ``~/innate-os/ros2_ws``).

    Args:
        host: Robot host (IP or hostname). If None, use active connection.
        user: SSH user. If None, use connection or the robot's default user.
        password: SSH password. If None, use connection or EMET_ROBOT_PASSWORD or SSH key.
        connection_name: Use this saved connection by name instead of active.
        workspace: Remote ROS2 workspace. Default ``~/ament_ws``; Mars profiles override.
        emet_dir: Remote path for emet_core (e.g. ~/emet).
        start_bridge: If True, launch the bridge after deploy (Mars: innate-os tmux; Stretch: nohup).
        with_da3: If True (Mars only), sync emet perception + install torch / depth-anything-3.
        robot: ``stretch`` or ``innate_mars`` (aliases: hello_stretch, mars). Resolved from
            connection profile when omitted.
        root: Project root (default: repo root containing ``src/emet_core`` and bridge packages).
    """
    root = root or _project_root()
    robot_id = resolve_deploy_robot(
        robot,
        connection_name=connection_name,
        host=host,
        workspace=workspace,
    )
    spec = get_deploy_spec(robot_id)

    conn = None
    if connection_name:
        conn = get_connection(connection_name)
    elif not host:
        conn = get_active_connection()
    if conn:
        host = host or conn.get("host")
        user = user or conn.get("user")
        password = password if password is not None else conn.get("password")
        if workspace == DEFAULT_STRETCH_WORKSPACE and conn.get("workspace"):
            workspace = str(conn["workspace"])
        if emet_dir == DEFAULT_EMET_DIR and conn.get("emet_dir"):
            emet_dir = str(conn["emet_dir"])
    if workspace == DEFAULT_STRETCH_WORKSPACE and robot_id == "innate_mars":
        workspace = spec.default_workspace
    if not host:
        raise SystemExit(
            "No host. Use --host, or run: emet connect save <host> --user USER --robot stretch|innate_mars"
        )
    user = user or spec.default_user
    password = password or os.environ.get("EMET_ROBOT_PASSWORD")
    use_paramiko = bool(password and not _has_sshpass())

    if with_da3 and robot_id != "innate_mars":
        raise SystemExit("--with-da3 / onboard DA3 is only supported for --robot innate_mars")

    emet_core_src = root / "src" / "emet_core"
    bridge_src = root / "src" / spec.bridge_pkg
    if not emet_core_src.is_dir():
        raise SystemExit(f"emet_core not found at {emet_core_src}")
    if not bridge_src.is_dir():
        raise SystemExit(f"{spec.bridge_pkg} not found at {bridge_src}")

    remote_emet = emet_dir.rstrip("/")
    remote_ws = workspace.rstrip("/")
    remote_emet_core = f"{remote_emet}/emet_core"
    remote_ws_src = f"{remote_ws}/src"
    remote_bridge = f"{remote_ws_src}/{spec.bridge_pkg}"
    ros_setup = f"source /opt/ros/humble/setup.bash && source {remote_ws}/install/setup.bash"
    robot_reqs = (
        root / "configs" / "robots" / spec.requirements_basename
        if spec.requirements_basename
        else None
    )
    da3_reqs = root / "configs" / "robots" / "innate_mars_da3_requirements.txt"
    emet_src = root / "src" / "emet"

    print(f"Deploy robot={robot_id} bridge={spec.bridge_pkg} → {user}@{host}")
    print(f"Syncing emet_core → {host}:{remote_emet_core} (rsync --delete)...")
    _rsync_to_robot(emet_core_src, remote_emet_core, host, user, password, use_paramiko)
    print(f"Syncing {spec.bridge_pkg} → {host}:{remote_bridge}...")
    _ssh_run(host, user, password, f"mkdir -p {remote_ws_src}", use_paramiko)
    _rsync_to_robot(bridge_src, remote_bridge, host, user, password, use_paramiko)

    if robot_reqs is not None and robot_reqs.is_file():
        print("Installing bridge Python deps on robot...")
        remote_reqs = f"{remote_emet}/{robot_reqs.name}"
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

    print(f"Building ROS2 workspace on robot (colcon build --packages-select {spec.bridge_pkg})...")
    build_cmd = (
        f"bash -lc '{ros_setup} && cd {remote_ws} && "
        f"colcon build --packages-select {spec.bridge_pkg}'"
    )
    _ssh_run(host, user, password, build_cmd, use_paramiko)

    print("Verifying bridge imports emet_core on robot...")
    verify_cmd = build_remote_bridge_import_verify_cmd(
        remote_emet=remote_emet,
        remote_ws=remote_ws,
        robot=robot_id,
    )
    _ssh_run(host, user, password, verify_cmd, use_paramiko)

    launch_hint = (
        f"source {remote_emet}/bridge_env.sh && cd {remote_ws} && "
        f"source /opt/ros/humble/setup.bash && source install/setup.bash && "
        f"ros2 launch {spec.bridge_pkg} {spec.launch_file}"
    )
    if start_bridge:
        if spec.start_via == "mars_tmux":
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
            start_stretch_bridge_on_robot(
                host,
                user,
                password,
                workspace=remote_ws,
                emet_dir=remote_emet,
                launch_file=spec.launch_file,
            )
            print("Stretch bridge started (nohup). Log: /tmp/emet-stretch-bridge.log · emet view-bridge")
    else:
        print("To start the bridge on the robot:")
        if robot_id == "innate_mars":
            print(f"  uv run emet mars start --connection {connection_name or host}")
            print(f"  # or SSH: {launch_hint}")
        else:
            print("  uv run emet deploy --robot stretch --start-bridge")
            print("  # or Docker on robot: ./scripts/run_stretch_ai_ros2_bridge_server.sh")
            print(f"  # or SSH: {launch_hint}")
