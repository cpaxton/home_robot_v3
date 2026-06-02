# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Install optional system CLI tools declared in pyproject.toml [tool.emet.system-packages]."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_SYSTEM_PACKAGES_SECTION = "[tool.emet.system-packages]"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_system_packages() -> dict[str, str]:
    """Parse ``[tool.emet.system-packages]`` from pyproject.toml (stdlib-only, Py3.10+)."""
    pyproject = _project_root() / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    out: dict[str, str] = {}
    in_section = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == _SYSTEM_PACKAGES_SECTION:
            in_section = True
            continue
        if not in_section:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


_GH_RELEASE_VERSION = "2.86.0"


def _install_gh_binary(*, dest_dir: Path) -> int:
    """Install official GitHub CLI release tarball into *dest_dir* (no sudo)."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
    name = f"gh_{_GH_RELEASE_VERSION}_linux_{arch}"
    url = f"https://github.com/cli/cli/releases/download/v{_GH_RELEASE_VERSION}/{name}.tar.gz"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "gh"
    print(f"Downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except OSError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.tar.gz"
        archive.write_bytes(data)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp)
        src = Path(tmp) / name / "bin" / "gh"
        if not src.is_file():
            print(f"Expected binary missing in archive: {src}", file=sys.stderr)
            return 1
        shutil.copy2(src, dest)
        dest.chmod(0o755)
    print(f'Installed gh to {dest}. Add to PATH: export PATH="{dest_dir}:$PATH"')
    print("Authenticate: gh auth login")
    return 0


def ensure_apt_package(
    tool_key: str,
    *,
    non_interactive: bool = False,
) -> int:
    """Install *tool_key* via apt if missing. *tool_key* maps to an apt package name in pyproject."""
    packages = _load_system_packages()
    apt_name = packages.get(tool_key)
    if not apt_name:
        known = ", ".join(sorted(packages)) or "(none)"
        print(f"Unknown system tool {tool_key!r}. Declared in pyproject: {known}", file=sys.stderr)
        return 1

    if shutil.which(tool_key):
        try:
            ver = subprocess.check_output([tool_key, "--version"], text=True, stderr=subprocess.STDOUT).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            ver = "(installed)"
        print(f"{tool_key} already on PATH: {ver}")
        return 0

    if shutil.which("apt-get") is None:
        print(
            f"{tool_key} not found and apt-get is unavailable. "
            f"Install the {apt_name!r} package for your OS (see https://cli.github.com/).",
            file=sys.stderr,
        )
        return 1

    cmd = ["sudo", "apt-get", "install", "-y", apt_name]
    if not non_interactive:
        print(f"Will run: {' '.join(cmd)}")
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    print(f"Installing {apt_name!r} ({tool_key})...")
    r = subprocess.call(["sudo", "apt-get", "update"])
    if r != 0:
        print("apt-get update failed (continuing anyway).", file=sys.stderr)
    r = subprocess.call(cmd)
    if r != 0 and tool_key == "gh":
        print("apt install failed or sudo unavailable; trying user-local binary install...", file=sys.stderr)
        local_bin = Path.home() / ".local" / "bin"
        r = _install_gh_binary(dest_dir=local_bin)
        if r == 0:
            os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        return r if r != 0 else (0 if shutil.which(tool_key) else 1)
    if r != 0:
        print(f"apt-get install {apt_name!r} failed with exit {r}.", file=sys.stderr)
        return r
    if shutil.which(tool_key):
        print(f"Installed {tool_key}. Run: {tool_key} auth login")
        return 0
    print(f"Install finished but {tool_key!r} is still not on PATH.", file=sys.stderr)
    return 1


def list_system_packages() -> dict[str, str]:
    return _load_system_packages()
