# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Helpers for freeing network ports (e.g. before starting a server)."""

import subprocess


def kill_processes_on_port(port: int) -> bool:
    """Kill processes using the given port. Returns True if any were killed."""
    try:
        out = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if out.returncode != 0 or not out.stdout.strip():
        return False
    pids = [s for s in out.stdout.strip().split() if s.isdigit()]
    if not pids:
        return False
    for pid in pids:
        try:
            subprocess.run(["kill", pid], check=False, capture_output=True)
        except Exception:
            pass
    return True
