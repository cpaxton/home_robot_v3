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

"""Optional MolmoBot policy server delegation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_molmobot_python() -> Path | None:
    env = os.environ.get("MOLMOBOT_PYTHON", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    venv_py = _project_root() / ".venv-molmobot" / "bin" / "python"
    if venv_py.is_file():
        return venv_py
    which = shutil.which("python")
    return Path(which) if which else None


def serve_policy(argv: list[str]) -> int:
    """Run upstream ``launch_scripts/serve_molmo.py`` if MolmoBot is installed."""
    py = resolve_molmobot_python()
    if py is None:
        print(
            "MolmoBot python not found. Clone https://github.com/allenai/MolmoBot, "
            "create .venv-molmobot, or set MOLMOBOT_PYTHON.",
            file=sys.stderr,
        )
        return 1
    molmobot_root = os.environ.get("MOLMOBOT_ROOT", "").strip()
    if not molmobot_root:
        print(
            "Set MOLMOBOT_ROOT to the MolmoBot/MolmoBot checkout containing launch_scripts/serve_molmo.py",
            file=sys.stderr,
        )
        return 1
    script = Path(molmobot_root) / "launch_scripts" / "serve_molmo.py"
    if not script.is_file():
        print(f"Missing {script}", file=sys.stderr)
        return 1
    cmd = [str(py), str(script), *argv]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", molmobot_root)
    return subprocess.call(cmd, cwd=molmobot_root, env=env)
