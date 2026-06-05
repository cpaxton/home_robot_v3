# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Discover and invoke the ``emet-habitat`` wrapper (``.venv-habitat``)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _python_can_import_habitat(py: Path) -> bool:
    try:
        r = subprocess.run(
            [
                str(py),
                "-c",
                "import habitat_sim; import emet_habitat; "
                "from emet.habitat.config import default_habitat_eqa_data_dir",
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def build_habitat_wrapper_command(args: list[str]) -> list[str] | None:
    """Return argv to run ``emet-habitat``, or None if the wrapper venv is missing."""
    root = _project_root()
    local_py = root / ".venv-habitat" / "bin" / "python"
    local_exe = root / ".venv-habitat" / "bin" / "emet-habitat"

    if local_py.exists() and _python_can_import_habitat(local_py):
        if local_exe.exists():
            return [str(local_exe)] + args
        return [str(local_py), "-m", "emet_habitat.cli"] + args

    env_py = os.environ.get("HABITAT_PYTHON", "").strip()
    if env_py:
        p = Path(env_py).resolve()
        if p.exists() and _python_can_import_habitat(p):
            exe = p.parent / "emet-habitat"
            if exe.exists():
                return [str(exe)] + args
            return [str(p), "-m", "emet_habitat.cli"] + args

    return None


def ensure_habitat_eqa_data_dir_env(env: dict[str, str] | None = None) -> Path:
    """Set ``HABITAT_EQA_DATA_DIR`` default if unset; return resolved path."""
    from emet.habitat.config import default_habitat_eqa_data_dir

    target = env if env is not None else os.environ
    if not target.get("HABITAT_EQA_DATA_DIR", "").strip():
        target["HABITAT_EQA_DATA_DIR"] = str(default_habitat_eqa_data_dir())
    return Path(target["HABITAT_EQA_DATA_DIR"]).expanduser()
