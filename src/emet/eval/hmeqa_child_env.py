# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run the HM-EQA H2H script with its explicit allowlisted environment."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from emet.eval.hmeqa_launch import build_hmeqa_child_env, hmeqa_run_config_from_env
from emet.utils.job_registry import validated_gpu_lock_fd


class HmeqaChildEnvironmentError(RuntimeError):
    """Raised when the H2H child cannot be re-executed safely."""


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = str(env.get(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def sanitized_hmeqa_child_env(
    ambient: Mapping[str, str],
) -> dict[str, str]:
    """Build the final script environment from frozen and operational inputs."""

    config = hmeqa_run_config_from_env(ambient)
    return build_hmeqa_child_env(
        config,
        base_env=ambient,
        resume=_bool_env(ambient, "RESUME", False),
        coverage_qids=str(ambient.get("COVERAGE_QIDS", "")),
        cooldown=int(ambient.get("EPISODE_COOLDOWN_SEC", "20")),
        crash_policy=str(ambient.get("NATIVE_CRASH_POLICY", "skip")),
        streak_abort=int(ambient.get("NATIVE_CRASH_STREAK_ABORT", "2")),
        egl_fail_abort=int(ambient.get("EGL_FAIL_ABORT", "2")),
        manifest_prepared=_bool_env(ambient, "EMET_HMEQA_MANIFEST_PREPARED", False),
        inherit_managed_context=True,
    )


def run_hmeqa_script(script: Path, args: Sequence[str]) -> int:
    """Run the canonical H2H script under a cancellable process-tree boundary."""

    from emet.utils.process_tree import popen_session, terminate_process_tree

    project_root = Path(__file__).resolve().parents[3]
    expected = (project_root / "scripts" / "run_hmeqa_agentic_h2h.sh").resolve()
    supplied = Path(script).resolve()
    if supplied != expected:
        raise HmeqaChildEnvironmentError(
            f"refusing to execute non-canonical H2H script {supplied}; expected {expected}"
        )
    env = sanitized_hmeqa_child_env(os.environ)
    env["EMET_HMEQA_ENV_SANITIZED"] = "1"
    lock_fd = validated_gpu_lock_fd()
    pass_fds = (lock_fd,) if lock_fd is not None else ()
    process = popen_session(
        [str(expected), *args],
        cwd=str(project_root),
        env=env,
        pass_fds=pass_fds,
    )
    try:
        return int(process.wait())
    except BaseException:
        terminate_process_tree(process, grace_s=20.0)
        raise
    finally:
        terminate_process_tree(process, grace_s=1.0)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        raise HmeqaChildEnvironmentError("expected the H2H script path")
    return run_hmeqa_script(Path(values[0]), values[1:])


if __name__ == "__main__":
    raise SystemExit(main())
