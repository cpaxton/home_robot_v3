#!/usr/bin/env python
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

"""Emet CLI — start simulations, run agents, sync deps, view logs, run tests."""

import os
import sys

import click

from emet.cli_cmds.bootstrap import (
    _CONTEXT_SETTINGS,
    _MAIN_EPILOG,
    _ensure_uv_project,
    _jobs_run_id_from_output,
    _project_root,
    _require_repo_venv_when_in_repo,
)
from emet.cli_cmds.connect import register as register_connect
from emet.cli_cmds.eval_status import register as register_eval_status
from emet.cli_cmds.habitat import register as register_habitat
from emet.cli_cmds.hmeqa import register as register_hmeqa
from emet.cli_cmds.jobs import register as register_jobs
from emet.cli_cmds.lazy_group import LazyClickGroup
from emet.cli_cmds.molmospaces import register as register_molmospaces
from emet.cli_cmds.run_sync import register as register_run_sync
from emet.cli_cmds.serve import register as register_serve

# Import these only when the user actually invokes them. Registering at
# import time pulled MuJoCo via export-sim-gt / eval-dynagraph and SIGSEGV'd
# ``emet jobs`` / ``emet eval`` right after a sim job released the GPU lock.
_LAZY_APP_COMMANDS: dict[str, tuple[str, str, str]] = {
    "capture": (
        "emet.app.capture",
        "main",
        "One ZMQ frame + metadata (shortcut: zmq_obs capture profile)",
    ),
    "stream": (
        "emet.app.stream",
        "main",
        "Live ZMQ → Rerun (shortcut: zmq_obs stream profile)",
    ),
    "debug-da3-depth": (
        "emet.app.debug_da3_depth",
        "main",
        "Live DA3 depth + point cloud from ZMQ (Rerun)",
    ),
    "debug-lingbot-depth": (
        "emet.app.debug_lingbot_depth",
        "main",
        "Live LingBot-Map depth + pose from ZMQ (Rerun)",
    ),
    "export-sim-gt": (
        "emet.app.export_sim_gt",
        "main",
        "Export Robocasa sim GT objects (3D bounds + head 2D boxes)",
    ),
    "tune-graph-fusion": (
        "emet.app.tune_graph_fusion",
        "main",
        "Grid-search GraphObjectFusion vs GT + calibration frames",
    ),
    "eval-calibration": (
        "emet.app.eval_calibration",
        "main",
        "Score calibration frames vs sim GT (spatial recall)",
    ),
    "eval-dynagraph": (
        "emet.app.eval_dynagraph",
        "main",
        "Unified Dynagraph episode eval (explore, graph, fusion, EQA)",
    ),
    "eval-sqa3d": (
        "emet.app.eval_sqa3d",
        "eval_sqa3d_main",
        "Score SQA3D QA predictions (EM@1)",
    ),
    "sqa3d": (
        "emet.app.eval_sqa3d",
        "sqa3d_group",
        "SQA3D embodied QA (ScanNet / DynaMem / Dynagraph)",
    ),
    "ovmm": (
        "emet.app.eval_ovmm",
        "ovmm_group",
        "OVMM find / full / sweep paper paths",
    ),
    "robovista": (
        "emet.app.eval_robovista",
        "robovista_group",
        "RoboVista offline robot-centric MCQ-VQA",
    ),
}

__all__ = ["main", "_jobs_run_id_from_output"]


@click.group(
    cls=LazyClickGroup,
    lazy_subcommands=_LAZY_APP_COMMANDS,
    context_settings=_CONTEXT_SETTINGS,
    epilog=_MAIN_EPILOG,
)
@click.version_option(version="0.3.3", prog_name="emet")
def main() -> None:
    """Emet — Embodied Multi-robot Environment Toolkit.

    Start simulations, run robot agents, sync dependencies, view logs, and run tests.
    When run from inside the project directory and uv is installed, emet automatically
    uses the project environment (as if you had run uv run emet ...).
    """
    _ensure_uv_project()
    _require_repo_venv_when_in_repo()
    active = os.environ.get("EMET_ACTIVE_REPO")
    if active:
        click.secho(
            f"emet: using checkout {active} ({_project_root()})",
            fg="green",
            err=True,
        )


register_serve(main)
register_molmospaces(main)
register_habitat(main)
register_jobs(main)
register_eval_status(main)
register_hmeqa(main)
register_connect(main)
register_run_sync(main)


@main.command("install-completion", short_help="Print shell completion script")
@click.option(
    "--shell",
    "-s",
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    default=None,
    help="Shell (default: auto-detect from SHELL).",
)
def install_completion(shell: str | None) -> None:
    """Print shell completion script for bash, zsh, or fish.

    Add to your shell config so that 'emet' and subcommands tab-complete:

      # Bash: add to ~/.bashrc
      eval \"$(emet install-completion --shell bash)\"

      # Zsh: add to ~/.zshrc
      eval \"$(emet install-completion --shell zsh)\"

      # Fish: add to ~/.config/fish/config.fish
      emet install-completion --shell fish | source
    """
    from click.shell_completion import get_completion_class

    if shell is None:
        shell_env = os.environ.get("SHELL", "")
        if "fish" in shell_env:
            shell = "fish"
        elif "zsh" in shell_env:
            shell = "zsh"
        else:
            shell = "bash"
    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        click.echo(f"Completion not supported for {shell}", err=True)
        sys.exit(1)
    comp = comp_cls(main, {}, "emet", "_EMET_COMPLETE")
    click.echo(comp.source())


if __name__ == "__main__":
    main()
