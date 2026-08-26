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
from emet.cli_cmds.molmospaces import register as register_molmospaces
from emet.cli_cmds.run_sync import register as register_run_sync
from emet.cli_cmds.serve import register as register_serve

__all__ = ["main", "_jobs_run_id_from_output"]


@click.group(
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


from emet.comm import comm_group  # noqa: E402

main.add_command(comm_group)

from emet.app.capture import main as _capture_app  # noqa: E402

_capture_app.short_help = "One ZMQ frame + metadata (shortcut: zmq_obs capture profile)"
main.add_command(_capture_app)

from emet.app.stream import main as _stream_app  # noqa: E402

_stream_app.short_help = "Live ZMQ → Rerun (shortcut: zmq_obs stream profile)"
main.add_command(_stream_app)

# Full Click options (not a thin wrapper) so `emet debug-da3-depth --help` lists all flags.
from emet.app.debug_da3_depth import main as _debug_da3_depth_app  # noqa: E402

_debug_da3_depth_app.short_help = "Live DA3 depth + point cloud from ZMQ (Rerun)"
main.add_command(_debug_da3_depth_app)

from emet.app.debug_lingbot_depth import main as _debug_lingbot_depth_app  # noqa: E402

_debug_lingbot_depth_app.short_help = "Live LingBot-Map depth + pose from ZMQ (Rerun)"
main.add_command(_debug_lingbot_depth_app)

from emet.app.export_sim_gt import main as _export_sim_gt_app  # noqa: E402

_export_sim_gt_app.short_help = "Export Robocasa sim GT objects (3D bounds + head 2D boxes)"
main.add_command(_export_sim_gt_app)

from emet.app.tune_graph_fusion import main as _tune_graph_fusion_app  # noqa: E402

_tune_graph_fusion_app.short_help = "Grid-search GraphObjectFusion vs GT + calibration frames"
main.add_command(_tune_graph_fusion_app)

from emet.app.eval_calibration import main as _eval_calibration_app  # noqa: E402

_eval_calibration_app.short_help = "Score calibration frames vs sim GT (spatial recall)"
main.add_command(_eval_calibration_app)

from emet.app.eval_dynagraph import main as _eval_dynagraph_app  # noqa: E402

_eval_dynagraph_app.short_help = "Unified Dynagraph episode eval (explore, graph, fusion, EQA)"
main.add_command(_eval_dynagraph_app)

from emet.app.eval_ovmm import ovmm_group as _ovmm_group  # noqa: E402
from emet.app.eval_sqa3d import eval_sqa3d_main as _eval_sqa3d_app  # noqa: E402
from emet.app.eval_sqa3d import sqa3d_group as _sqa3d_group  # noqa: E402

_eval_sqa3d_app.short_help = "Score SQA3D QA predictions (EM@1)"
main.add_command(_eval_sqa3d_app)
main.add_command(_sqa3d_group)
main.add_command(_ovmm_group)

from emet.app.eval_robovista import robovista_group as _robovista_group  # noqa: E402

_robovista_group.short_help = "RoboVista offline robot-centric MCQ-VQA"
main.add_command(_robovista_group)

if __name__ == "__main__":
    main()
