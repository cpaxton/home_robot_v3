#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
#
# Load a memory directory and display it in Rerun (unified memory view).
#
# Usage:
#   uv run python -m emet.app.show_memory [PATH]
#   emet show [PATH]

import sys
from pathlib import Path

import click

from emet.memory.format import is_memory_directory, load_memory
from emet.visualization.rerun import RerunVisualizer


@click.command()
@click.argument(
    "path",
    type=click.Path(path_type=Path),
    default=Path("saved_memory"),
    required=False,
)
@click.option(
    "--open-browser",
    is_flag=True,
    help="Open browser to Rerun web viewer.",
)
def main(path: Path, open_browser: bool) -> None:
    """Open a saved memory directory in Rerun.

    PATH is a directory in the common memory format (manifest.json, point_cloud.npz, etc.).
    Default: saved_memory.
    """
    path = path.resolve()
    if not path.is_dir():
        click.echo(f"Not a directory: {path}", err=True)
        sys.exit(1)
    if not is_memory_directory(str(path)):
        click.echo(
            f"Not a memory directory (no manifest.json): {path}\n"
            "Use a path saved by 'emet run create-and-print-memory' or a backend save().",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Loading memory from {path} ...")
    state = load_memory(str(path))
    num_frames = len(state.frames)
    click.echo("Starting Rerun viewer ...")
    viz = RerunVisualizer(
        display_robot_mesh=False,
        spawn_gui=True,
        open_browser=open_browser,
        memory_view=True,
        num_frames=num_frames,
    )
    viz.log_memory_state(state, static=True)
    viz.send_memory_blueprint()
    click.echo("Memory logged to Rerun. Close the Rerun window or press Enter here to exit.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
