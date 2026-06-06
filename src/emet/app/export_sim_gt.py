# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI: export Robocasa sim ground-truth object JSON (3D bounds + head 2D boxes)."""

from __future__ import annotations

import click

from emet.simulation.mujoco_gt_objects import export_robocasa_gt_scene


@click.command("export-sim-gt")
@click.option("--robot", default="innate_mars", show_default=True, help="Robot merged into kitchen")
@click.option("--seed", default=0, type=int, show_default=True, help="NumPy / placement RNG seed")
@click.option("--layout", default=1, type=int, show_default=True, help="Robocasa kitchen layout id")
@click.option("--style", default=1, type=int, show_default=True, help="Robocasa kitchen style id")
@click.option(
    "--task",
    default="PickPlaceCounterToCabinet",
    show_default=True,
    help="Robocasa task used to spawn objects",
)
@click.option("--out", "-o", "out_path", required=True, type=click.Path(), help="Output .json path")
@click.option(
    "--no-project-head-bbox",
    is_flag=True,
    help="Skip 2D head-camera bbox projection (3D bounds only)",
)
def main(
    robot: str,
    seed: int,
    layout: int,
    style: int,
    task: str,
    out_path: str,
    no_project_head_bbox: bool,
) -> None:
    """Export ground-truth objects for fusion calibration (offline Robocasa load)."""
    dest = export_robocasa_gt_scene(
        robot=robot,
        seed=seed,
        layout=layout,
        style=style,
        task=task,
        out_path=out_path,
        project_head_bbox=not no_project_head_bbox,
    )
    click.echo(f"Wrote GT scene ({dest.stat().st_size} bytes) -> {dest}")


if __name__ == "__main__":
    main()
