# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

from emet.cli_cmds.bootstrap import (
    _project_root,
)


def _run_molmospaces_wrapper(args: list[str]) -> int:
    """Run the emet-molmospaces wrapper (list-scenes, install-scene, merge-scene, serve). Returns exit code."""
    from emet.simulation.molmospaces_config import (
        build_molmospaces_wrapper_command,
        ensure_molmospaces_assets_dir_env,
    )

    cmd = build_molmospaces_wrapper_command(args)
    if cmd is None:
        click.echo(
            "MolmoSpaces wrapper not found. The package `emet-molmospaces` is part of this repo and is not "
            "published on PyPI, so `pip install emet-molmospaces` will not work.\n\n"
            "From the project root, run (sim install also creates `.venv-molmospaces` by default):\n"
            "  ./install.sh -y\n"
            "Or wrapper only:\n"
            "  ./install.sh --molmospaces -y\n\n"
            "Or create the venv and install the local packages:\n"
            "  uv venv .venv-molmospaces\n"
            "  uv pip install --python .venv-molmospaces/bin/python --no-deps -e .\n"
            "  uv pip install --python .venv-molmospaces/bin/python -e packages/emet_molmospaces\n\n"
            "Optional: set MLSPACES_ASSETS_DIR (defaults to ~/.cache/molmospaces/assets) and "
            "MLSPACES_CACHE_DIR (defaults to ~/.cache/molmospaces/resource_cache; must differ from assets; "
            "see docs/molmospaces.md).",
            err=True,
        )
        return 1
    env = os.environ.copy()
    ensure_molmospaces_assets_dir_env(env)
    return subprocess.call(cmd, cwd=_project_root(), env=env)


@click.group("molmospaces", short_help="MolmoSpaces scenes and robots (requires emet-molmospaces wrapper)")
def molmospaces_cmd() -> None:
    """Set up MolmoSpaces scenes, list robots (e.g. rby1 / Galaxea R1), and run simulation.

    list-robots works without the wrapper. list-scenes, install-scene, merge-scene, and serve
    require the local emet-molmospaces package (see docs/molmospaces.md). ``write-spawn-metadata``
    and ``build-occ-map`` are core-only offline tools (see docs/molmospaces_spawn_metadata.md).
    ``export-nerfstudio`` is core-only (reads an explore episode directory). Install wrapper with:
      ./install.sh -y   (default sim path)   or   ./install.sh --molmospaces -y   or   editable install of packages/emet_molmospaces
    """


@molmospaces_cmd.command("list-robots", short_help="List supported robot IDs")
def molmospaces_list_robots() -> None:
    """Print MolmoSpaces robot IDs and emet vendored robots with MJCF (xlerobot, franka_fr3, …)."""
    from emet.app.robots_cli import CANONICAL_ROBOT_KEYS
    from emet.simulation.molmospaces_config import DEFAULT_MOLMOSPACES_ROBOT, MOLMOSPACES_ROBOT_IDS
    from emet.utils.assets import get_robot_mjcf_path

    click.echo("MolmoSpaces wrapper IDs: " + ", ".join(MOLMOSPACES_ROBOT_IDS))
    click.echo(f"MolmoSpaces default: {DEFAULT_MOLMOSPACES_ROBOT}")
    click.echo("")
    click.echo("Emet registry robots (use with emet serve mujoco --robot <key>):")
    for key in CANONICAL_ROBOT_KEYS:
        mjcf = get_robot_mjcf_path(key)
        tag = "mjcf" if mjcf and mjcf.is_file() else "no mjcf"
        click.echo(f"  {key} ({tag})")


@molmospaces_cmd.command("list-scenes", short_help="List scene names and split sizes")
def molmospaces_list_scenes() -> None:
    """Print available MolmoSpaces scene names and split counts. Requires emet-molmospaces wrapper."""
    sys.exit(_run_molmospaces_wrapper(["list-scenes"]))


@molmospaces_cmd.command("install-scene", short_help="Install a scene and optionally write XML path")
@click.option("--scene", default="ithor", help="Scene name (e.g. ithor, procthor-10k)")
@click.option("--split", default="train", type=click.Choice(["train", "val", "test"]))
@click.option("--index", default=0, type=int, help="Scene index within split")
@click.option(
    "--scene-path", type=click.Path(path_type=Path), default=None, help="Write installed scene XML to this path"
)
@click.option(
    "--install-if-missing",
    is_flag=True,
    help="Download/link the scene archive without prompting if it is not on disk yet.",
)
def molmospaces_install_scene(
    scene: str, split: str, index: int, scene_path: Path | None, install_if_missing: bool
) -> None:
    """Download and install a MolmoSpaces scene; optionally copy the scene XML to a path."""
    args = ["install-scene", "--scene", scene, "--split", split, "--index", str(index)]
    if scene_path is not None:
        args.extend(["--scene-path", str(scene_path)])
    if install_if_missing:
        args.append("--install-if-missing")
    sys.exit(_run_molmospaces_wrapper(args))


@molmospaces_cmd.command(
    "write-spawn-metadata",
    short_help="Measure spawn hints from a merged MJCF and update molmospaces_spawn.json",
)
@click.option(
    "--robot",
    required=True,
    help="Robot id merged into the MJCF (e.g. stretch, rby1, galaxea_r1, innate_mars).",
)
@click.option(
    "--mjcf",
    "mjcf_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Merged scene+robot MJCF (from merge-scene or a temp path from emet serve molmospaces).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output JSON (default: <robot_mjcf_dir>/molmospaces_spawn.json).",
)
@click.option(
    "--no-merge",
    is_flag=True,
    help="Replace the file instead of merging measured keys into existing JSON.",
)
@click.option("--base-body", default="base_link", show_default=True, help="Free-joint base body name.")
@click.option("--seed-x", default=0.0, show_default=True, type=float, help="World X (m) for measurement pose.")
@click.option("--seed-y", default=0.0, show_default=True, type=float, help="World Y (m) for measurement pose.")
def molmospaces_write_spawn_metadata(
    robot: str,
    mjcf_path: Path,
    output_path: Path | None,
    no_merge: bool,
    base_body: str,
    seed_x: float,
    seed_y: float,
) -> None:
    """Offline spawn tuning: kinematic floor settle → foot clearance / base height in JSON.

    Full workflow and JSON field reference: docs/molmospaces_spawn_metadata.md
    """
    from emet.app.write_molmospaces_spawn_metadata import run_write

    out = run_write(
        robot,
        mjcf_path,
        output_path=output_path,
        merge_existing=not no_merge,
        base_body_name=base_body,
        seed_x=seed_x,
        seed_y=seed_y,
    )
    click.echo(f"Wrote {out}")


@molmospaces_cmd.command("build-occ-map", short_help="Build vendored iTHOR orthographic occupancy map from merged MJCF")
@click.argument("mjcf", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: same directory as MJCF).",
)
@click.option("--agent-radius", default=0.32, type=float, show_default=True)
@click.option("--px-per-m", default=120, type=int, show_default=True)
def molmospaces_build_occ_map(mjcf: Path, output_dir: Path | None, agent_radius: float, px_per_m: int) -> None:
    """Orthographic segmentation occupancy (Molmo-style) for spawn QA; writes occupancy.png + occupancy_meta.json."""
    from emet.app.build_molmo_occupancy_map import run_build

    png, meta = run_build(mjcf, output_dir, agent_radius=agent_radius, px_per_m=px_per_m)
    click.echo(f"Wrote {png}")
    click.echo(f"Wrote {meta}")


@molmospaces_cmd.command("merge-scene", short_help="Write merged scene+robot MJCF for emet serve mujoco")
@click.option("--scene", default="ithor", help="Scene name")
@click.option("--split", default="train", type=click.Choice(["train", "val", "test"]))
@click.option("--index", default=0, type=int, help="Scene index within split")
@click.option("--robot", default="stretch", help="Robot ID (default: stretch)")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Where to write the merged MJCF (for emet serve mujoco --scene_path)",
)
@click.option(
    "--install-if-missing",
    is_flag=True,
    help="Download/link the scene archive without prompting if it is not on disk yet.",
)
def molmospaces_merge_scene(
    scene: str, split: str, index: int, robot: str, output: Path, install_if_missing: bool
) -> None:
    """Install scene if needed, merge rby1 MJCF, write XML for ZMQ server + emet run agent."""
    args = [
        "merge-scene",
        "--scene",
        scene,
        "--split",
        split,
        "--index",
        str(index),
        "--robot",
        robot,
        "--output",
        str(output),
    ]
    if install_if_missing:
        args.append("--install-if-missing")
    sys.exit(_run_molmospaces_wrapper(args))


@molmospaces_cmd.command("serve", short_help="Run MolmoSpaces simulation (scene + robot)")
@click.option("--scene", default="ithor", help="Scene name")
@click.option("--split", default="train", type=click.Choice(["train", "val", "test"]))
@click.option("--index", default=0, type=int, help="Scene index")
@click.option("--robot", default="stretch", help="Robot ID (default: stretch)")
@click.option("--headless", is_flag=True, help="Run without viewer")
@click.option("--viewer", is_flag=True, help="Open MuJoCo viewer")
@click.option("--rerun", type=str, default="", metavar="PORT_OR_PATH", help="Log to rerun (port or RRD path)")
@click.option(
    "--scene-path", type=click.Path(path_type=Path), default=None, help="Write installed scene XML path to this file"
)
@click.option(
    "--install-if-missing",
    is_flag=True,
    help="Download/link the scene archive without prompting if it is not on disk yet.",
)
def molmospaces_serve(
    scene: str,
    split: str,
    index: int,
    robot: str,
    headless: bool,
    viewer: bool,
    rerun: str,
    scene_path: Path | None,
    install_if_missing: bool,
) -> None:
    """Run MuJoCo simulation with a MolmoSpaces scene and robot. Use --viewer to see the sim."""
    args = ["serve", "--scene", scene, "--split", split, "--index", str(index), "--robot", robot]
    if headless:
        args.append("--headless")
    if viewer:
        args.append("--viewer")
    if rerun:
        args.extend(["--rerun", rerun])
    if scene_path is not None:
        args.extend(["--scene-path", str(scene_path)])
    if install_if_missing:
        args.append("--install-if-missing")
    sys.exit(_run_molmospaces_wrapper(args))


@molmospaces_cmd.command("export-nerfstudio", short_help="Build transforms.json from explore episode metadata")
@click.option(
    "--episode-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    required=True,
    help="Directory containing metadata.jsonl and images/ from molmospaces-explore.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="transforms.json",
    show_default=True,
    help="Output filename inside episode-dir (default transforms.json).",
)
def molmospaces_export_nerfstudio(episode_dir: str, output: str) -> None:
    """Convert ``metadata.jsonl`` + images into NERFStudio-style ``transforms.json``."""
    from pathlib import Path

    from emet.molmospaces.episode_writer import export_nerfstudio_transforms

    p = export_nerfstudio_transforms(Path(episode_dir), output_name=output)
    click.echo(str(p))


def register(main: click.Group) -> None:
    main.add_command(molmospaces_cmd)
