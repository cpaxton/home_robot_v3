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

"""``emet robots`` — list robot backends, MJCF/camera wiring, and preview helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Canonical registry keys (skip aliases like rb_y1, franka, xlerobot_dual).
CANONICAL_ROBOT_KEYS = (
    "stretch",
    "mobile_aloha",
    "galaxea_r1",
    "rby1",
    "innate_mars",
    "xlerobot",
    "franka_fr3",
    "sourccey",
    "yor",
)


def _resolve_robot_key(name: str) -> str:
    from emet.robots import ROBOT_REGISTRY

    key = name.lower().replace("-", "_")
    if key not in ROBOT_REGISTRY:
        raise click.ClickException(f"Unknown robot {name!r}. Run: emet robots list")
    return key


def _robot_row(robot_key: str) -> dict:
    import mujoco

    from emet.robots import get_robot_spec
    from emet.simulation.stereo_camera_utils import stereo_right_camera_name_from_spec
    from emet.utils.assets import get_robot_mjcf_path

    try:
        spec = get_robot_spec(robot_key)
    except NotImplementedError:
        return {"key": robot_key, "status": "stub backend"}
    if spec is None:
        return {"key": robot_key, "status": "missing backend"}

    mjcf_path = get_robot_mjcf_path(robot_key) or (Path(spec.mjcf_path) if spec.mjcf_path else None)
    mjcf_ok = mjcf_path is not None and Path(mjcf_path).is_file()
    cameras_in_mjcf: list[str] = []
    if mjcf_ok:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        for i in range(int(model.ncam)):
            cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cname:
                cameras_in_mjcf.append(cname)

    spec_cams = list(spec.camera_names)
    stereo_right = stereo_right_camera_name_from_spec(spec_cams)
    spawn_json = Path(mjcf_path).parent / "molmospaces_spawn.json" if mjcf_ok else None

    return {
        "key": robot_key,
        "name": spec.name,
        "dof": spec.dof,
        "mjcf": str(mjcf_path) if mjcf_ok else None,
        "cameras_spec": spec_cams,
        "cameras_mjcf": cameras_in_mjcf,
        "stereo_right": stereo_right,
        "planar_base": spec.planar_base_joint_names,
        "spawn_metadata": str(spawn_json) if spawn_json and spawn_json.is_file() else None,
    }


@click.group("robots", short_help="Robot registry, cameras, and MJCF diagnostics", invoke_without_command=True)
@click.pass_context
def robots_cmd(ctx: click.Context) -> None:
    """Inspect emet robot backends (stretch, innate_mars, xlerobot, franka_fr3, …).

    Examples:
      emet robots list
      emet robots info xlerobot
      emet robots preview-cameras xlerobot --source local
      emet robots preview-cameras xlerobot --source zmq
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(robots_list, as_json=False)


@robots_cmd.command("list", short_help="List robot backends with MJCF and camera summary")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")
def robots_list(as_json: bool) -> None:
    """Print canonical robot keys, MJCF availability, and camera names from RobotSpec."""
    rows = [_robot_row(k) for k in CANONICAL_ROBOT_KEYS]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return

    click.echo(f"{'Robot':<14} {'DOF':>4}  {'MJCF':<5}  Cameras (spec)")
    click.echo("-" * 72)
    for row in rows:
        if row.get("status") in ("missing backend", "stub backend"):
            click.echo(f"{row['key']:<14} {'—':>4}  {'no':<5}  ({row.get('status')})")
            continue
        mjcf_ok = "yes" if row.get("mjcf") else "no"
        cams = ", ".join(row.get("cameras_spec") or [])
        stereo = row.get("stereo_right")
        if stereo:
            cams += f"  [stereo→{stereo}]"
        click.echo(f"{row['key']:<14} {row['dof']:>4}  {mjcf_ok:<5}  {cams}")


@robots_cmd.command("info", short_help="Detailed RobotSpec + MJCF camera wiring for one robot")
@click.argument("robot")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")
def robots_info(robot: str, as_json: bool) -> None:
    """Show MJCF path, spec cameras, MJCF camera ids, stereo pair, and spawn metadata."""
    key = _resolve_robot_key(robot)
    row = _robot_row(key)
    if as_json:
        click.echo(json.dumps(row, indent=2))
        return

    click.echo(f"Robot: {row.get('name', key)} ({key})")
    click.echo(f"DOF: {row.get('dof', '—')}")
    click.echo(f"MJCF: {row.get('mjcf') or 'missing'}")
    click.echo(f"Cameras (spec): {', '.join(row.get('cameras_spec') or []) or '—'}")
    click.echo(f"Cameras (MJCF): {', '.join(row.get('cameras_mjcf') or []) or '—'}")
    if row.get("stereo_right"):
        click.echo(f"Stereo right: {row['stereo_right']}")
    if row.get("planar_base"):
        click.echo(f"Planar base joints: {', '.join(row['planar_base'])}")
    if row.get("spawn_metadata"):
        click.echo(f"MolmoSpaces spawn metadata: {row['spawn_metadata']}")
    else:
        click.echo("MolmoSpaces spawn metadata: (none — run emet molmospaces write-spawn-metadata after merge)")

    missing = set(row.get("cameras_spec") or []) - set(row.get("cameras_mjcf") or [])
    if missing:
        click.secho(f"Warning: spec cameras missing from MJCF: {', '.join(sorted(missing))}", fg="yellow")


@robots_cmd.command(
    "preview-cameras",
    short_help="Montage robot cameras (local MJCF or ZMQ)",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.argument("robot")
@click.pass_context
def robots_preview_cameras(ctx: click.Context, robot: str) -> None:
    """Shortcut for ``emet preview-cameras --robot ROBOT`` (forwards remaining flags)."""
    import subprocess

    from emet.utils.pythonpath import _venv_python, sanitize_emet_subprocess_env

    key = _resolve_robot_key(robot)
    venv_py = _venv_python()
    cmd = [str(venv_py or sys.executable), "-m", "emet.app.preview_robot_cameras", "--robot", key, *list(ctx.args)]
    # Same child PYTHONPATH rewrite as emet serve / emet run (see pythonpath.py).
    sys.exit(subprocess.call(cmd, env=sanitize_emet_subprocess_env()))
