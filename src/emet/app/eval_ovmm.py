# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""First-class ``emet ovmm`` CLI for find/full/sweep paper OVMM paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

from emet.eval.memory_backends import OVMM_MEMORY_BACKENDS
from emet.eval.ovmm_batch import BACKENDS, MANIP_MODES, OvmmBatchOptions, run_ovmm_batch
from emet.eval.ovmm_sweep import (
    DEFAULT_PRESET,
    aggregate_ovmm_rates,
    load_ovmm_sweep_preset,
    ovmm_sweep_status,
    prepare_multi_env_sweep,
    write_rerun_episodes_yaml,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_out_dir(preset: str = DEFAULT_PRESET) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = str(preset).replace("-", "_").split("/")[-1].removesuffix(".yaml")
    return Path.home() / "runs" / "emet" / f"ovmm_{name}" / stamp


def _parse_backends(backend: tuple[str, ...] | None) -> list[str] | None:
    if not backend:
        return None
    return list(backend)


def _batch_options_from_click(
    *,
    episodes: str,
    backends: tuple[str, ...] | None,
    tier: tuple[str, ...] | None,
    episode_id: tuple[str, ...] | None,
    merge_xy_m: float | None,
    staleness_horizon: int | None,
    compare_to_gt: bool,
    cpu_only: bool,
    sensor_perception: bool,
    graph_query: bool,
    not_rotate: bool,
    no_perfect_depth: bool,
    port_offset: int,
    port_stride: int,
    benchmark: str,
    output_dir: Path | None,
    dry_run: bool,
    explore_steps: int | None = None,
    no_scene_cache: bool = False,
    manip_mode: str | None = None,
    oneshot_localize: bool = False,
    agentic_max_rounds: int | None = None,
    agentic_max_nav_steps: int | None = None,
    full: bool = False,
) -> OvmmBatchOptions:
    return OvmmBatchOptions(
        episodes=episodes,
        backends=_parse_backends(backends),
        tiers=list(tier) if tier else None,
        episode_ids=list(episode_id) if episode_id else None,
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
        compare_to_gt=compare_to_gt,
        cpu_only=cpu_only,
        sensor_perception=sensor_perception,
        graph_query=graph_query,
        not_rotate=not_rotate,
        no_perfect_depth=no_perfect_depth,
        port_offset=port_offset,
        port_stride=port_stride,
        benchmark=benchmark,
        output_dir=output_dir,
        dry_run=dry_run,
        explore_steps=explore_steps,
        no_scene_cache=no_scene_cache,
        manip_mode=manip_mode,
        oneshot_localize=oneshot_localize,
        agentic_max_rounds=agentic_max_rounds,
        agentic_max_nav_steps=agentic_max_nav_steps,
        full=full,
    )


def _common_batch_options(f):
    """Shared Click options for find/full."""
    opts = [
        click.option(
            "--episodes",
            type=click.Path(exists=False, dir_okay=False, path_type=str),
            required=True,
            help="Episode YAML (find_episodes.yaml / full_episodes.yaml)",
        ),
        click.option(
            "--backend",
            "backends",
            type=click.Choice(list(BACKENDS)),
            multiple=True,
            help="Memory backend (repeat; default: dynagraph)",
        ),
        click.option("--tier", multiple=True, help="Filter by tier (S0, S1, S2)"),
        click.option("--episode-id", multiple=True, help="Filter by episode id"),
        click.option("--merge-xy-m", type=float, default=None),
        click.option("--staleness-horizon", type=int, default=None),
        click.option("--compare-to-gt", is_flag=True, default=False),
        click.option("--cpu-only", is_flag=True, default=False),
        click.option("--sensor-perception", is_flag=True, default=False),
        click.option("--graph-query", is_flag=True, default=False),
        click.option("--not-rotate", is_flag=True, default=False),
        click.option("--no-perfect-depth", is_flag=True, default=False),
        click.option(
            "--port-offset",
            type=int,
            default=None,
            help="Base ZMQ port offset (default: pid-based)",
        ),
        click.option("--port-stride", type=int, default=2, show_default=True),
        click.option("--benchmark", default="configs/ovmm/benchmark.yaml", show_default=True),
        click.option(
            "--output-dir",
            type=click.Path(file_okay=False, path_type=Path),
            default=None,
            help="Per-run JSON + aggregate CSV directory",
        ),
        click.option("--dry-run", is_flag=True, default=False),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


def _launch_via_jobs(
    *,
    argv_tail: list[str],
    job_name: str,
    need_mib: int,
    out_dir: Path | None,
    foreground: bool = False,
) -> None:
    """Re-enter ``emet jobs run`` with the same ovmm subcommand (sans --via-jobs)."""
    root = _project_root()
    cmd = [
        sys.executable,
        "-m",
        "emet.cli",
        "jobs",
        "run",
        "--name",
        job_name,
        "--need-mib",
        str(int(need_mib)),
    ]
    if out_dir is not None:
        cmd.extend(["--out-dir", str(out_dir)])
    if foreground:
        cmd.append("--foreground")
    cmd.append("--")
    # Prefer SDPA for in-process VL (FA2 has hung on MuJoCo+Qwen co-resident runs).
    cmd.extend(
        [
            "env",
            "EMET_ALLOW_SDPA_ATTN=1",
            sys.executable,
            "-m",
            "emet.cli",
            "ovmm",
            *argv_tail,
        ]
    )
    click.echo(f"launching via emet jobs: name={job_name} need_mib={need_mib}", err=True)
    rc = subprocess.call(cmd, cwd=str(root))
    raise SystemExit(rc)


@click.group("ovmm", invoke_without_command=True)
@click.pass_context
def ovmm_group(ctx: click.Context) -> None:
    """OVMM find/full paper benchmarks and multi-env sweeps (Molmo + Robocasa)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@ovmm_group.command("find", short_help="Batch OVMM find-phase (FindObj / FindRec)")
@_common_batch_options
@click.option("--explore-steps", type=int, default=None, help="Override episode explore_steps")
@click.option("--no-scene-cache", is_flag=True, default=False)
@click.option(
    "--oneshot-localize",
    is_flag=True,
    default=False,
    help="Ablation: skip AgenticEQA find; one-shot voxel/graph localize only",
)
@click.option(
    "--agentic-max-rounds",
    type=int,
    default=None,
    help="Agentic find: cap VLM router rounds per question (default: eqa.agentic_max_rounds)",
)
@click.option(
    "--agentic-max-nav-steps",
    type=int,
    default=None,
    help="Agentic find: cap nav steps per question (default: eqa.agentic_max_nav_steps)",
)
def ovmm_find(
    episodes: str,
    backends: tuple[str, ...],
    tier: tuple[str, ...],
    episode_id: tuple[str, ...],
    merge_xy_m: float | None,
    staleness_horizon: int | None,
    compare_to_gt: bool,
    cpu_only: bool,
    sensor_perception: bool,
    graph_query: bool,
    not_rotate: bool,
    no_perfect_depth: bool,
    port_offset: int | None,
    port_stride: int,
    benchmark: str,
    output_dir: Path | None,
    dry_run: bool,
    explore_steps: int | None,
    no_scene_cache: bool,
    oneshot_localize: bool,
    agentic_max_rounds: int | None,
    agentic_max_nav_steps: int | None,
) -> None:
    """Run find-phase episodes (same path as scripts/eval_ovmm_find_phases.py).

    Dynagraph default: shared AgenticEQA loop (OVMM phrased as questions).
    """
    po = int(port_offset) if port_offset is not None else int(os.getpid() % 400 + 140)
    opts = _batch_options_from_click(
        episodes=episodes,
        backends=backends or None,
        tier=tier or None,
        episode_id=episode_id or None,
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
        compare_to_gt=compare_to_gt,
        cpu_only=cpu_only,
        sensor_perception=sensor_perception,
        graph_query=graph_query,
        not_rotate=not_rotate,
        no_perfect_depth=no_perfect_depth,
        port_offset=po,
        port_stride=port_stride,
        benchmark=benchmark,
        output_dir=output_dir,
        dry_run=dry_run,
        explore_steps=explore_steps,
        no_scene_cache=no_scene_cache,
        oneshot_localize=oneshot_localize,
        agentic_max_rounds=agentic_max_rounds,
        agentic_max_nav_steps=agentic_max_nav_steps,
        full=False,
    )
    raise SystemExit(run_ovmm_batch(opts, repo_root=_project_root()))


@ovmm_group.command("full", short_help="Batch full OVMM (find + pick/place)")
@_common_batch_options
@click.option(
    "--manip-mode",
    type=click.Choice(list(MANIP_MODES)),
    default=None,
    help="Override full-OVMM episode modes (default: use episode YAML, otherwise oracle)",
)
@click.option("--explore-steps", type=int, default=None, help="Override episode explore_steps")
@click.option("--no-scene-cache", is_flag=True, default=False)
@click.option(
    "--oneshot-localize",
    is_flag=True,
    default=False,
    help="Ablation: skip AgenticEQA find; one-shot voxel/graph localize only",
)
@click.option(
    "--agentic-max-rounds",
    type=int,
    default=None,
    help="Agentic find: cap VLM router rounds per question (default: eqa.agentic_max_rounds)",
)
@click.option(
    "--agentic-max-nav-steps",
    type=int,
    default=None,
    help="Agentic find: cap nav steps per question (default: eqa.agentic_max_nav_steps)",
)
def ovmm_full(
    episodes: str,
    backends: tuple[str, ...],
    tier: tuple[str, ...],
    episode_id: tuple[str, ...],
    merge_xy_m: float | None,
    staleness_horizon: int | None,
    compare_to_gt: bool,
    cpu_only: bool,
    sensor_perception: bool,
    graph_query: bool,
    not_rotate: bool,
    no_perfect_depth: bool,
    port_offset: int | None,
    port_stride: int,
    benchmark: str,
    output_dir: Path | None,
    dry_run: bool,
    manip_mode: str | None,
    explore_steps: int | None,
    no_scene_cache: bool,
    oneshot_localize: bool,
    agentic_max_rounds: int | None,
    agentic_max_nav_steps: int | None,
) -> None:
    """Run full OVMM episodes (same path as scripts/eval_ovmm_full.py)."""
    po = int(port_offset) if port_offset is not None else int(os.getpid() % 400 + 140)
    opts = _batch_options_from_click(
        episodes=episodes,
        backends=backends or None,
        tier=tier or None,
        episode_id=episode_id or None,
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
        compare_to_gt=compare_to_gt,
        cpu_only=cpu_only,
        sensor_perception=sensor_perception,
        graph_query=graph_query,
        not_rotate=not_rotate,
        no_perfect_depth=no_perfect_depth,
        port_offset=po,
        port_stride=port_stride,
        benchmark=benchmark,
        output_dir=output_dir,
        dry_run=dry_run,
        manip_mode=manip_mode,
        explore_steps=explore_steps,
        no_scene_cache=no_scene_cache,
        oneshot_localize=oneshot_localize,
        agentic_max_rounds=agentic_max_rounds,
        agentic_max_nav_steps=agentic_max_nav_steps,
        full=True,
    )
    raise SystemExit(run_ovmm_batch(opts, repo_root=_project_root()))


@ovmm_group.command("prepare", short_help="Write sim/ + find/full episode YAMLs from a preset")
@click.option("--preset", default=DEFAULT_PRESET, show_default=True, help="Preset name or YAML path")
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (default: ~/runs/emet/ovmm_<preset>/TIMESTAMP)",
)
@click.option(
    "--sync-registry/--no-sync-registry",
    default=None,
    help="Sync Robocasa LightWheel fixture registry (default: from preset)",
)
def ovmm_prepare(preset: str, out_dir: Path | None, sync_registry: bool | None) -> None:
    """Generate multi-env sweep tree (no default_table)."""
    out = out_dir or _default_out_dir(preset)
    prepared = prepare_multi_env_sweep(out, preset, sync_robocasa_registry=sync_registry)
    click.echo(f"preset={prepared.preset_name}")
    click.echo(f"out={prepared.out_dir}")
    click.echo(f"find_episodes={prepared.find_episodes}")
    click.echo(f"full_episodes={prepared.full_episodes}")
    click.echo(f"sim_dir={prepared.sim_dir}")


@ovmm_group.command("rates", short_help="Aggregate find/full JSON → rates.json")
@click.option(
    "--out",
    "out_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Sweep OUT directory containing find/ and/or full/",
)
@click.option(
    "--backend",
    default="dynagraph",
    show_default=True,
    type=click.Choice(list(OVMM_MEMORY_BACKENDS)),
)
@click.option(
    "--include-bind-fails",
    is_flag=True,
    default=False,
    help="Include bind/task-init failures in rates (default: exclude)",
)
def ovmm_rates(out_dir: Path, backend: str, include_bind_fails: bool) -> None:
    aggregate_ovmm_rates(out_dir, backend=backend, exclude_bind_fails=not include_bind_fails)


@ovmm_group.command("status", short_help="Print per-episode outcomes under OUT")
@click.option(
    "--out",
    "out_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--backend",
    default="dynagraph",
    show_default=True,
    type=click.Choice(list(OVMM_MEMORY_BACKENDS)),
)
@click.option("--json", "as_json", is_flag=True, default=False)
def ovmm_status(out_dir: Path, backend: str, as_json: bool) -> None:
    st = ovmm_sweep_status(out_dir, backend=backend)
    if as_json:
        click.echo(json.dumps(st, indent=2))
        return
    click.echo(f"out={st['out_dir']}")
    click.echo(
        f"find_episodes={st['find_episodes_exists']} full_episodes={st['full_episodes_exists']} "
        f"rates={st['rates_exists']}"
    )
    click.echo(
        f"find: n={st['n_find']} bind_or_init_fail={st['n_find_bind_fail']} | "
        f"full: n={st['n_full']} bind_or_init_fail={st['n_full_bind_fail']}"
    )
    for phase in ("find", "full"):
        rows = st[phase]
        if not rows:
            continue
        click.echo(f"\n[{phase}]")
        for r in rows:
            click.echo(
                f"  {r['episode_id']}\t{r['kind']}\t"
                f"partial={r.get('find_partial') if phase == 'find' else r.get('ovmm_full_partial')}\t"
                f"{r.get('error') or ''}"
            )


@ovmm_group.command("sweep", short_help="prepare → find → full → rates (paper multi-env path)")
@click.option("--preset", default=DEFAULT_PRESET, show_default=True)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Sweep OUT (default: ~/runs/emet/ovmm_<preset>/TIMESTAMP)",
)
@click.option(
    "--episodes",
    "episodes_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Skip prepare; use existing OUT with find_episodes.yaml / full_episodes.yaml",
)
@click.option("--backend", default=None, help="Override preset backend (default: dynagraph)")
@click.option(
    "--manip-mode",
    type=click.Choice(list(MANIP_MODES)),
    default=None,
    help="Override preset manip_mode for full (default: sim)",
)
@click.option("--port-offset", type=int, default=None, help="Override find port offset")
@click.option("--port-stride", type=int, default=None)
@click.option("--find-only", is_flag=True, default=False)
@click.option("--full-only", is_flag=True, default=False)
@click.option(
    "--rerun-failed",
    is_flag=True,
    default=False,
    help="Rebuild episode lists from bind/task-init failures in OUT",
)
@click.option("--via-jobs", is_flag=True, default=False, help="Wrap in emet jobs run")
@click.option("--need-mib", type=int, default=None, help="VRAM for --via-jobs (default from preset)")
@click.option("--job-name", default=None, help="Job name when using --via-jobs")
@click.option("--foreground", is_flag=True, default=False, help="Foreground jobs run")
@click.option("--dry-run", is_flag=True, default=False)
@click.option(
    "--no-scene-cache",
    is_flag=True,
    default=False,
    help="Force live Dynagraph rotate/explore (skip prebuilt scene map cache)",
)
@click.option(
    "--sync-registry/--no-sync-registry",
    default=None,
    help="Sync Robocasa registry during prepare (default: from preset)",
)
def ovmm_sweep(
    preset: str,
    out_dir: Path | None,
    episodes_dir: Path | None,
    backend: str | None,
    manip_mode: str | None,
    port_offset: int | None,
    port_stride: int | None,
    find_only: bool,
    full_only: bool,
    rerun_failed: bool,
    via_jobs: bool,
    need_mib: int | None,
    job_name: str | None,
    foreground: bool,
    dry_run: bool,
    no_scene_cache: bool,
    sync_registry: bool | None,
) -> None:
    """Run the multi-env OVMM paper sweep (Robocasa + MolmoSpaces)."""
    if find_only and full_only:
        raise click.UsageError("Use only one of --find-only / --full-only")

    data = load_ovmm_sweep_preset(preset)
    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    be = backend or str(defaults.get("backend", "dynagraph"))
    mm = manip_mode or str(defaults.get("manip_mode", "sim"))
    stride = int(port_stride if port_stride is not None else defaults.get("port_stride", 4))
    po_find = int(port_offset if port_offset is not None else defaults.get("port_offset_find", 200))
    po_full = int(defaults.get("port_offset_full", 260))
    if port_offset is not None:
        po_full = int(port_offset)
    mib = int(need_mib if need_mib is not None else defaults.get("need_mib", 8000))
    # Prefer live agent mapping for paper agent-path sweeps unless caller keeps cache.
    force_live = bool(no_scene_cache or defaults.get("no_scene_cache", False))
    agentic_find = defaults.get("agentic_find")
    if agentic_find is not None:
        agentic_find = bool(agentic_find)
    agentic_max_rounds = defaults.get("agentic_max_rounds")
    agentic_max_nav_steps = defaults.get("agentic_max_nav_steps")

    out = Path(episodes_dir or out_dir or _default_out_dir(preset)).expanduser().resolve()

    if via_jobs:
        # Rebuild argv without --via-jobs for the inner process.
        argv_tail: list[str] = [
            "sweep",
            "--preset",
            preset,
            "--out",
            str(out),
            "--backend",
            be,
            "--manip-mode",
            mm,
            "--port-offset",
            str(po_find),
            "--port-stride",
            str(stride),
        ]
        if episodes_dir is not None:
            argv_tail.extend(["--episodes", str(Path(episodes_dir).expanduser().resolve())])
        if find_only:
            argv_tail.append("--find-only")
        if full_only:
            argv_tail.append("--full-only")
        if rerun_failed:
            argv_tail.append("--rerun-failed")
        if dry_run:
            argv_tail.append("--dry-run")
        if force_live:
            argv_tail.append("--no-scene-cache")
        if sync_registry is True:
            argv_tail.append("--sync-registry")
        elif sync_registry is False:
            argv_tail.append("--no-sync-registry")
        name = job_name or f"ovmm-{str(data.get('name') or 'sweep')}"
        _launch_via_jobs(argv_tail=argv_tail, job_name=name, need_mib=mib, out_dir=out, foreground=foreground)
        return

    if episodes_dir is None and not (out / "find_episodes.yaml").is_file():
        prepared = prepare_multi_env_sweep(out, preset, sync_robocasa_registry=sync_registry)
        click.echo(f"prepared {prepared.preset_name} → {prepared.out_dir}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        click.echo(f"using existing OUT={out}")

    find_eps = out / "find_episodes.yaml"
    full_eps = out / "full_episodes.yaml"
    if rerun_failed:
        rerun_find = write_rerun_episodes_yaml(out, phase="find", backend=be)
        rerun_full = write_rerun_episodes_yaml(out, phase="full", backend=be)
        if rerun_find is not None:
            find_eps = rerun_find
            click.echo(f"rerun find episodes → {find_eps}")
        if rerun_full is not None:
            full_eps = rerun_full
            click.echo(f"rerun full episodes → {full_eps}")

    run_find = not full_only
    run_full = not find_only

    if run_find and find_eps.is_file():
        opts = OvmmBatchOptions(
            episodes=str(find_eps),
            backends=[be],
            port_offset=po_find,
            port_stride=stride,
            output_dir=out / "find",
            dry_run=dry_run,
            no_scene_cache=force_live,
            agentic_find=agentic_find,
            agentic_max_rounds=agentic_max_rounds,
            agentic_max_nav_steps=agentic_max_nav_steps,
            full=False,
        )
        rc = run_ovmm_batch(opts, repo_root=_project_root())
        if rc != 0:
            raise SystemExit(rc)
    elif run_find:
        click.echo(f"skip find: missing {find_eps}", err=True)

    if run_full and full_eps.is_file():
        opts = OvmmBatchOptions(
            episodes=str(full_eps),
            backends=[be],
            port_offset=po_full,
            port_stride=stride,
            output_dir=out / "full",
            dry_run=dry_run,
            manip_mode=mm,
            no_scene_cache=force_live,
            agentic_find=agentic_find,
            agentic_max_rounds=agentic_max_rounds,
            agentic_max_nav_steps=agentic_max_nav_steps,
            full=True,
        )
        rc = run_ovmm_batch(opts, repo_root=_project_root())
        if rc != 0:
            raise SystemExit(rc)
    elif run_full:
        click.echo(f"skip full: missing {full_eps}", err=True)

    if not dry_run:
        aggregate_ovmm_rates(out, backend=be)
