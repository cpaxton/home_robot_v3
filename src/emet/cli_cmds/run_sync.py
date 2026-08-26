# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
from click.core import ParameterSource

from emet.app.robots_cli import robots_cmd
from emet.cli_cmds.bootstrap import (
    _CONTEXT_SETTINGS,
    _EMET_RUN_APPS_WITH_ROBOT,
    _has_uv,
    _project_root,
    _project_venv_python,
    _require_repo_venv_when_in_repo,
    _run_module,
)


@click.group("dataset", short_help="Dataset inspect, export, and replay")
def dataset_cmd() -> None:
    """Learning datasets (MolmoBot-Data H5, etc.)."""


from emet.datasets.molmobot.cli import molmobot_dataset_group  # noqa: E402

dataset_cmd.add_command(molmobot_dataset_group)


@click.group("molmobot", short_help="MolmoBot policy bridge (optional venv)")
def molmobot_cmd() -> None:
    """Delegate to packages/emet_molmobot when installed."""


@molmobot_cmd.command("serve-policy")
@click.option("--hf-repo", required=True)
@click.option("--action-type", default="joint_pos", show_default=True)
@click.argument("extra_args", nargs=-1)
def molmobot_serve_policy(hf_repo: str, action_type: str, extra_args: tuple[str, ...]) -> None:
    """Run upstream MolmoBot serve_molmo.py (requires MOLMOBOT_ROOT)."""
    try:
        from emet_molmobot.runner import serve_policy
    except ImportError:
        click.echo(
            "Install packages/emet_molmobot editable or set PYTHONPATH. See docs/datasets/molmobot.md.",
            err=True,
        )
        raise SystemExit(1) from None
    argv = ["--hf-repo", hf_repo, "--action-type", action_type, *extra_args]
    raise SystemExit(serve_policy(argv))


@click.command("show-memory", short_help="Open a saved memory in Rerun")
@click.argument(
    "path",
    type=click.Path(path_type=Path),
    default="saved_memory",
    required=False,
)
@click.option("--open-browser", is_flag=True, help="Open browser to Rerun web viewer")
def show_memory(path: str, open_browser: bool) -> None:
    """Load a memory directory and display it in Rerun.

    PATH defaults to saved_memory. Must be a directory with manifest.json
    (common memory format from create-and-print-memory or backend save).

    Examples:
      emet show-memory
      emet show-memory ./my_memory --open-browser
    """
    args = [str(path)]
    if open_browser:
        args.append("--open-browser")
    sys.exit(_run_module("emet.app.show_memory", args))


@click.command("graph-memory-show", short_help="Print formatted scene graph from saved memory directory")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--max-nodes", type=int, default=None, help="Truncate long node lists")
def graph_memory_show(path: Path, max_nodes: int | None) -> None:
    """Load common memory format (graph_eqa) and print nodes/edges to stdout."""

    args = [str(path)]
    if max_nodes is not None:
        args.extend(["--max-nodes", str(max_nodes)])
    sys.exit(_run_module("emet.app.graph_memory_show", args))


@click.command(
    "reprocess-graph-eqa-cache",
    short_help="Rebuild GraphEQA graph from saved memory frames (offline)",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def reprocess_graph_eqa_cache(args: tuple[str, ...]) -> None:
    """Re-run instance→graph logic on ``frames/`` under a saved memory directory.

    Example: ``emet reprocess-graph-eqa-cache ./saved_memory -o ./saved_memory_v2``
    """
    if not args:
        raise click.UsageError("Missing INPUT_DIR. See emet reprocess-graph-eqa-cache --help.")
    sys.exit(_run_module("emet.app.reprocess_graph_eqa_cache", list(args)))


@click.command("print", short_help="Print summary of a saved memory directory")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
def print_memory(path: Path) -> None:
    """Load and print a summary of a saved memory directory.

    Use this with memory saved by emet run dynamem, emet run create-and-print-memory,
    or other runs that write the common memory format (manifest.json, point_cloud.npz, etc.).
    """
    from emet.memory.utils import print_memory_from_path

    print_memory_from_path(str(path))


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.argument(
    "app",
    type=click.Choice(
        [
            "dynamem",
            "scene-graph",
            "graph-eqa",
            "dynagraph",
            "lazy-graph",
            "mapping",
            "grasp",
            "chat",
            "agent",
            "web-chat",
            "ai_pickup",
            "timing",
            "discord",
            "create-and-print-memory",
            "molmospaces-explore",
            "debug-da3-depth",
            "debug-lingbot-depth",
            "graph-eqa-habitat",
        ]
    ),
)
@click.option("--robot-ip", "--robot_ip", default="127.0.0.1", help="Robot or simulator IP")
@click.option(
    "--robot",
    default="stretch",
    help=(
        "Robot backend (stretch, innate_mars, rby1, galaxea_r1). Must match emet serve mujoco --robot. "
        "If you omit this flag, ``emet run`` does not forward ``--robot`` to sub-apps—so e.g. "
        "``emet run agent --agent-config configs/agent_innate_mars.yaml`` uses top-level ``robot:`` from that YAML."
    ),
)
@click.option("--server-ip", "--server_ip", default="127.0.0.1", help="Server IP (e.g. for AnyGrasp)")
@click.option("-S", "--skip", "skip_confirmations", is_flag=True, help="Skip confirmations")
@click.option("--headless", is_flag=True, help="Run without display")
@click.option("--visual-servo", "-V", "--visual_servo", is_flag=True, help="Use visual servoing (dynamem)")
@click.option("--target-object", "--target_object", help="Target object to grasp (grasp) or pick (dynamem)")
@click.option("--target-receptacle", "--target_receptacle", help="Target receptacle to place on (dynamem)")
@click.option("--parameter-file", "--parameter_file", help="Planner config (e.g. sim_planner.yaml)")
@click.option(
    "--port-offset",
    default=0,
    type=int,
    help="Add to default ZMQ ports (e.g. 100 → 4501-4504). Must match the server.",
)
@click.pass_context
def run(
    ctx: click.Context,
    app: str,
    robot_ip: str,
    robot: str,
    server_ip: str,
    skip_confirmations: bool,
    headless: bool,
    visual_servo: bool,
    target_object: str | None,
    target_receptacle: str | None,
    parameter_file: str | None,
    port_offset: int,
) -> None:
    """Run a robot agent or app.

    Unknown options (e.g. --match-method, --rerun-debug) are passed through to the app.

    Examples:
      emet run dynamem --robot-ip 127.0.0.1 -S
      emet run dynamem -S --port-offset 100
      emet run dynamem -S --visual-servo --match-method class --target-object apple --target-receptacle plate
      emet run molmospaces-explore --output-dir ./ep0 --steps 40
      emet run molmospaces-explore --start-sim --robot xlerobot --scene ithor --headless \\
        --output-dir ./ep0 --steps 40
      # --robot optional: omit to read emet_robot_id from the running ZMQ server
      emet run mapping --robot-ip 127.0.0.1
      emet run grasp --target-object "red cylinder" --parameter-file sim_planner.yaml
      emet run discord --robot-ip 192.168.1.15 --task pickup   # requires DISCORD_TOKEN in env
      emet run debug-da3-depth --robot innate_mars   # DA3 depth + point cloud in Rerun (or: emet debug-da3-depth)
    """
    _require_repo_venv_when_in_repo()
    args = list(ctx.args)
    # Do not inject wrapper defaults for ``--robot_ip`` / ``--robot``: they would override
    # ``--connection`` host resolution and YAML ``robot:`` (same pattern for both).
    if app != "graph-eqa-habitat" and ctx.get_parameter_source("robot_ip") != ParameterSource.DEFAULT:
        args.extend(["--robot_ip", robot_ip])
    if app in _EMET_RUN_APPS_WITH_ROBOT:
        if ctx.get_parameter_source("robot") != ParameterSource.DEFAULT:
            args.extend(["--robot", robot])
    if port_offset:
        args.extend(["--port-offset", str(port_offset)])
    if app == "dynamem":
        args.extend(["--server_ip", server_ip])
        if skip_confirmations:
            args.append("-S")
        if headless:
            args.append("--headless")
        if visual_servo:
            args.append("--visual-servo")
        if target_object:
            args.extend(["--target_object", target_object])
        if target_receptacle:
            args.extend(["--target_receptacle", target_receptacle])
        sys.exit(_run_module("emet.app.run_dynamem", args))
    elif app == "scene-graph":
        args.extend(["--server_ip", server_ip])
        if skip_confirmations:
            args.append("-S")
        if headless:
            args.append("--headless")
        if visual_servo:
            args.append("--visual-servo")
        if target_object:
            args.extend(["--target_object", target_object])
        if target_receptacle:
            args.extend(["--target_receptacle", target_receptacle])
        sys.exit(_run_module("emet.app.run_scene_graph", args))
    elif app == "graph-eqa":
        sys.exit(_run_module("emet.app.run_graph_eqa", args))
    elif app == "dynagraph":
        sys.exit(_run_module("emet.app.run_dynagraph", args))
    elif app == "lazy-graph":
        sys.exit(_run_module("emet.app.run_lazy_graph", args))
    elif app == "graph-eqa-habitat":
        sys.exit(_run_module("emet.app.run_graph_eqa_habitat", args))
    elif app == "molmospaces-explore":
        if headless:
            args.append("--headless")
        sys.exit(_run_module("emet.app.run_molmospaces_explore", args))
    elif app == "mapping":
        sys.exit(_run_module("emet.app.mapping", args))
    elif app == "grasp":
        if target_object:
            args.extend(["--target_object", target_object])
        if parameter_file:
            args.extend(["--parameter_file", parameter_file])
        args.append("--show_gui")
        sys.exit(_run_module("emet.app.grasp_object", args))
    elif app == "chat":
        sys.exit(_run_module("emet.app.chat", args))
    elif app == "agent":
        sys.exit(_run_module("emet.app.run_agent", args))
    elif app == "web-chat":
        sys.exit(_run_module("emet.app.web_chat", args))
    elif app == "ai_pickup":
        if skip_confirmations:
            args.append("-S")
        sys.exit(_run_module("emet.app.ai_pickup", args))
    elif app == "timing":
        args.extend(["--robot_ip", robot_ip])
        if headless:
            args.append("--headless")
        sys.exit(_run_module("emet.app.timing", args))
    elif app == "discord":
        args.extend(["--robot_ip", robot_ip])
        if server_ip:
            args.extend(["--server_ip", server_ip])
        if parameter_file:
            args.extend(["--parameter_file", parameter_file])
        sys.exit(_run_module("emet.app.run_discord", args))
    elif app == "create-and-print-memory":
        args.extend(["--robot-ip", robot_ip])
        sys.exit(_run_module("emet.app.create_and_print_memory", args))
    elif app == "debug-da3-depth":
        sys.exit(_run_module("emet.app.debug_da3_depth", args))
    elif app == "debug-lingbot-depth":
        sys.exit(_run_module("emet.app.debug_lingbot_depth", args))
    else:
        click.echo(f"Unknown app: {app}", err=True)
        sys.exit(1)


_SYNC_ALL_EXTRAS = ("dev", "sim", "hand_tracker", "dynamem", "da3")


@click.command(short_help="Sync dependencies (uv or pip)")
@click.option("--extra", "-e", "extra_list", multiple=True, help="Extra to install (sim, dynamem, dev, etc.)")
@click.option(
    "--all",
    "sync_all",
    is_flag=True,
    help="Install all common extras (same as defaults: dev, sim, hand_tracker, dynamem, da3)",
)
@click.option("--sim", is_flag=True, help="Include sim (MuJoCo, robocasa)")
@click.option("--dynamem", "dynamem_flag", is_flag=True, help="Include dynamem (SAM-2)")
@click.option("--dev", "dev_flag", is_flag=True, help="Include dev (pytest, black, mypy)")
@click.option("--hand-tracker", is_flag=True, help="Include hand_tracker (mediapipe)")
@click.option("--no-install", is_flag=True, help="Only sync lockfile, do not install emet")
def sync(
    extra_list: tuple[str, ...],
    sync_all: bool,
    sim: bool,
    dynamem_flag: bool,
    dev_flag: bool,
    hand_tracker: bool,
    no_install: bool,
) -> None:
    """Sync dependencies (uv sync or pip install -e .).

    With **uv**, a plain ``emet sync`` runs ``uv sync``, which installs
    **default dependency groups** from ``pyproject.toml`` (dev, sim, hand_tracker,
    dynamem, da3). Use ``uv sync --no-default-groups`` for base dependencies only.

    robosuite/robocasa are installed editable by: ``emet install sim`` (not from the lockfile).
    A plain ``emet sync`` reinstalls them when present under ``third_party/`` (default groups
    include sim). ``emet sync -e dynamem`` alone does not.

    Examples:

      emet sync
      emet sync --all
      emet sync -e sim -e dynamem
      emet sync --sim --dynamem
    """
    extras: list[str] = list(extra_list)
    if sync_all:
        extras.extend(_SYNC_ALL_EXTRAS)
    if sim:
        extras.append("sim")
    if dynamem_flag:
        extras.append("dynamem")
    if dev_flag:
        extras.append("dev")
    if hand_tracker:
        extras.append("hand_tracker")
    # Deduplicate, preserve order
    extras = list(dict.fromkeys(extras))

    root = _project_root()
    os.chdir(root)

    # Sim extra: pip-installable deps (mujoco, stretch-urdf, etc.) are in pyproject.toml.
    # robosuite/robocasa live in third_party/ and are *not* lockfile deps — ``uv sync`` removes them unless
    # we reinstall with ``uv pip install -e`` after sync whenever sim is part of this sync.
    want_sim_third_party = (not extras) or ("sim" in extras)
    if want_sim_third_party:
        missing = [name for name in ("robosuite", "robocasa") if not (root / "third_party" / name).is_dir()]
        if missing:
            click.echo(
                f"Note: third_party missing ({', '.join(missing)}). "
                "Sim pip deps (mujoco, etc.) will install, but robosuite/robocasa need: emet install sim",
                err=True,
            )

    if extras:
        click.echo("Syncing extras: " + ", ".join(extras))

    if _has_uv():
        cmd = ["uv", "sync"]
        for e in extras:
            cmd.extend(["--extra", e])
        if no_install:
            cmd.append("--no-install-project")
        result = subprocess.call(cmd)
        if result != 0:
            sys.exit(result)
        if want_sim_third_party:
            result = _uv_pip_install_sim_third_party_editables(root)
            if result != 0:
                sys.exit(result)
        sys.exit(result)
    else:
        # Fallback to pip
        extras_str = "[" + ",".join(extras) + "]" if extras else ""
        spec = f".{extras_str}"
        result = subprocess.call([sys.executable, "-m", "pip", "install", "-e", spec])
        sys.exit(result)


@click.command(short_help="View Rerun logs (.rrd)")
@click.argument("path", type=click.Path(exists=True))
@click.option("--web", is_flag=True, help="Open in web viewer (default: native)")
def show(path: str, web: bool) -> None:
    """View Rerun logs (.rrd) or other visualization data.

    Examples:

      emet show data_0.rrd
      emet show logs/run_001.rrd --web
    """
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        click.echo(f"File not found: {path_obj}", err=True)
        sys.exit(1)

    suffix = path_obj.suffix.lower()
    if suffix == ".rrd":
        env = os.environ.copy()
        if web:
            env["RERUN_VIEWER"] = "web"
        result = subprocess.call(
            [sys.executable, "-m", "rerun", str(path_obj)],
            env=env,
        )
        sys.exit(result)
    else:
        click.echo(f"Unknown format: {suffix}. Supported: .rrd (Rerun)", err=True)
        sys.exit(1)


_AGENT_REGRESSION_PATHS: tuple[str, ...] = (
    "src/test/agent/test_agent_prompt_and_tools.py",
    "src/test/agent/test_dispatch_tool_calls.py",
    "src/test/agent/test_run_agent_loop_mock.py",
    "src/test/agent/test_call_llm.py",
    "src/test/agent/test_thinking_status.py",
    "src/test/agent/test_dynagraph_import_cycle.py",
    "src/test/agent/test_manual_find_command.py",
    "src/test/cli/test_run_agent_defaults.py",
    "src/test/app/test_stream_dynav_resolve.py",
    "src/test/controller/test_graph_eqa_answer_only.py",
    "src/test/eval/test_agentic_eqa_verification.py",
    "src/test/memory/test_memory_backends_smoke.py",
    "src/test/memory/test_graph_eqa_beliefs.py",
)


@click.command(
    short_help="Run pytest (use uv: uv run emet test)",
    context_settings={**_CONTEXT_SETTINGS, "ignore_unknown_options": True},
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-cov", "no_cov", is_flag=True, help="Disable coverage")
@click.option(
    "--no-sim",
    "no_sim_tests",
    is_flag=True,
    help="Disable sim tests (RUN_SIM_TESTS=0); sim tests run by default",
)
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def test(
    verbose: bool,
    no_cov: bool,
    no_sim_tests: bool,
    pytest_args: tuple[str, ...],
) -> None:
    """Run tests with pytest.

    Uses the project environment: run from repo root with uv so dev deps (pytest,
    pytest-timeout) are available. Sim tests (e.g. red cylinder in MuJoCo) run by default;
    use --no-sim to skip them for a faster run.

      uv sync
      uv run emet test
      uv run emet test -v
      uv run emet test --no-sim           # skip sim tests (faster)
      uv run emet test agent-regression   # Discord / Herman / agent pack (no sim)
      uv run emet test -v src/test/memory/test_memory_backends_smoke.py
      uv run emet test src/test/mapping/test_red_cylinder_in_sim.py -k innate_mars
      uv run emet test -k test_red_cylinder
      Heavy VLLM tests (@pytest.mark.vllm_load) are excluded by default; see docs/plans/TESTING_VLLM_LOAD.md
    """
    root = _project_root()
    os.chdir(root)
    env = os.environ.copy()
    args = list(pytest_args)
    quiet = False
    if args and args[0] == "agent-regression":
        rest = args[1:]
        args = ["-q", *_AGENT_REGRESSION_PATHS, "-m", "not sim", *rest]
        no_sim_tests = True
        quiet = True
        no_cov = True
    if no_sim_tests:
        env["RUN_SIM_TESTS"] = "0"
    else:
        env["RUN_SIM_TESTS"] = "1"
    # Prefer project .venv so pytest and deps match the project (e.g. pytest-timeout)
    venv_py = _project_venv_python()
    python = str(venv_py) if venv_py is not None else sys.executable
    src = root / "src"
    if src.exists():
        # Prepend project src; drop ROS site-packages so ament-* pytest plugins
        # (auto-loaded via PYTHONPATH) do not break src/test/habitat collection.
        prev_parts = [
            p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and "/opt/ros/" not in p.replace("\\", "/")
        ]
        env["PYTHONPATH"] = os.pathsep.join([str(src), *prev_parts])

    cmd = [python, "-m", "pytest"]
    if verbose and not quiet:
        cmd.append("-v")
    if not no_cov and (root / "pyproject.toml").exists():
        try:
            import pytest_cov  # noqa: F401

            cmd.extend(["--cov=emet", "--cov-report=term-missing"])
        except ImportError:
            pass
    cmd.extend(args)
    if not args:
        # pytest uses testpaths from pyproject.toml ([tool.pytest.ini_options] testpaths = ["src/test"])
        pass
    sys.exit(subprocess.call(cmd, env=env, cwd=root))


_SIM_THIRD_PARTY_DIRS = ("robosuite", "robosuite_models", "robocasa")


def _third_party_sim_editable_paths(root: Path) -> list[Path]:
    """Paths under ``third_party/`` to install with ``uv pip install -e`` (not declared in uv.lock)."""
    return [root / "third_party" / name for name in _SIM_THIRD_PARTY_DIRS if (root / "third_party" / name).is_dir()]


def _uv_pip_install_sim_third_party_editables(root: Path) -> int:
    """Re-register robosuite / robosuite_models / robocasa after ``uv sync`` (sync prunes undeclared packages)."""
    paths = _third_party_sim_editable_paths(root)
    if not paths or not _has_uv():
        return 0
    click.echo("Installing robosuite/robocasa from third_party (not in uv.lock)...")
    args: list[str] = []
    for p in paths:
        args.extend(["-e", str(p)])
    # Do not use ``--no-deps``: robocasa/robosuite need install_requires (e.g. h5py, imageio).
    return subprocess.call(["uv", "pip", "install"] + args, cwd=root)


@click.command("clean", short_help="Remove third-party sim clones (robosuite, robocasa, etc.)")
@click.option(
    "-y",
    "--yes",
    "skip_confirm",
    is_flag=True,
    help="Skip confirmation; remove without prompting",
)
def clean(skip_confirm: bool) -> None:
    """Remove third-party directories created by emet install sim / install robocasa.

    Deletes third_party/robosuite, third_party/robosuite_models, and third_party/robocasa.
    Re-run emet install sim (or install robocasa) to clone and install them again.

    Examples:
      emet clean
      emet clean -y
    """
    root = _project_root()
    third_party = root / "third_party"
    if not third_party.exists():
        click.echo("Nothing to clean: third_party/ does not exist.")
        return
    to_remove = [third_party / name for name in _SIM_THIRD_PARTY_DIRS if (third_party / name).exists()]
    if not to_remove:
        click.echo("Nothing to clean: none of third_party/robosuite, robosuite_models, robocasa exist.")
        return
    if not skip_confirm:
        click.echo("The following directories will be removed:")
        for p in to_remove:
            click.echo(f"  {p}")
        click.confirm("Continue?", default=False, abort=True)
    for p in to_remove:
        click.echo(f"Removing {p}...")
        shutil.rmtree(p)
    click.echo("Done. Re-run emet install sim to reinstall.")


@click.group(
    short_help="Install submodules, sim, full setup, pre-commit",
    invoke_without_command=True,
)
@click.pass_context
def install(ctx: click.Context) -> None:
    """Install submodules, simulation extras, or full setup.

    With no subcommand, opens the interactive install menu (manage sub-assets and sync).
    Use a subcommand for direct install: emet install sim, emet install full, etc.
    """
    if ctx.invoked_subcommand is not None:
        return
    from emet.install_ui import run_install_menu

    sys.exit(run_install_menu())


@install.command("gh", short_help="Install GitHub CLI (apt)")
@click.option("-y", "--yes", "non_interactive", is_flag=True, help="Skip confirmation; sudo may still prompt")
def install_gh(non_interactive: bool) -> None:
    """Install the GitHub CLI (``gh``) for pull requests and issues.

    Package name is declared in ``pyproject.toml`` under ``[tool.emet.system-packages]``.
    After install, authenticate once: ``gh auth login``.

    Examples:
      emet install gh -y
    """
    from emet.dev_system_packages import ensure_apt_package

    sys.exit(ensure_apt_package("gh", non_interactive=non_interactive))


@install.command("paper", short_help="Install LaTeX paper toolchain (apt)")
@click.option("-y", "--yes", "non_interactive", is_flag=True, help="Skip confirmation; sudo may still prompt")
def install_paper(non_interactive: bool) -> None:
    """Install ``latexmk`` and the TeX Live packages used by ``paper/main.tex``.

    Package names are declared in ``pyproject.toml`` under
    ``[tool.emet.system-packages]``.

    Examples:
      emet install paper -y
      ./paper/build.sh
    """
    from emet.dev_system_packages import ensure_apt_package

    sys.exit(ensure_apt_package("latexmk", non_interactive=non_interactive))


@install.command("submodules", short_help="Init and update git submodules")
@click.option("--recursive/--no-recursive", default=True, help="Recursively init submodules")
def install_submodules(recursive: bool) -> None:
    """Init and update git submodules (segment-anything-2, ok-robot, etc.).

    Examples:
      emet install submodules
      emet install submodules --no-recursive
    """
    root = _project_root()
    os.chdir(root)
    cmd = ["git", "submodule", "update", "--init"]
    if recursive:
        cmd.append("--recursive")
    result = subprocess.call(cmd)
    if result == 0:
        click.echo("Submodules initialized and updated.")
    sys.exit(result)


def _run_install_simulation(
    root: Path,
    skip_download_assets: bool = False,
    setup_macros: bool = False,
) -> int:
    """Run scripts/install_simulation.sh (robosuite + robocasa). Returns exit code.
    Asset download is part of installation by default; pass skip_download_assets to omit.
    """
    script = root / "scripts" / "install_simulation.sh"
    if not script.exists():
        click.echo(f"Script not found: {script}", err=True)
        return 1
    args = []
    if skip_download_assets:
        args.append("-n")
    if setup_macros:
        args.append("-a")
    env = os.environ.copy()
    env["EMET_PYTHON"] = sys.executable
    if _has_uv():
        env["EMET_USE_UV"] = "1"
    return subprocess.call(["bash", str(script)] + args, env=env)


@install.command("sim", short_help="Install Robocasa, robosuite")
@click.option(
    "-n",
    "--no-download-assets",
    "skip_download_assets",
    is_flag=True,
    help="Skip downloading Robocasa kitchen assets (~10GB)",
)
@click.option(
    "-a",
    "--setup-macros",
    is_flag=True,
    help="Force run macro setup (overwrite existing macros_private.py); by default macros are set up only when missing",
)
@click.option("--no-sync", is_flag=True, help="Skip uv sync and third_party reinstall after clone/install")
def install_sim(skip_download_assets: bool, setup_macros: bool, no_sync: bool) -> None:
    """Install simulation third-party deps (Robocasa + robosuite).

    Clones robosuite and robocasa, runs macro setup when missing (silences
    "No private macro file" warnings), downloads kitchen assets (~10GB), then
    runs ``uv sync`` and reinstalls editable ``third_party/`` robosuite /
    robosuite_models / robocasa (not in uv.lock; sync would otherwise remove them).
    Use -n to skip the asset download (e.g. CI).

    Examples:
      emet install sim
      emet install sim -a
      emet install sim --no-download-assets --no-sync
    """
    root = _project_root()
    result = _run_install_simulation(root, skip_download_assets, setup_macros)
    if result != 0:
        sys.exit(result)
    if no_sync:
        click.echo("Simulation install complete. Run: uv sync")
        return
    click.echo("Syncing project env (uv sync); then reinstalling third_party sim packages...")
    os.chdir(root)
    if _has_uv():
        result = subprocess.call(["uv", "sync"], cwd=root)
        if result == 0:
            result = _uv_pip_install_sim_third_party_editables(root)
    else:
        result = subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".[sim]"])
    if result == 0:
        click.echo("Simulation install complete.")
        click.echo("Verify: uv run python -c \"import robocasa; print('robocasa OK')\"")
    sys.exit(result)


@install.command("robocasa", short_help="Install Robocasa (same as install sim)")
@click.option(
    "-n",
    "--no-download-assets",
    "skip_download_assets",
    is_flag=True,
    help="Skip downloading Robocasa kitchen assets (~10GB)",
)
@click.option(
    "-a",
    "--setup-macros",
    is_flag=True,
    help="Force run macro setup (overwrite existing macros_private.py); by default macros are set up only when missing",
)
@click.option("--no-sync", is_flag=True, help="Skip uv sync and third_party reinstall after clone/install")
def install_robocasa(skip_download_assets: bool, setup_macros: bool, no_sync: bool) -> None:
    """Install Robocasa and robosuite (same as emet install sim).

    Clones robosuite and robocasa, runs macro setup when missing, downloads
    kitchen assets (~10GB), then runs ``uv sync`` and reinstalls editable
    third_party sim packages. Use -n to skip the asset download.

    Examples:
      emet install robocasa
      emet install robocasa -a
      emet install robocasa --no-download-assets
    """
    root = _project_root()
    result = _run_install_simulation(root, skip_download_assets, setup_macros)
    if result != 0:
        sys.exit(result)
    if no_sync:
        click.echo("Robocasa install complete. Run: uv sync")
        return
    click.echo("Syncing project env (uv sync); then reinstalling third_party sim packages...")
    os.chdir(root)
    if _has_uv():
        result = subprocess.call(["uv", "sync"], cwd=root)
        if result == 0:
            result = _uv_pip_install_sim_third_party_editables(root)
    else:
        result = subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".[sim]"])
    if result == 0:
        click.echo("Robocasa install complete.")
        click.echo("Verify: uv run python -c \"import robocasa; print('robocasa OK')\"")
    sys.exit(result)


@install.command("menu", short_help="Interactive menu to manage sub-assets")
@click.option(
    "--text-only",
    "text_only",
    is_flag=True,
    help="ASCII menu only (skip Rich plan wizard when Rich is installed).",
)
def install_menu(text_only: bool) -> None:
    """Open a menu to install or update sub-assets.

    With Rich (dev extra), starts a colored plan: choose defaults, then review
    before running commands. Use --text-only for the legacy ASCII asset menu only.

    Examples:
      emet install menu
      emet install menu --text-only
    """
    from emet.install_ui import run_install_menu

    sys.exit(run_install_menu(text_only=text_only))


@install.command("full", short_help="Run full install (install.sh)")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompts (non-interactive apt, link emet)")
@click.option(
    "--profile",
    type=click.Choice(["minimal", "standard", "full", "jetson"], case_sensitive=False),
    default=None,
    help=(
        "Install profile forwarded to install.sh / EMET_INSTALL_PROFILE: "
        "full (default)=sim-on; standard/minimal=no sim unless --sim; "
        "jetson=Orin/Tegra lean (MuJoCo pip + dev; no SAM2/Molmo/Robocasa)."
    ),
)
@click.option("--sim", is_flag=True, help="Include simulation extras")
@click.option("--cpu", is_flag=True, help="CPU-only (skip SAM2)")
@click.option("--no-sam2", is_flag=True, help="Skip Segment Anything 2")
@click.option(
    "--molmospaces",
    is_flag=True,
    help="Create .venv-molmospaces (MolmoSpaces; Python 3.11+). Forwards to install.sh --molmospaces",
)
@click.option(
    "--no-molmospaces",
    "no_molmospaces",
    is_flag=True,
    help="Skip MolmoSpaces wrapper even when sim installs (forwards to install.sh --no-molmospaces).",
)
@click.option("--paper", is_flag=True, help="Install latexmk and the TeX Live packages used to build the paper")
@click.option("--no-paper", is_flag=True, help="Skip paper tooling when combined with --all")
@click.option(
    "--all",
    "install_all",
    is_flag=True,
    help="Same as install.sh --all (includes simulation, MolmoSpaces, and paper tooling)",
)
def install_full(
    yes: bool,
    profile: str | None,
    sim: bool,
    cpu: bool,
    no_sam2: bool,
    molmospaces: bool,
    no_molmospaces: bool,
    paper: bool,
    no_paper: bool,
    install_all: bool,
) -> None:
    """Run full install (./install.sh).

    Installs uv, system deps, git-lfs, and syncs dependencies.

    The default ``full`` profile installs Robocasa/simulation and the MolmoSpaces
    wrapper. Paper tooling remains optional; pass ``--paper`` or ``--all``.

    Examples:
      emet install full
      emet install full -y --sim
      emet install full -y --profile full
      emet install full -y --profile jetson
      emet install full -y --no-molmospaces
      emet install full -y --molmospaces
      emet install full -y --paper
      emet install full -y --all
      emet install full --cpu
    """
    root = _project_root()
    script = root / "install.sh"
    if not script.exists():
        click.echo(f"install.sh not found: {script}", err=True)
        sys.exit(1)
    args = []
    if yes:
        args.append("-y")
    if profile:
        args.append(f"--profile={profile}")
    if sim:
        args.append("--sim")
    if cpu:
        args.append("--cpu")
    if no_sam2:
        args.append("--no-sam2")
    if install_all:
        args.append("--all")
    if molmospaces:
        args.append("--molmospaces")
    if no_molmospaces:
        args.append("--no-molmospaces")
    if paper:
        args.append("--paper")
    if no_paper:
        args.append("--no-paper")
    env = os.environ.copy()
    if profile:
        env["EMET_INSTALL_PROFILE"] = profile.lower()
    result = subprocess.call(["bash", str(script)] + args, env=env)
    sys.exit(result)


@install.command("pre-commit", short_help="Install pre-commit hooks")
@click.option("--install-hooks", is_flag=True, default=True, help="Install git hook scripts")
@click.option("--run/--no-run", "run_hooks", default=False, help="Run hooks on all files after install")
def install_pre_commit(install_hooks: bool, run_hooks: bool) -> None:
    """Install pre-commit hooks (lint, format, type-check).

    Requires: emet sync --dev (installs pre-commit).

    Examples:
      emet install pre-commit
      emet install pre-commit --run
    """
    root = _project_root()
    os.chdir(root)
    try:
        import pre_commit  # noqa: F401
    except ImportError:
        click.echo("pre-commit not installed. Run: emet sync --dev", err=True)
        sys.exit(1)
    if install_hooks:
        result = subprocess.call([sys.executable, "-m", "pre_commit", "install"])
        if result != 0:
            sys.exit(result)
        click.echo("Pre-commit hooks installed.")
    if run_hooks:
        result = subprocess.call([sys.executable, "-m", "pre_commit", "run", "--all-files"])
        sys.exit(result)
    sys.exit(0)


def register(main: click.Group) -> None:
    main.add_command(dataset_cmd)
    main.add_command(molmobot_cmd)
    main.add_command(robots_cmd)
    main.add_command(show_memory)
    main.add_command(graph_memory_show)
    main.add_command(reprocess_graph_eqa_cache)
    main.add_command(print_memory)
    main.add_command(run)
    main.add_command(sync)
    main.add_command(show)
    main.add_command(test)
    main.add_command(clean)
    main.add_command(install)
