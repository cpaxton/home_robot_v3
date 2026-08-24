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
# Text-based UI for managing installation of sub-assets (sim, kitchen assets,
# MolmoSpaces, SAM-2, etc.). Used by `emet install menu`.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from emet.dev_system_packages import ensure_apt_package


@dataclass
class WizardPlan:
    """User-selected install steps (Rich wizard or defaults)."""

    submodules: bool = True
    uv_dev: bool = True
    include_dynamem: bool = True
    uv_sim: bool = False
    install_simulation: bool = False
    molmospaces: bool = False
    paper: bool = False

    def __post_init__(self) -> None:
        if self.install_simulation:
            self.uv_sim = True


@dataclass
class AssetStatus:
    """Status of an installable asset."""

    id: str
    name: str
    description: str
    status: str  # "installed", "missing", "can_update", "optional"
    detail: str = ""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _has_uv() -> bool:
    try:
        subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _molmospaces_ensure_py311_venv(root: Path) -> None:
    """MolmoSpaces upstream requires Python >=3.11; recreate an older .venv-molmospaces."""
    venv = root / ".venv-molmospaces"
    py = venv / "bin" / "python"
    if py.is_file():
        r = subprocess.run(
            [str(py), "-c", "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)"],
            cwd=root,
            capture_output=True,
        )
        if r.returncode != 0:
            print("  Replacing .venv-molmospaces (molmo-spaces requires Python >=3.11)...")
            shutil.rmtree(venv, ignore_errors=True)
    if not venv.is_dir():
        r = subprocess.call(["uv", "venv", ".venv-molmospaces", "--python", "3.11"], cwd=root)
        if r != 0:
            subprocess.call(["uv", "venv", ".venv-molmospaces", "--python", "3.12"], cwd=root)


def _molmospaces_pip_install_chain(py_molmo: Path, root: Path, use_uv: str | None) -> None:
    """Install local emet (--no-deps), then emet-molmospaces (pulls molmo-spaces from GitHub). emet is not on PyPI."""
    if use_uv:
        subprocess.call(
            ["uv", "pip", "install", "--python", str(py_molmo), "--upgrade", "pip"],
            cwd=root,
        )
        subprocess.call(["uv", "pip", "install", "--python", str(py_molmo), "--no-deps", "-e", "."], cwd=root)
        wrapper_dir = root / "packages" / "emet_molmospaces"
        if wrapper_dir.exists():
            subprocess.call(["uv", "pip", "install", "--python", str(py_molmo), "-e", str(wrapper_dir)], cwd=root)
        else:
            subprocess.call(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(py_molmo),
                    "molmo-spaces @ git+https://github.com/allenai/molmospaces.git@62b416089b2eddff339e52a32106a6bc08ed92b1",
                    "mujoco>=3.4",
                    "numpy>=2.2",
                ],
                cwd=root,
            )
    else:
        subprocess.call([str(py_molmo), "-m", "pip", "install", "--upgrade", "pip"], cwd=root)
        subprocess.call([str(py_molmo), "-m", "pip", "install", "--no-deps", "-e", "."], cwd=root)
        wrapper_dir = root / "packages" / "emet_molmospaces"
        if wrapper_dir.exists():
            subprocess.call([str(py_molmo), "-m", "pip", "install", "-e", str(wrapper_dir)], cwd=root)
        else:
            subprocess.call(
                [
                    str(py_molmo),
                    "-m",
                    "pip",
                    "install",
                    "molmo-spaces @ git+https://github.com/allenai/molmospaces.git@62b416089b2eddff339e52a32106a6bc08ed92b1",
                    "mujoco>=3.4",
                    "numpy>=2.2",
                ],
                cwd=root,
            )


def _molmospaces_venv_needs_install_or_repair(root: Path, py_molmo: Path) -> bool:
    if not py_molmo.is_file():
        return True
    r = subprocess.run(
        [
            str(py_molmo),
            "-c",
            "import emet; import emet_molmospaces; "
            "from molmo_spaces.molmo_spaces_constants import get_scenes; "
            "from molmo_spaces.utils.lazy_loading_utils import "
            "install_scene_with_objects_and_grasps_from_path",
        ],
        cwd=root,
        capture_output=True,
    )
    if r.returncode != 0:
        return True
    exe = root / ".venv-molmospaces" / "bin" / "emet-molmospaces"
    return not exe.is_file()


def _check_submodules(root: Path) -> AssetStatus:
    sam2 = root / "third_party" / "segment-anything-2"
    if not sam2.exists():
        return AssetStatus(
            id="submodules",
            name="Submodules (SAM-2 / segment-anything-2)",
            description="Required for dynamem; git submodule under third_party/.",
            status="missing",
            detail="Run: git submodule update --init --recursive third_party/segment-anything-2",
        )
    return AssetStatus(
        id="submodules",
        name="Submodules (SAM-2 / segment-anything-2)",
        description="Required for dynamem.",
        status="installed",
        detail=str(sam2),
    )


def _check_sim(root: Path) -> AssetStatus:
    robosuite = root / "third_party" / "robosuite"
    robocasa = root / "third_party" / "robocasa"
    if not robosuite.exists() or not robocasa.exists():
        missing = [d.name for d in (robosuite, robocasa) if not d.exists()]
        return AssetStatus(
            id="sim",
            name="Simulation (robosuite + robocasa)",
            description="MuJoCo kitchen scenes; required for emet serve robocasa.",
            status="missing",
            detail=f"Missing: {', '.join(missing)}. Run: emet install sim",
        )
    return AssetStatus(
        id="sim",
        name="Simulation (robosuite + robocasa)",
        description="MuJoCo kitchen scenes.",
        status="installed",
        detail="Can update in place (fetch/pull).",
    )


def _check_kitchen_assets(root: Path) -> AssetStatus:
    assets = root / "third_party" / "robocasa" / "robocasa" / "models" / "assets"
    textures = assets / "textures"
    if not assets.exists() or not textures.exists():
        if not (root / "third_party" / "robocasa").exists():
            return AssetStatus(
                id="kitchen_assets",
                name="Kitchen assets (textures, fixtures, ~10GB)",
                description="Robocasa scene assets. Install sim first.",
                status="missing",
                detail="Install simulation first: emet install sim",
            )
        return AssetStatus(
            id="kitchen_assets",
            name="Kitchen assets (textures, fixtures, ~10GB)",
            description="Robocasa scene assets.",
            status="missing",
            detail="Run: emet install sim (will prompt to download), or scripts/download_robocasa_assets.py",
        )
    has_content = next(textures.iterdir(), None) is not None
    if not has_content:
        return AssetStatus(
            id="kitchen_assets",
            name="Kitchen assets (textures, fixtures, ~10GB)",
            description="Robocasa scene assets.",
            status="missing",
            detail="Directory exists but empty; re-run asset download.",
        )
    return AssetStatus(
        id="kitchen_assets",
        name="Kitchen assets (textures, fixtures, ~10GB)",
        description="Robocasa scene assets.",
        status="installed",
        detail="Re-download? Use: python scripts/download_robocasa_assets.py --force",
    )


def _check_molmospaces(root: Path) -> AssetStatus:
    venv = root / ".venv-molmospaces"
    wrapper_exe = venv / "bin" / "emet-molmospaces"
    if not wrapper_exe.exists():
        return AssetStatus(
            id="molmospaces",
            name="MolmoSpaces wrapper (.venv-molmospaces / emet-molmospaces)",
            description="Scenes + rby1 robot; thin wrapper (molmo-spaces, mujoco 3.4).",
            status="missing",
            detail="Run: ./install.sh -y  (sim + MolmoSpaces wrapper by default) or  ./install.sh --molmospaces -y",
        )
    return AssetStatus(
        id="molmospaces",
        name="MolmoSpaces wrapper (.venv-molmospaces / emet-molmospaces)",
        description="Scenes + rby1 robot.",
        status="installed",
        detail=str(venv),
    )


def _check_paper(_: Path) -> AssetStatus:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        return AssetStatus(
            id="paper",
            name="Paper toolchain (latexmk + TeX Live)",
            description="Builds paper/main.tex locally; optional Docker fallback remains available.",
            status="missing",
            detail="Run: emet install paper",
        )
    return AssetStatus(
        id="paper",
        name="Paper toolchain (latexmk + TeX Live)",
        description="Builds paper/main.tex locally.",
        status="installed",
        detail=latexmk,
    )


def _get_all_statuses(root: Path) -> list[AssetStatus]:
    return [
        _check_submodules(root),
        _check_sim(root),
        _check_kitchen_assets(root),
        _check_molmospaces(root),
        _check_paper(root),
    ]


def _draw_box(lines: list[str], width: int = 72) -> list[str]:
    top = "┌" + "─" * (width - 2) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    out = [top]
    content_width = width - 4
    for line in lines:
        padded = (line[:content_width] if len(line) > content_width else line).ljust(content_width)
        out.append("│ " + padded + " │")
    out.append(bottom)
    return out


def _run_script(script: Path, args: list[str], env: dict | None = None) -> int:
    env = env or os.environ.copy()
    return subprocess.call(["bash", str(script)] + args, env=env)


def _execute_wizard_plan(root: Path, plan: WizardPlan) -> int:
    """Run git / uv / bash steps selected in the wizard. Returns shell exit code (0 ok)."""
    env = os.environ.copy()
    env["EMET_PYTHON"] = sys.executable
    if _has_uv():
        env["EMET_USE_UV"] = "1"
    code = 0
    if plan.submodules:
        print("\n--- Git submodule (SAM-2) ---")
        code = subprocess.call(
            ["git", "submodule", "update", "--init", "--recursive", "third_party/segment-anything-2"],
            cwd=root,
        )
        if code != 0:
            return code
    if plan.uv_dev or plan.uv_sim or plan.include_dynamem:
        if not _has_uv():
            print("uv not found; install uv or run sync manually.")
            return 1
        cmd = ["uv", "sync"]
        if not plan.uv_dev:
            cmd.extend(["--no-group", "dev"])
        if not plan.uv_sim:
            cmd.extend(["--no-group", "sim"])
        dynamem_ok = plan.include_dynamem and (root / "third_party" / "segment-anything-2").is_dir()
        if not dynamem_ok:
            cmd.extend(["--no-group", "dynamem"])
        label = "defaults" if (plan.uv_dev and plan.uv_sim and dynamem_ok) else "custom groups"
        print(f"\n--- uv sync ({label}) ---\n  {' '.join(cmd)}")
        code = subprocess.call(cmd, cwd=root)
        if code != 0:
            return code
    if plan.install_simulation:
        script_sim = root / "scripts" / "install_simulation.sh"
        print("\n--- Simulation installer (robosuite + robocasa; may download assets) ---")
        if not script_sim.is_file():
            print(f"Missing: {script_sim}")
            return 1
        code = _run_script(script_sim, [], env=env)
        if code != 0:
            return code
    if plan.molmospaces:
        print("\n--- MolmoSpaces venv ---")
        venv_molmo = root / ".venv-molmospaces"
        use_uv = shutil.which("uv")
        _molmospaces_ensure_py311_venv(root)
        py_molmo = venv_molmo / "bin" / "python"
        if _molmospaces_venv_needs_install_or_repair(root, py_molmo):
            print("  Installing / repairing .venv-molmospaces...")
            _molmospaces_pip_install_chain(py_molmo, root, use_uv)
        else:
            print("  .venv-molmospaces already looks complete.")
    if plan.paper:
        print("\n--- Paper toolchain (latexmk + TeX Live) ---")
        code = ensure_apt_package("latexmk", non_interactive=True)
        if code != 0:
            return code
    return 0


def _run_rich_plan_wizard(root: Path) -> bool | None:
    """Colored plan + toggles. Returns True if user finished (stay out of legacy menu), False to fall back to ASCII-only, None = skip wizard (no Rich)."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm
        from rich.table import Table
    except ImportError:
        return None

    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Emet install planner[/]\n"
            "[dim]Defaults avoid multi‑GB Robocasa assets. Enable simulation only if you need "
            "`emet serve mujoco` / robocasa.[/dim]\n"
            "[dim]Shell install: [bold]./install.sh[/bold] defaults to the full simulation profile; "
            "use [bold]./install.sh --profile=standard[/bold] for no simulation.[/dim]",
            border_style="cyan",
        )
    )

    plan = WizardPlan()
    plan.submodules = Confirm.ask("1) Init/update SAM-2 submodule (third_party/segment-anything-2)?", default=True)
    plan.include_dynamem = Confirm.ask(
        "2) Include [bold]dynamem[/bold] dependency group (editable SAM-2)?",
        default=plan.submodules,
    )
    plan.uv_dev = Confirm.ask("3) Run [bold]uv sync[/bold] (pytest, ruff, rich, … from default groups)?", default=True)
    plan.uv_sim = Confirm.ask(
        "4) Include [bold]sim[/bold] dependency group (MuJoCo pip deps, grpcio, …)?",
        default=False,
    )
    plan.install_simulation = Confirm.ask(
        "5) Run [bold]scripts/install_simulation.sh[/bold] (clone robosuite/robocasa; large downloads)?",
        default=False,
    )
    plan.molmospaces = Confirm.ask(
        "6) Create [bold].venv-molmospaces[/bold] (MolmoSpaces; Python 3.11+ separate venv)?",
        default=False,
    )
    plan.paper = Confirm.ask(
        "7) Install [bold]latexmk + TeX Live[/bold] to build paper/main.tex locally?",
        default=False,
    )
    if plan.install_simulation:
        plan.uv_sim = True

    table = Table(title="Plan summary", show_lines=True)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Run?", justify="center")
    table.add_column("Notes", style="dim")
    rows = [
        ("SAM-2 submodule", "yes" if plan.submodules else "no", "git submodule update"),
        ("uv dev", "yes" if plan.uv_dev else "no", "pytest, ruff, rich, …"),
        ("uv dynamem", "yes" if plan.include_dynamem else "no", "SAM-2 / segmentation"),
        ("uv sim", "yes" if plan.uv_sim else "no", "mujoco pip extra"),
        ("install_simulation.sh", "yes" if plan.install_simulation else "no", "robosuite + robocasa"),
        ("MolmoSpaces venv", "yes" if plan.molmospaces else "no", ".venv-molmospaces"),
        ("Paper toolchain", "yes" if plan.paper else "no", "latexmk + TeX Live"),
    ]
    for r in rows:
        table.add_row(*r)
    console.print(table)

    if not Confirm.ask("\n[bold green]Run this plan now?[/bold green]", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return True

    code = _execute_wizard_plan(root, plan)
    if code != 0:
        console.print(f"[red]A step failed (exit {code}).[/red]")
    else:
        console.print(
            "[green]Plan finished.[/green] Hint: [bold]uv run emet serve mujoco --headless[/bold] after sim install."
        )

    if Confirm.ask("Open the detailed ASCII asset menu?", default=False):
        return False
    return True


def _run_sync_menu(root: Path) -> None:
    """Prompt for uv sync variant (default dependency groups vs minimal)."""
    if not _has_uv():
        print("uv not found. Run: pip install -e .[sim,dynamem,dev] as needed.")
        return
    print("\nSync:")
    print("  1. uv sync  (default groups: dev, sim, hand_tracker, dynamem, da3)")
    print("  2. uv sync --no-default-groups  (base [project] dependencies only)")
    print("  Q. Cancel")
    try:
        c = input("Choice (1–2, Q) [1]: ").strip().upper() or "1"
    except (EOFError, KeyboardInterrupt):
        return
    if c == "Q":
        return
    if c == "1":
        cmd = ["uv", "sync"]
    elif c == "2":
        cmd = ["uv", "sync", "--no-default-groups"]
    else:
        print("Invalid choice.")
        return
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.call(cmd, cwd=root)
    if result == 0:
        print("Sync complete.")
    else:
        print("Sync failed (check missing third_party or enable_sim_pyproject).")


def _legacy_asset_menu_loop(root: Path) -> int:
    """Original ASCII menu (status + per-asset actions)."""
    statuses = _get_all_statuses(root)
    width = 72

    while True:
        # Header
        print()
        for line in _draw_box(["Emet – Manage sub-assets", ""], width):
            print(line)
        print()

        # Menu
        for i, s in enumerate(statuses, 1):
            icon = "✓" if s.status == "installed" else "○" if s.status == "missing" else "↻"
            print(f"  {i}. {icon} {s.name}")
            print(f"      {s.description}")
            if s.detail:
                print(f"      → {s.detail}")
            print()

        print("  A. Run all (submodules, sim, kitchen assets, MolmoSpaces, paper tooling – with prompts)")
        print("  S. Sync (uv: full defaults or base-only)")
        print("  Q. Quit")
        print()

        try:
            choice = input(f"Choice (1–{len(statuses)}, A, S, Q) [Q]: ").strip().upper() or "Q"
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice == "Q":
            return 0

        if choice == "S":
            _run_sync_menu(root)
            print()
            input("Press Enter to return to menu...")
            continue

        if choice == "A":
            # Run all: submodules, then sim (with assets), then molmospaces
            print("\n--- Submodules ---")
            subprocess.call(
                ["git", "submodule", "update", "--init", "--recursive", "third_party/segment-anything-2"], cwd=root
            )
            print("\n--- Simulation (robosuite + robocasa + assets) ---")
            script_sim = root / "scripts" / "install_simulation.sh"
            if script_sim.exists():
                env = os.environ.copy()
                env["EMET_PYTHON"] = sys.executable
                if _has_uv():
                    env["EMET_USE_UV"] = "1"
                _run_script(script_sim, [], env=env)
            print("\n--- MolmoSpaces venv (emet-molmospaces wrapper) ---")
            venv_molmo = root / ".venv-molmospaces"
            py_molmo = venv_molmo / "bin" / "python"
            use_uv = shutil.which("uv") if shutil else None
            _molmospaces_ensure_py311_venv(root)
            py_molmo = venv_molmo / "bin" / "python"
            if _molmospaces_venv_needs_install_or_repair(root, py_molmo):
                print("  Installing / repairing .venv-molmospaces (editable emet + wrapper + molmo-spaces)...")
                _molmospaces_pip_install_chain(py_molmo, root, use_uv)
            else:
                print("  .venv-molmospaces already complete.")
            print("\n--- Paper toolchain (latexmk + TeX Live) ---")
            ensure_apt_package("latexmk")
            print("\nDone. Run uv sync from the repo root if you have not already.")
            continue

        try:
            idx = int(choice)
        except ValueError:
            print("Invalid choice.")
            continue

        if idx < 1 or idx > len(statuses):
            print("Invalid choice.")
            continue

        s = statuses[idx - 1]
        env = os.environ.copy()
        env["EMET_PYTHON"] = sys.executable
        if _has_uv():
            env["EMET_USE_UV"] = "1"

        if s.id == "submodules":
            print("\nRunning: git submodule update --init --recursive third_party/segment-anything-2")
            subprocess.call(
                ["git", "submodule", "update", "--init", "--recursive", "third_party/segment-anything-2"], cwd=root
            )
        elif s.id == "sim":
            script = root / "scripts" / "install_simulation.sh"
            if not script.exists():
                print(f"Script not found: {script}")
            else:
                _run_script(script, [], env=env)
            print("Then run: emet sync -e sim")
        elif s.id == "kitchen_assets":
            script_py = root / "scripts" / "download_robocasa_assets.py"
            if script_py.exists():
                print("\nRunning: python scripts/download_robocasa_assets.py")
                subprocess.call([sys.executable, str(script_py)], cwd=root)
            else:
                print("Run: emet install sim  (includes asset download prompt)")
        elif s.id == "molmospaces":
            venv_molmo = root / ".venv-molmospaces"
            use_uv = shutil.which("uv") if shutil else None
            _molmospaces_ensure_py311_venv(root)
            py_molmo = venv_molmo / "bin" / "python"
            if _molmospaces_venv_needs_install_or_repair(root, py_molmo):
                print("\nInstalling / repairing MolmoSpaces venv (editable emet + wrapper + molmo-spaces)...")
                _molmospaces_pip_install_chain(py_molmo, root, use_uv)
            else:
                print(".venv-molmospaces already has emet-molmospaces wrapper.")
            print(
                "Set MLSPACES_ASSETS_DIR for scene data (e.g. ~/.cache/molmospaces/assets) and "
                "MLSPACES_CACHE_DIR for extracted archives (e.g. ~/.cache/molmospaces/resource_cache); "
                "they must differ (upstream ResourceManager)."
            )
        elif s.id == "paper":
            ensure_apt_package("latexmk")

        print()
        input("Press Enter to return to menu...")

    return 0


def run_install_menu(*, text_only: bool = False) -> int:
    """Rich plan wizard when available; then optional legacy ASCII asset menu."""
    root = _project_root()
    os.chdir(root)

    if not text_only:
        rich_result = _run_rich_plan_wizard(root)
        if rich_result is True:
            return 0
        if rich_result is None:
            print("Tip: install Rich for a colored plan wizard:  uv sync")
    return _legacy_asset_menu_loop(root)
