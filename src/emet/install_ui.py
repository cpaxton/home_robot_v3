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
            detail="Run: ./install.sh --molmospaces -y  or  emet install full --all",
        )
    return AssetStatus(
        id="molmospaces",
        name="MolmoSpaces wrapper (.venv-molmospaces / emet-molmospaces)",
        description="Scenes + rby1 robot.",
        status="installed",
        detail=str(venv),
    )


def _get_all_statuses(root: Path) -> list[AssetStatus]:
    return [
        _check_submodules(root),
        _check_sim(root),
        _check_kitchen_assets(root),
        _check_molmospaces(root),
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


def _run_sync_menu(root: Path) -> None:
    """Prompt for extras and run uv sync (or pip)."""
    if not _has_uv():
        print("uv not found. Run: pip install -e .[sim,dynamem,dev] as needed.")
        return
    print("\nSync extras:")
    print("  1. sim      (MuJoCo, robocasa – requires third_party/robocasa)")
    print("  2. dynamem  (SAM-2 – requires third_party/segment-anything-2)")
    print("  3. dev      (pytest, ruff, mypy)")
    print("  4. all      (sim + dynamem + dev)")
    print("  Q. Cancel")
    try:
        c = input("Choice (1–4, Q) [Q]: ").strip().upper() or "Q"
    except (EOFError, KeyboardInterrupt):
        return
    if c == "Q":
        return
    extras: list[str] = []
    if c == "1":
        extras = ["sim"]
    elif c == "2":
        extras = ["dynamem"]
    elif c == "3":
        extras = ["dev"]
    elif c == "4":
        extras = ["sim", "dynamem", "dev"]
    else:
        print("Invalid choice.")
        return
    cmd = ["uv", "sync"] + [arg for e in extras for arg in ("--extra", e)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.call(cmd, cwd=root)
    if result == 0:
        print("Sync complete.")
    else:
        print("Sync failed (check missing third_party or enable_sim_pyproject).")


def run_install_menu() -> int:
    """Run the text-based install menu. Returns exit code."""
    root = _project_root()
    os.chdir(root)

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

        print("  A. Run all (submodules, sim, kitchen assets, MolmoSpaces – with prompts)")
        print("  S. Sync extras (uv sync: sim, dynamem, dev – prompt for which)")
        print("  Q. Quit")
        print()

        try:
            choice = input("Choice (1–4, A, S, Q) [Q]: ").strip().upper() or "Q"
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

            def _molmo_pip(*args):
                if use_uv:
                    subprocess.call(["uv", "pip", "install", "--python", str(py_molmo)] + list(args), cwd=root)
                else:
                    subprocess.call([str(py_molmo), "-m", "pip", "install"] + list(args), cwd=root)

            if not py_molmo.exists():
                subprocess.call(["uv", "venv", ".venv-molmospaces"], cwd=root)
                _molmo_pip("--upgrade", "pip")
                _molmo_pip("--no-deps", "-e", ".")
                wrapper_dir = root / "packages" / "emet_molmospaces"
                if wrapper_dir.exists():
                    _molmo_pip("-e", str(wrapper_dir))
                else:
                    _molmo_pip("molmo-spaces", "mujoco>=3.4", "numpy>=2.2")
            else:
                print("  .venv-molmospaces already exists.")
                wrapper_dir = root / "packages" / "emet_molmospaces"
                if wrapper_dir.exists() and not (venv_molmo / "bin" / "emet-molmospaces").exists():
                    _molmo_pip("-e", str(wrapper_dir))
            print("\nDone. Run emet sync -e sim (and -e dynamem if needed) to sync extras.")
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
            py_molmo = venv_molmo / "bin" / "python"
            use_uv = shutil.which("uv") if shutil else None

            def _molmo_pip(*args):
                if use_uv:
                    subprocess.call(["uv", "pip", "install", "--python", str(py_molmo)] + list(args), cwd=root)
                else:
                    subprocess.call([str(py_molmo), "-m", "pip", "install"] + list(args), cwd=root)

            if (venv_molmo / "bin" / "emet-molmospaces").exists():
                print(".venv-molmospaces already has emet-molmospaces wrapper.")
            elif py_molmo.exists():
                print(".venv-molmospaces exists but wrapper missing. Install: pip install -e packages/emet_molmospaces")
            else:
                print("\nCreating .venv-molmospaces and installing emet-molmospaces wrapper...")
                subprocess.call(["uv", "venv", ".venv-molmospaces"], cwd=root)
                _molmo_pip("--upgrade", "pip")
                _molmo_pip("--no-deps", "-e", ".")
                wrapper_dir = root / "packages" / "emet_molmospaces"
                if wrapper_dir.exists():
                    _molmo_pip("-e", str(wrapper_dir))
                else:
                    _molmo_pip("molmo-spaces", "mujoco>=3.4", "numpy>=2.2")
                print(
                    "Set MLSPACES_ASSETS_DIR for scene data (e.g. export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets)"
                )

        print()
        input("Press Enter to return to menu...")

    return 0
