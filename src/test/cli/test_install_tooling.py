# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import emet.dev_system_packages as system_packages
import emet.install_ui as install_ui


def test_paper_toolchain_declares_all_required_apt_packages():
    packages = system_packages.list_system_packages()

    assert packages["latexmk"].split() == [
        "latexmk",
        "texlive-latex-extra",
        "texlive-bibtex-extra",
    ]


def test_ensure_apt_package_installs_multi_package_toolchain(monkeypatch):
    commands: list[list[str]] = []
    latexmk_checks = iter([None, "/usr/bin/latexmk"])

    def fake_which(name: str) -> str | None:
        if name == "latexmk":
            return next(latexmk_checks)
        if name == "apt-get":
            return "/usr/bin/apt-get"
        return None

    monkeypatch.setattr(system_packages.shutil, "which", fake_which)
    monkeypatch.setattr(system_packages.subprocess, "call", lambda cmd: commands.append(cmd) or 0)

    assert system_packages.ensure_apt_package("latexmk", non_interactive=True) == 0
    assert commands == [
        ["sudo", "apt-get", "update"],
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "latexmk",
            "texlive-latex-extra",
            "texlive-bibtex-extra",
        ],
    ]


def test_install_menu_exposes_and_executes_paper_toolchain(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(install_ui.shutil, "which", lambda _: None)
    paper_status = install_ui._check_paper(tmp_path)
    assert paper_status.id == "paper"
    assert paper_status.status == "missing"

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(install_ui, "_has_uv", lambda: False)

    def fake_ensure_apt_package(tool: str, *, non_interactive: bool = False) -> int:
        calls.append((tool, non_interactive))
        return 0

    monkeypatch.setattr(install_ui, "ensure_apt_package", fake_ensure_apt_package)
    plan = install_ui.WizardPlan(
        submodules=False,
        uv_dev=False,
        include_dynamem=False,
        paper=True,
    )

    assert install_ui._execute_wizard_plan(tmp_path, plan) == 0
    assert calls == [("latexmk", True)]


def test_run_install_simulation_non_interactive_passes_yes(monkeypatch, tmp_path: Path):
    from emet.cli_cmds import run_sync

    calls: list[list[str]] = []
    monkeypatch.setattr(run_sync.subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "install_simulation.sh").write_text("#!/bin/bash\n")

    assert run_sync._run_install_simulation(tmp_path, non_interactive=True) == 0
    assert calls == [["bash", str(tmp_path / "scripts" / "install_simulation.sh"), "-y"]]

    calls.clear()
    assert run_sync._run_install_simulation(tmp_path) == 0
    assert "-y" not in calls[0]

def test_lightwheel_pack_requires_the_mesh_not_just_the_registry(tmp_path: Path):
    import importlib.util

    script = Path(__file__).resolve().parents[3] / "scripts" / "download_robocasa_assets.py"
    spec = importlib.util.spec_from_file_location("download_robocasa_assets", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = tmp_path / "models" / "assets"
    fixtures = base / "fixtures"
    (fixtures / "fixture_registry").mkdir(parents=True)
    (fixtures / "fixture_registry" / "sink.yaml").write_text("Sink025:\n  alias: Sink001\n")

    # Registry entry alone is not the pack: the server still needs the Sink025 mesh.
    assert module._asset_present(base, "fixtures", "lightwheel_registry") is False

    (fixtures / "sinks" / "Sink025").mkdir(parents=True)
    (fixtures / "sinks" / "Sink025" / "model.xml").write_text("<mujoco/>")
    assert module._asset_present(base, "fixtures", "lightwheel_registry") is True

def test_downloader_base_fixtures_require_all_sentinels(tmp_path: Path):
    import importlib.util

    script = Path(__file__).resolve().parents[3] / "scripts" / "download_robocasa_assets.py"
    spec = importlib.util.spec_from_file_location("download_robocasa_assets", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fixtures = tmp_path / "fixtures"
    counter = fixtures / "counters/counter/model.xml"
    counter.parent.mkdir(parents=True)
    counter.write_text("<mujoco/>")
    assert module._basic_fixtures_present(fixtures) is False

    for rel in ("sinks/2_bins_stainless/model.xml", "sinks/white_sink/model.xml"):
        path = fixtures / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<mujoco/>")
    assert module._basic_fixtures_present(fixtures) is True
