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
