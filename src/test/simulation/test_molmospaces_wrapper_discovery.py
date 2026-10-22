# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.simulation import molmospaces_config as config


def test_explicit_venv_preserves_symlink_interpreter(monkeypatch, tmp_path):
    binary = tmp_path / "shared-python"
    binary.touch()
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    python = bindir / "python"
    python.symlink_to(binary)
    wrapper = bindir / "emet-molmospaces"
    wrapper.touch()
    checked = []

    def can_import(path):
        checked.append(path)
        return path == python

    monkeypatch.setattr(config, "_project_root", lambda: tmp_path / "checkout")
    monkeypatch.setattr(config, "_python_can_import_molmo_spaces", can_import)
    monkeypatch.setenv("MOLMOSPACES_PYTHON", str(python))
    assert config.build_molmospaces_wrapper_command(["--help"]) == [str(wrapper), "--help"]
    assert checked == [python]

    wrapper.unlink()
    assert config.build_molmospaces_wrapper_command([]) == [str(python), "-m", "emet_molmospaces"]
