# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from pathlib import Path
from unittest.mock import Mock

import pytest

from emet.simulation import molmospaces_config as config


@pytest.mark.parametrize("has_wrapper", [True, False])
def test_interpreter_override_preserves_virtualenv_symlink(tmp_path, monkeypatch, has_wrapper):
    base = tmp_path / "base-python"
    base.touch()
    python = tmp_path / "environment" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base)
    wrapper = python.with_name("emet-molmospaces")
    if has_wrapper:
        wrapper.touch()
    monkeypatch.setattr(config, "_project_root", lambda: tmp_path / "checkout")
    monkeypatch.setenv("MOLMOSPACES_PYTHON", str(python))
    monkeypatch.setattr("shutil.which", lambda _: None)
    probe = Mock(side_effect=lambda path: Path(path) == python)
    monkeypatch.setattr(config, "_python_can_import_molmo_spaces", probe)
    expected = [str(wrapper)] if has_wrapper else [str(python), "-m", "emet_molmospaces"]
    assert config.build_molmospaces_wrapper_command(["merge-scene"]) == expected + ["merge-scene"]
    probe.assert_called_once_with(python)


def test_invalid_override_still_requires_successful_import_probe(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setattr(config, "_project_root", lambda: tmp_path / "checkout")
    monkeypatch.setenv("MOLMOSPACES_PYTHON", str(python))
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(config, "_python_can_import_molmo_spaces", lambda _: False)
    assert config.build_molmospaces_wrapper_command([]) is None
