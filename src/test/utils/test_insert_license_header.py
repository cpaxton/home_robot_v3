# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for the pre-commit license-header hook (scripts/insert_license_header.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LICENSE_FILE = REPO_ROOT / "docs" / "license_header_chris_paxton.txt"

_spec = importlib.util.spec_from_file_location(
    "insert_license_header", REPO_ROOT / "scripts" / "insert_license_header.py"
)
assert _spec is not None and _spec.loader is not None
insert_license_header = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(insert_license_header)

HELLO_ROBOT_HEADER = "# Copyright (c) Hello Robot, Inc.\n# All rights reserved.\n\nimport os\n"
SPDX_HEADER = "# SPDX-License-Identifier: Apache-2.0\n# Vendored upstream.\n\nimport os\n"
ATTRIBUTION_HEADER = "# Copyright 2024 Chris Paxton\n# From https://example.invalid/x.py\n\nimport os\n"


@pytest.fixture
def header() -> list[str]:
    return insert_license_header.build_header(LICENSE_FILE)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "name,text",
    [
        ("legacy.py", HELLO_ROBOT_HEADER),
        ("vendored.py", SPDX_HEADER),
        ("attributed.py", ATTRIBUTION_HEADER),
        ("empty.py", ""),
    ],
)
def test_existing_license_is_left_untouched(tmp_path, header, name, text):
    """Never stamp a second header onto a file that already declares provenance."""
    path = _write(tmp_path, name, text)
    assert insert_license_header.insert_header(path, header) is False
    assert path.read_text(encoding="utf-8") == text


def test_new_file_gets_chris_paxton_header(tmp_path, header):
    path = _write(tmp_path, "new_module.py", '"""New module."""\n\nimport os\n')
    assert insert_license_header.insert_header(path, header) is True
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Copyright (c) Chris Paxton 2026\n")
    assert "Hello Robot" not in text
    assert text.endswith('"""New module."""\n\nimport os\n')


def test_header_goes_below_shebang_and_encoding(tmp_path, header):
    path = _write(tmp_path, "script.py", "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nimport os\n")
    assert insert_license_header.insert_header(path, header) is True
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "#!/usr/bin/env python3"
    assert lines[1] == "# -*- coding: utf-8 -*-"
    assert lines[2] == "# Copyright (c) Chris Paxton 2026"


def test_insert_is_idempotent(tmp_path, header):
    path = _write(tmp_path, "new_module.py", "import os\n")
    assert insert_license_header.insert_header(path, header) is True
    after_first = path.read_text(encoding="utf-8")
    assert insert_license_header.insert_header(path, header) is False
    assert path.read_text(encoding="utf-8") == after_first


def test_main_reports_modified_files(tmp_path, capsys):
    untouched = _write(tmp_path, "legacy.py", HELLO_ROBOT_HEADER)
    modified = _write(tmp_path, "new_module.py", "import os\n")
    code = insert_license_header.main(["--license-file", str(LICENSE_FILE), str(untouched), str(modified)])
    assert code == 1  # non-zero so pre-commit fails the commit and re-stages
    out = capsys.readouterr().out
    assert "new_module.py" in out
    assert "legacy.py" not in out


def test_repo_python_files_have_no_duplicate_headers():
    """Guard the bug this hook replaced: Hello Robot stamped on top of a Chris Paxton file."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    offenders = []
    for rel in tracked:
        if rel.startswith("third_party/"):
            continue
        head = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore").split("\n")[:20]
        joined = "\n".join(head)
        # Match the header line itself; prose mentioning Hello Robot is not a stamp.
        if "Copyright (c) Hello Robot" in joined and "Copyright (c) Chris Paxton" in joined:
            offenders.append(rel)
    assert offenders == []
