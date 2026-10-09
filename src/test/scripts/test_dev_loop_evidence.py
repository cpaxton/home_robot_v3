# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""The dev loop must not overwrite another arm's diagnostic bundles."""

import os
import subprocess
from pathlib import Path


def test_eqa_diagnostics_are_unique_across_questions_and_invocations(tmp_path):
    root = Path(__file__).resolve().parents[3]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    habitat = bin_dir / "habitat"
    habitat.write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        "while (( $# )); do\n"
        '  case "$1" in\n'
        '    --debug-run-tag) printf "%s\\n" "$2" >> "$CAPTURE_TAGS"; shift;;\n'
        '    --output) printf "{}\\n" > "$2"; shift;;\n'
        "  esac\n  shift\ndone\n"
    )
    habitat.chmod(0o755)
    for name in ("python", "sleep"):
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    capture = tmp_path / "tags.txt"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        HABITAT_BIN=str(habitat),
        EMET_PY=str(bin_dir / "python"),
        CAPTURE_TAGS=str(capture),
        OUT_DIR=str(tmp_path / "out"),
        PHASE="eqa",
        DEV_EQA_IDS="12 14",
        DEV_SEED="0",
    )
    for _ in range(2):
        subprocess.run(["bash", str(root / "scripts/run_dev_loop.sh")], env=env, check=True)
    tags = capture.read_text().splitlines()
    assert len(tags) == len(set(tags)) == 4
    assert [tag.rsplit("_", 1)[1] for tag in tags] == ["q12", "q14", "q12", "q14"]
    assert all("_s0_" in tag for tag in tags)
    assert len((tmp_path / "out/diagnostic_tags.txt").read_text().splitlines()) == 2
