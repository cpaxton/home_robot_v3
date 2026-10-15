# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Require scored metrics, not merely a zero process exit or an error JSON."""

import argparse
import json
from pathlib import Path


def validate_metrics(phase: str, path: Path) -> None:
    try:
        text = path.read_text()
    except OSError as error:
        raise ValueError(f"Infrastructure error: cannot read episode metrics {path}: {error}") from error
    try:
        rows = [json.loads(text)]
    except json.JSONDecodeError:
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ValueError(f"Infrastructure error: malformed episode metrics {path}: {error}") from error
    if len(rows) != 1:
        raise ValueError(f"Expected one episode result: {path}")
    row = rows[0]
    if not isinstance(row, dict) or row.get("error"):
        raise ValueError(f"Infrastructure error in {path}: {row}")
    fields = ("correct",) if phase == "eqa" else ("find_object_success", "find_recep_success")
    if any(type(row.get(field)) is not bool for field in fields):
        raise ValueError(f"Missing boolean outcome {fields}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("eqa", "room"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        validate_metrics(args.phase, args.path)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
