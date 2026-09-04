# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Versioned, inspectable paper-evaluation matrix loader."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PAPER_MATRIX = "configs/benchmarks/paper_eval.yaml"


def load_paper_matrix(path: str = DEFAULT_PAPER_MATRIX) -> dict[str, Any]:
    """Load a matrix whose paper rows and datasets are explicitly declared."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("rows"), dict)
        or not isinstance(data.get("datasets"), dict)
    ):
        raise ValueError(f"invalid paper matrix: {path}")
    return data


def resolve_paper_row(dataset: str, row: str, *, path: str = DEFAULT_PAPER_MATRIX) -> dict[str, Any]:
    """Resolve one declared row with copied global policy and dataset settings."""
    data = load_paper_matrix(path)
    dataset_cfg = data["datasets"].get(dataset)
    row_cfg = data["rows"].get(row)
    if not isinstance(dataset_cfg, dict):
        raise KeyError(f"unknown paper dataset {dataset!r}")
    if not isinstance(row_cfg, dict) or row not in dataset_cfg.get("rows", []):
        raise KeyError(f"row {row!r} is not declared for dataset {dataset!r}")
    result = deepcopy(row_cfg)
    result.update(
        dataset=dataset,
        row=row,
        policy=deepcopy(data.get("policy") or {}),
        dataset_config=deepcopy(dataset_cfg),
    )
    return result
