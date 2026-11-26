# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Case registry and promotion gates for WBC coverage.

The registry defines what the controller must handle; the gate consumes a
per-case evaluation report and refuses promotion unless every active family
passes its held-out cases. Planned families are reported explicitly instead of
being counted as passes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAMILIES = (
    "free_space_reach",
    "hard_posture",
    "shelf_entry",
    "carry_payload",
    "contact_recovery",
    "base_arm",
    "articulation",
)
SPLITS = ("train", "validation", "test")
STATUSES = ("active", "planned")
OCCUPANCY = ("none", "optional", "required")
CONTACT = ("none", "optional", "required")

_DEFAULT_GATES: dict[str, dict[str, float]] = {
    "free_space_reach": {"min_success": 0.90, "max_collision_rate": 0.0, "max_tracking_error_m": 0.03},
    "hard_posture": {"min_success": 0.80, "max_collision_rate": 0.0, "max_tracking_error_m": 0.05},
    "base_arm": {"min_success": 0.85, "max_collision_rate": 0.0, "max_tracking_error_m": 0.04},
    "shelf_entry": {"min_success": 0.75, "max_collision_rate": 0.0, "max_tracking_error_m": 0.05},
    "carry_payload": {"min_success": 0.75, "max_collision_rate": 0.0, "max_tracking_error_m": 0.05},
    "contact_recovery": {"min_success": 0.70, "max_collision_rate": 0.0, "max_tracking_error_m": 0.06},
    "articulation": {"min_success": 0.70, "max_collision_rate": 0.0, "max_tracking_error_m": 0.06},
}


@dataclass(frozen=True)
class CoverageCase:
    id: str
    family: str
    robot: str
    split: str
    status: str
    source: str
    occupancy: str = "none"
    contact: str = "none"
    gate: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.robot or not self.source:
            raise ValueError("coverage case id/robot/source must be non-empty")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown case family: {self.family!r}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown case split: {self.split!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown case status: {self.status!r}")
        if self.occupancy not in OCCUPANCY:
            raise ValueError(f"unknown occupancy requirement: {self.occupancy!r}")
        if self.contact not in CONTACT:
            raise ValueError(f"unknown contact requirement: {self.contact!r}")
        thresholds = self.thresholds()
        for key in ("min_success", "max_collision_rate", "max_tracking_error_m"):
            if key not in thresholds:
                raise ValueError(f"case {self.id} has no {key} gate")
        if not 0.0 <= thresholds["min_success"] <= 1.0:
            raise ValueError(f"case {self.id} min_success must be in [0, 1]")
        if not 0.0 <= thresholds["max_collision_rate"] <= 1.0:
            raise ValueError(f"case {self.id} max_collision_rate must be in [0, 1]")
        if thresholds["max_tracking_error_m"] < 0:
            raise ValueError(f"case {self.id} max_tracking_error_m must be nonnegative")

    def thresholds(self) -> dict[str, float]:
        merged = dict(_DEFAULT_GATES[self.family])
        merged.update({key: float(value) for key, value in self.gate.items()})
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "robot": self.robot,
            "split": self.split,
            "status": self.status,
            "source": self.source,
            "occupancy": self.occupancy,
            "contact": self.contact,
            "gate": dict(self.gate),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CoverageCase:
        return cls(
            id=str(raw["id"]),
            family=str(raw["family"]),
            robot=str(raw["robot"]),
            split=str(raw["split"]),
            status=str(raw["status"]),
            source=str(raw["source"]),
            occupancy=str(raw.get("occupancy") or "none"),
            contact=str(raw.get("contact") or "none"),
            gate={str(key): float(value) for key, value in (raw.get("gate") or {}).items()},
            notes=str(raw.get("notes") or ""),
        )


def load_coverage(path: str | Path) -> list[CoverageCase]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases") or []
    if not isinstance(raw, list):
        raise ValueError("coverage registry must be a list or {'cases': [...]}")
    cases = [CoverageCase.from_dict(item) for item in raw]
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("coverage registry has duplicate case ids")
    for case in cases:
        if case.status == "active" and case.family not in {c.family for c in cases if c.split == "test"}:
            raise ValueError(f"active family {case.family!r} has no test case")
    return cases


def _case_gate(case: CoverageCase, result: dict[str, Any]) -> list[str]:
    thresholds = case.thresholds()
    failures: list[str] = []
    episodes = int(result.get("episodes") or 0)
    if episodes < 1:
        return [f"{case.id}: no evaluation episodes"]
    success = float(result.get("success_rate", 0.0))
    collisions = int(result.get("collisions") or 0)
    error = float(result.get("tracking_error_m", float("inf")))
    duration = float(result.get("duration_s", 0.0) or 0.0)
    if success < thresholds["min_success"]:
        failures.append(f"{case.id}: success {success:.2f} < {thresholds['min_success']:.2f}")
    collision_rate = collisions / episodes
    if collision_rate > thresholds["max_collision_rate"]:
        failures.append(f"{case.id}: collision rate {collision_rate:.2f} > {thresholds['max_collision_rate']:.2f}")
    if error > thresholds["max_tracking_error_m"]:
        failures.append(f"{case.id}: tracking {error:.3f} m > {thresholds['max_tracking_error_m']:.3f} m")
    max_duration = thresholds.get("max_duration_s")
    if max_duration is not None and duration > float(max_duration):
        failures.append(f"{case.id}: duration {duration:.1f}s > {float(max_duration):.1f}s")
    if case.contact == "required":
        rate = result.get("contact_state_rate")
        if rate is None:
            failures.append(f"{case.id}: contact state missing from result")
        elif float(rate) < 1.0:
            failures.append(f"{case.id}: contact state rate {float(rate):.2f} < 1.00")
    return failures


def gate_report(
    cases: Iterable[CoverageCase],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate every active held-out case and summarize family coverage."""
    cases = list(cases)
    families: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    missing: list[str] = []
    for family in FAMILIES:
        family_cases = [case for case in cases if case.family == family]
        if not family_cases:
            continue
        active = [case for case in family_cases if case.status == "active"]
        test_cases = [case for case in active if case.split == "test"]
        family_failures: list[str] = []
        if active and not test_cases:
            family_failures.append(f"{family}: active family has no test case")
        for case in test_cases:
            result = results.get(case.id)
            if result is None:
                missing.append(case.id)
                family_failures.append(f"{case.id}: missing result")
                continue
            family_failures.extend(_case_gate(case, result))
        passed = bool(active and test_cases and not family_failures)
        if active:
            failures.extend(family_failures)
        families[family] = {
            "status": "gated" if active else "not_gated",
            "test_cases": [case.id for case in test_cases],
            "cases": [case.id for case in family_cases],
            "passed": passed,
            "failures": family_failures,
        }
    return {
        "schema_version": 1,
        "passed": not failures and not missing and any(item["status"] == "gated" for item in families.values()),
        "families": families,
        "missing_results": missing,
        "failures": failures,
    }


def _load_results(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("results report must be a mapping of case id -> metrics")
    return {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WBC coverage gate")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    cases = load_coverage(args.cases)
    results = _load_results(args.results) if args.results else {}
    report = gate_report(cases, results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CoverageCase",
    "FAMILIES",
    "SPLITS",
    "STATUSES",
    "OCCUPANCY",
    "gate_report",
    "load_coverage",
    "main",
]
