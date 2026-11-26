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
import math
import numbers
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
        for key, value in thresholds.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"case {self.id} gate {key} must be finite")
        if "max_duration_s" in thresholds and float(thresholds["max_duration_s"]) <= 0:
            raise ValueError(f"case {self.id} max_duration_s must be positive")

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


def _finite_scalar(
    case_id: str, label: str, value: Any, *, minimum: float | None = None
) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None, f"{case_id}: {label} must be a finite scalar"
    number = float(value)
    if not math.isfinite(number):
        return None, f"{case_id}: {label} must be finite"
    if minimum is not None and number < minimum:
        return None, f"{case_id}: {label} {number:.3f} < {minimum:.3f}"
    return number, None


def _integer_metric(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(number)


def coverage_result(
    *,
    episodes: int,
    success_rate: float,
    collisions: int = 0,
    tracking_error_m: float,
    duration_s: float,
    contact_state_rate: float | None = None,
) -> dict[str, Any]:
    """Build the strict per-case metrics consumed by :func:`gate_report`."""
    episodes = _integer_metric(episodes, label="episodes", minimum=1)
    collisions = _integer_metric(collisions, label="collisions", minimum=0)
    if collisions > episodes:
        raise ValueError("collisions must be in [0, episodes]")
    success = float(success_rate)
    tracking = float(tracking_error_m)
    duration = float(duration_s)
    if not all(math.isfinite(value) for value in (success, tracking, duration)):
        raise ValueError("coverage metrics must be finite")
    if not 0.0 <= success <= 1.0 or tracking < 0.0 or duration < 0.0:
        raise ValueError("coverage metrics are out of range")
    result: dict[str, Any] = {
        "episodes": episodes,
        "success_rate": success,
        "collisions": collisions,
        "tracking_error_m": tracking,
        "duration_s": duration,
    }
    if contact_state_rate is not None:
        contact = float(contact_state_rate)
        if not math.isfinite(contact) or not 0.0 <= contact <= 1.0:
            raise ValueError("contact_state_rate must be finite and in [0, 1]")
        result["contact_state_rate"] = contact
    return result


def task_result_metrics(seed_results: dict[str, dict[str, Any]], task: str) -> dict[str, Any]:
    """Aggregate per-seed task results into one coverage-gate row."""
    rows = []
    for seed, result in seed_results.items():
        if not isinstance(result, dict) or task not in result:
            raise ValueError(f"seed {seed} is missing task result {task!r}")
        row = result[task]
        if not isinstance(row, dict):
            raise ValueError(f"seed {seed} task {task!r} is not a mapping")
        rows.append(row)
    if not rows:
        raise ValueError(f"no seed results for task {task!r}")
    row_episodes = [_integer_metric(row.get("episodes"), label=f"task {task!r} episodes", minimum=1) for row in rows]
    row_collisions = [
        _integer_metric(row.get("collisions", 0), label=f"task {task!r} collisions", minimum=0) for row in rows
    ]
    for seed, count, collisions in zip(seed_results, row_episodes, row_collisions, strict=True):
        if collisions > count:
            raise ValueError(f"seed {seed} has more collisions than episodes for task {task!r}")
    episodes = sum(row_episodes)
    weights = [count / episodes for count in row_episodes]
    contact = [row.get("contact_state_rate") for row in rows]
    contact_rate = (
        sum(weight * float(value) for weight, value in zip(weights, contact, strict=True))
        if all(value is not None for value in contact)
        else None
    )
    return coverage_result(
        episodes=episodes,
        success_rate=sum(weight * float(row["success_rate"]) for weight, row in zip(weights, rows, strict=True)),
        collisions=sum(row_collisions),
        tracking_error_m=sum(
            weight * float(row["tracking_error_m"]) for weight, row in zip(weights, rows, strict=True)
        ),
        duration_s=sum(weight * float(row["duration_s"]) for weight, row in zip(weights, rows, strict=True)),
        contact_state_rate=contact_rate,
    )


def hard_case_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a hard-case evaluator case result into a coverage-gate row."""
    if not isinstance(result, dict):
        raise ValueError("hard-case result must be a mapping")
    episodes_raw = result.get("episodes")
    if isinstance(episodes_raw, list):
        episodes = len(episodes_raw)
    else:
        episodes = _integer_metric(episodes_raw, label="hard-case episodes", minimum=1)
    contact = result.get("contact_state_rate")
    return coverage_result(
        episodes=episodes,
        success_rate=float(result["success_rate"]),
        collisions=_integer_metric(result.get("collisions", 0), label="hard-case collisions", minimum=0),
        tracking_error_m=float(result.get("tracking_error_m", float("inf"))),
        duration_s=float(result.get("duration_s", 0.0)),
        contact_state_rate=None if contact is None else float(contact),
    )


def _case_gate(case: CoverageCase, result: dict[str, Any]) -> list[str]:
    thresholds = case.thresholds()
    failures: list[str] = []
    raw_episodes = result.get("episodes")
    if isinstance(raw_episodes, bool) or not isinstance(raw_episodes, numbers.Real):
        return [f"{case.id}: episodes must be a positive integer"]
    episodes_float = float(raw_episodes)
    if not math.isfinite(episodes_float) or not episodes_float.is_integer() or episodes_float < 1:
        return [f"{case.id}: episodes must be a positive integer"]
    episodes = int(episodes_float)
    success, error = _finite_scalar(case.id, "success_rate", result.get("success_rate"))
    if error:
        failures.append(error)
    elif not 0.0 <= success <= 1.0:
        failures.append(f"{case.id}: success_rate must be in [0, 1]")
    collisions, error = _finite_scalar(case.id, "collisions", result.get("collisions"), minimum=0.0)
    if error:
        failures.append(error)
    elif not float(collisions).is_integer() or int(collisions) > episodes:
        failures.append(f"{case.id}: collisions must be an integer in [0, episodes]")
    tracking, error = _finite_scalar(case.id, "tracking_error_m", result.get("tracking_error_m"), minimum=0.0)
    if error:
        failures.append(error)
    duration, error = _finite_scalar(case.id, "duration_s", result.get("duration_s"), minimum=0.0)
    if error:
        failures.append(error)
    if failures:
        return failures
    if float(success) < thresholds["min_success"]:
        failures.append(f"{case.id}: success {float(success):.2f} < {thresholds['min_success']:.2f}")
    collision_rate = int(collisions) / episodes
    if collision_rate > thresholds["max_collision_rate"]:
        failures.append(f"{case.id}: collision rate {collision_rate:.2f} > {thresholds['max_collision_rate']:.2f}")
    if float(tracking) > thresholds["max_tracking_error_m"]:
        failures.append(f"{case.id}: tracking {float(tracking):.3f} m > {thresholds['max_tracking_error_m']:.3f} m")
    max_duration = thresholds.get("max_duration_s")
    if max_duration is not None and float(duration) > float(max_duration):
        failures.append(f"{case.id}: duration {float(duration):.1f}s > {float(max_duration):.1f}s")
    if case.contact == "required":
        rate = result.get("contact_state_rate")
        contact, error = _finite_scalar(case.id, "contact_state_rate", rate, minimum=0.0)
        if error:
            failures.append(f"{case.id}: contact state missing or invalid from result")
        elif float(contact) < 1.0:
            failures.append(f"{case.id}: contact state rate {float(contact):.2f} < 1.00")
    elif result.get("contact_state_rate") is not None:
        _, error = _finite_scalar(case.id, "contact_state_rate", result.get("contact_state_rate"), minimum=0.0)
        if error:
            failures.append(error)
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
