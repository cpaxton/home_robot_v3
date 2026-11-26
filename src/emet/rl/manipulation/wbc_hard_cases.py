# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Hard cases for whole-body-control training, sourced from manipulation failures.

A hard case is a recorded arm posture and target posture: the actor starts at the
posture it actually reached when a manipulation attempt failed, and the teacher
is asked to reach the posture that the failed attempt was trying to command.
This is the right injection point for shelf/carry/grasp failures: it changes the
DAgger state distribution, not the reward or the policy schema.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from emet.controller.manipulation.wbc_robots import get_wbc_robot


@dataclass(frozen=True)
class WbcHardCase:
    """One recorded failure posture and its intended target posture."""

    name: str
    robot: str
    arm: str
    initial_q: tuple[float, ...]
    target_q: tuple[float, ...]
    failure_mode: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    scene: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.robot or not self.arm:
            raise ValueError("hard case name/robot/arm must be non-empty")
        if not isinstance(self.scene, dict):
            raise ValueError("hard case scene must be a mapping")
        if not self.initial_q or len(self.initial_q) != len(self.target_q):
            raise ValueError("hard case initial_q and target_q must have equal non-zero length")
        if not all(isinstance(value, (int, float)) for value in (*self.initial_q, *self.target_q)):
            raise ValueError("hard case joint postures must be numeric")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "robot": self.robot,
            "arm": self.arm,
            "initial_q": [float(value) for value in self.initial_q],
            "target_q": [float(value) for value in self.target_q],
            "failure_mode": self.failure_mode,
            "source": self.source,
            "scene": dict(self.scene),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WbcHardCase:
        return cls(
            name=str(raw["name"]),
            robot=str(raw["robot"]),
            arm=str(raw["arm"]),
            initial_q=tuple(float(value) for value in raw["initial_q"]),
            target_q=tuple(float(value) for value in raw["target_q"]),
            failure_mode=str(raw.get("failure_mode") or ""),
            source=dict(raw.get("source") or {}),
            scene=dict(raw.get("scene") or {}),
        )


def _joint_value_map(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    if len(names) != len(values):
        raise ValueError("joint names and values must have equal length")
    return {str(name): float(value) for name, value in zip(names, values, strict=True)}


def _ordered_arm_q(
    names: Sequence[str], values: Sequence[float], joint_names: Sequence[str]
) -> tuple[float, ...] | None:
    values_by_name = _joint_value_map(names, values)
    ordered: list[float] = []
    for joint in joint_names:
        value = values_by_name.get(str(joint))
        if value is None:
            # RoboCasa records use robot0_jointN; the contract uses jointN.
            suffix = str(joint).split("_")[-1]
            value = next((v for name, v in values_by_name.items() if name.endswith(suffix)), None)
        if value is None:
            return None
        ordered.append(float(value))
    return tuple(ordered)


def _canonical_robot(name: str) -> str:
    """Normalize contract/registry robot aliases (panda_omron, rb_y1, mars)."""
    if not name:
        return ""
    try:
        return get_wbc_robot(str(name)).name
    except KeyError:
        return str(name).lower().replace("-", "_").replace(" ", "_")


def hard_cases_from_failure_records(
    records: Iterable[dict[str, Any]],
    *,
    robot: str,
    arm: str = "right",
    joint_names: Sequence[str] | None = None,
    max_cases: int = 64,
) -> list[WbcHardCase]:
    """Extract unique arm postures from manipulation-failure JSONL records.

    ``joint_names`` is the contract arm order. It defaults to ``joint1..joint7``
    for the Panda profile; callers for other robots should pass their profile's
    arm joint names explicitly.
    """
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    names = tuple(str(name) for name in (joint_names or tuple(f"joint{i}" for i in range(1, 8))))
    cases: list[WbcHardCase] = []
    seen: set[tuple[float, ...]] = set()
    for index, record in enumerate(records):
        attempt = record.get("attempt") or {}
        context = record.get("context") or {}
        record_robot = _canonical_robot(str(context.get("robot_id") or ""))
        requested_robot = _canonical_robot(robot)
        if requested_robot and record_robot and record_robot != requested_robot:
            continue
        # Only a failed reach is a hard case. Place failures can carry the
        # preceding successful grasp as ``last_attempt`` (reached=True); those
        # postures are not failure states and must not enter the set.
        if attempt.get("reached") is not False:
            continue
        record_joints = attempt.get("joint_names") or []
        trace = attempt.get("trace") or []
        waypoints = attempt.get("planned_waypoints") or []
        if not record_joints or not trace or not waypoints:
            continue
        initial = attempt.get("initial_q")
        if not initial and isinstance(trace[0], dict):
            initial = trace[0].get("qpos")
        target = waypoints[-1]
        if not isinstance(initial, list) or not isinstance(target, list):
            continue
        initial_q = _ordered_arm_q(record_joints, initial, names)
        target_q = _ordered_arm_q(record_joints, target, names)
        if initial_q is None or target_q is None or initial_q == target_q:
            continue
        key = initial_q
        if key in seen:
            continue
        seen.add(key)
        case = WbcHardCase(
            name=f"failure-{index:03d}-{context.get('object_query') or 'object'}-"
            f"{context.get('receptacle_query') or 'receptacle'}",
            robot=record_robot or robot,
            arm=str(arm),
            initial_q=initial_q,
            target_q=target_q,
            failure_mode=str(attempt.get("failure_mode") or context.get("message") or ""),
            source={
                "record_index": index,
                "job_id": context.get("job_id"),
                "object_query": context.get("object_query"),
                "receptacle_query": context.get("receptacle_query"),
                "failure_mode": attempt.get("failure_mode"),
                "message": context.get("message"),
                "op": attempt.get("op"),
                "reached": attempt.get("reached"),
                "error_m": attempt.get("error_m"),
            },
            scene={
                key: context[key]
                for key in (
                    "robot_id",
                    "op",
                    "object_query",
                    "receptacle_query",
                    "object_body",
                    "receptacle_body",
                    "layout",
                    "style",
                    "scene",
                    "job_id",
                )
                if context.get(key) is not None
            },
        )
        cases.append(case)
        if len(cases) >= max_cases:
            break
    return cases


def load_hard_cases(path: str | Path) -> list[WbcHardCase]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases") or []
    if not isinstance(raw, list):
        raise ValueError("hard-case file must be a list or {'cases': [...]}")
    return [WbcHardCase.from_dict(item) for item in raw]


def hard_case_split(path: str | Path) -> str:
    """Return the declared train/test split for a hard-case file."""
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("split") or "")


def save_hard_cases(
    path: str | Path,
    cases: Sequence[WbcHardCase],
    *,
    split: str | None = None,
) -> Path:
    if split is not None and split not in {"train", "test", "validation"}:
        raise ValueError(f"unknown hard-case split: {split!r}")
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "cases": [case.to_dict() for case in cases]}
    if split is not None:
        payload["split"] = split
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _load_failure_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export WBC hard cases from manipulation failures")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--arm", default="right")
    parser.add_argument("--joints", default="", help="comma-separated contract arm joints; default joint1..7")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--split", choices=("train", "test", "validation"), default="train")
    args = parser.parse_args(argv)
    joint_names = [name.strip() for name in args.joints.split(",") if name.strip()] or None
    cases = hard_cases_from_failure_records(
        _load_failure_records(args.records),
        robot=args.robot,
        arm=args.arm,
        joint_names=joint_names,
        max_cases=args.max_cases,
    )
    out = save_hard_cases(args.out, cases, split=args.split)
    print(f"wrote {len(cases)} hard cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WbcHardCase",
    "hard_case_split",
    "hard_cases_from_failure_records",
    "load_hard_cases",
    "save_hard_cases",
]
