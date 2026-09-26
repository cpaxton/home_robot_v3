"""Acceptance scorer never promotes incomplete or fabricated success."""

import json

import numpy as np
import pytest

from emet.eval.physical_tamp import score_physical_acceptance


def rows():
    result = []
    rotation = np.eye(3).ravel().tolist()
    for i in range(65):
        t = i * 0.1
        carried = 1.5 <= t < 4.5
        placed = t >= 4.5
        row = {
            "sim_time": t,
            "object_pos": [0, 0, 0.2 if carried else (0.1 if placed else 0)],
            "object_rot": rotation,
            "relative_pos": [0, 0, 0.03],
            "relative_rot": rotation,
            "gripper_contact": carried,
            "support_contact": placed,
            "other_contact": not (carried or placed),
            "scored_execution": True,
            "robot_collision_audited": True,
            "forbidden_robot_contacts": [],
        }
        result.append(row)
    return result


def score(tmp_path, records, *, complete=True, forbidden=False):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in [{"schema": 1, "config": {}}, *records]) + "\n")
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"accepted": not forbidden}) + "\n")
    return score_physical_acceptance(trace, audit, execution_completed=complete)


def test_physical_trace_requires_complete_retained_supported_task(tmp_path):
    result = score(tmp_path, rows())
    assert result["task_success"], result


@pytest.mark.parametrize(
    "fault", ["drop", "wrong_support", "collision", "missing_audit", "truncated", "incomplete", "teleport"]
)
def test_physical_negatives(tmp_path, fault):
    records = rows()
    if fault == "drop":
        records[35]["gripper_contact"] = False
    elif fault == "wrong_support":
        for row in records[45:]:
            row["support_contact"] = False
            row["other_contact"] = True
    elif fault == "collision":
        records[-1]["forbidden_robot_contacts"] = [{"bodies": ["base", "wall"]}]
    elif fault == "missing_audit":
        records[-1].pop("robot_collision_audited")
    elif fault == "truncated":
        records = records[:49]
    result = score(tmp_path, records, complete=fault != "incomplete", forbidden=fault == "teleport")
    assert not result["task_success"], (fault, result)
