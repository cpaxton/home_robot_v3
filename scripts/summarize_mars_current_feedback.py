# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Small source-bound tables for reset/stop and frozen corrected-plant baselines."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def terminal_sample(info):
    """Single measured sample, explicitly not a stop-confirmation handshake."""
    import math

    names = ("max_joint_velocity", "base_linear_norm_m_s", "base_angular_norm_rad_s")
    values = [info[name] for name in names]
    return not info.get("invalid_physics", False) and all(
        math.isfinite(v) and 0 <= v < limit for v, limit in zip(values, (0.03, 0.005, 0.02), strict=True)
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compact_primitive(row):
    result = {key: value for key, value in row.items() if key != "commands"}
    commands = row.get("commands", [])
    result["command_count"] = len(commands)
    result["command_phases"] = [command.get("phase") for command in commands]
    result["commands_sha256"] = hashlib.sha256(
        json.dumps(commands, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def main():
    from emet.rl.manipulation.mastery import read_report
    from emet.rl.manipulation.mastery_objective import smoothness_budgets
    from emet.rl.manipulation.whole_body_training import load_development
    from emet.rl.regression_gate import verify_regressions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/eval/mars-current-feedback-validation-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loop, baseline = (load(args.root / name / "summary.json") for name in ("loop", "baseline"))
    if (
        not loop["complete"]
        or not baseline["complete"]
        or loop["producer_revision"] != baseline["producer_revision"]
        or loop["source_hashes"] != baseline["source_hashes"]
    ):
        raise ValueError("complete same-source loop and baseline reports required")
    receipt_path = args.root / "full.receipt.json"
    receipt = verify_regressions(receipt_path)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "format": "mars-current-feedback-reset-baselines-v2",
        "producer_revision": baseline["producer_revision"],
        "loop_summary_sha256": sha(args.root / "loop/summary.json"),
        "baseline_summary_sha256": sha(args.root / "baseline/summary.json"),
        "readout_script_sha256": sha(Path(__file__)),
        "new_ppo_steps": 0,
        "hardware_used": False,
        "qualified": False,
        "defaults_changed": False,
        "source_hashes_digest": hashlib.sha256(
            json.dumps(baseline["source_hashes"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "regressions": {
            "tests": len(receipt["test_ids"]),
            "returncode": receipt["returncode"],
            "elapsed_wall_s": receipt["finished_unix"] - receipt["started_unix"],
            "receipt_sha256": sha(receipt_path),
            "xml_sha256": receipt["xml_sha256"],
        },
        "loop": {},
        "baselines": {},
        "teacher": {**baseline["teacher"], "admission_is_separate_from_frozen_budget_anchor": True},
        "elapsed_wall_s": {"loop": loop["elapsed_wall_seconds"], "baseline": baseline["elapsed_wall_seconds"]},
    }
    previous = args.root.parent / "mars-current-feedback-validation-v1"
    note = previous / "interrupted-attempt.json"
    if note.exists():
        result["interrupted_attempt"] = load(note) | {"record_sha256": sha(note)}
        before = load(previous / "loop/summary.json")
        result["loop_exactly_replays_first_source"] = all(
            before["variants"][name]["cases"] == variant["cases"] for name, variant in loop["variants"].items()
        )
    for name, variant in loop["variants"].items():
        rows = variant["cases"]
        result["loop"][name] = {
            "cases": len(rows),
            "stops_confirmed": sum(r["stop"]["confirmed"] for r in rows),
            "max_stop_sim_s": max(r["stop"]["elapsed_sim_s"] for r in rows),
            "safety_failures": [
                r["id"]
                for r in rows
                if any(r["safety"][key] for key in ("wheel_lift", "unsafe_contact", "invalid_physics"))
            ],
            "cases_detail": [
                {key: value for key, value in row.items() if key != "admission"}
                | {
                    "admission_identity": {
                        key: row["admission"][key] for key in ("initial_state_sha256", "reference_sha256")
                    }
                }
                for row in rows
            ],
            "archive_sha256": variant["archive_sha256"],
        }
    result["teacher"]["tamp"] = [compact_primitive(row) for row in baseline["teacher"]["tamp"]]
    teacher = read_report(args.root / "baseline/general-teacher.json.gz")
    if len(teacher["cases"]) != 164:
        raise ValueError("expected the declared164 general teacher cases")
    result["teacher"]["failures"] = [
        {"id": r["id"], "code": r["code"], "reasons": r["failure_reasons"]}
        for r in teacher["cases"]
        if not r["mastered"]
    ]
    metric_keys = (
        "hand_position_error_m",
        "hand_orientation_error_rad",
        "base_error_m",
        "base_heading_error_rad",
        "hold_drift_m",
        "settled_seconds",
        "base_linear_jerk",
        "base_yaw_jerk",
        "joint_jerk",
        "peak_force_N",
    )
    columns = [
        "controller",
        "case_id",
        "code",
        "full",
        "tracking_and_settling",
        "smoothness",
        "terminal_velocity_sample_pass",
        "failure_reasons",
        *metric_keys,
    ]
    table = []
    populations = {"teacher": teacher}
    for name, record in baseline["baselines"].items():
        report = read_report(args.root / "baseline" / (name + ".json.gz"))
        expected = 164 if name == "retained_legacy" else 12
        if len(report["cases"]) != expected:
            raise ValueError("incomplete frozen baseline: " + name)
        populations[name] = report
        by_stage = {}
        for stage in {r["stage"] for r in report["cases"]}:
            cases = [r for r in report["cases"] if r["stage"] == stage]
            by_stage[stage] = {
                "cases": len(cases),
                "full": sum(c["mastered"] for c in cases),
                "tracking": sum(c["tracking_and_settling_pass"] for c in cases),
                "smoothness": sum(c["smoothness_pass"] for c in cases),
            }
        result["baselines"][name] = {
            key: value for key, value in record.items() if key not in {"outcomes", "archived_budget_comparison"}
        } | {
            "stage_counts": by_stage,
            "archived_budget_full": sum(c["comparison_pass"] for c in record["archived_budget_comparison"]),
            "archived_budget_rows": record["archived_budget_comparison"],
            "failure_counts": dict(
                collections.Counter(reason for row in report["cases"] for reason in row["failure_reasons"])
            ),
        }
        if "tamp" in record:
            result["baselines"][name]["tamp"] = [compact_primitive(row) for row in record["tamp"]]
    for name, report in populations.items():
        for row in report["cases"]:
            table.append(
                [
                    name,
                    row["id"],
                    row["code"],
                    row["mastered"],
                    row["tracking_and_settling_pass"],
                    row["smoothness_pass"],
                    terminal_sample(row["final_info"]),
                    row["failure_reasons"],
                    *[row["metrics"][key] for key in metric_keys],
                ]
            )
    common_ids = {r["id"] for r in populations["gru_candidate"]["cases"]}
    result["matched_twelve_case_comparison"] = {}
    for name, report in populations.items():
        cases = [r for r in report["cases"] if r["id"] in common_ids]
        if len(cases) != 12:
            raise ValueError("all controllers must cover the same twelve cases")
        result["matched_twelve_case_comparison"][name] = {
            "cases": 12,
            "full": sum(r["mastered"] for r in cases),
            "tracking": sum(r["tracking_and_settling_pass"] for r in cases),
            "arm_full": sum(r["mastered"] for r in cases if r["stage"] == "arm"),
            "arm_smoothness": sum(r["smoothness_pass"] for r in cases if r["stage"] == "arm"),
            "failures": [{"id": r["id"], "reasons": r["failure_reasons"]} for r in cases if not r["mastered"]],
        }
    anchor_path = Path("logs/manipulation/mars-reference-base-exploration-v1/preflight/general-teacher.json.gz")
    anchor = {r["id"]: r for r in read_report(anchor_path)["cases"]}
    _, general = load_development()
    budget_rows = []
    for row in teacher["cases"]:
        old, current = (smoothness_budgets(general, reference["metrics"]) for reference in (anchor[row["id"]], row))
        budget_rows.append({"id": row["id"], "archived": old, "current": current})
    write_json(
        args.output / "budget-matrix.json",
        {
            "archived_anchor_sha256": sha(anchor_path),
            "current_teacher_sha256": baseline["teacher"]["general_archive_sha256"],
            "cases": budget_rows,
        },
    )
    result["budget_matrix_sha256"] = sha(args.output / "budget-matrix.json")
    # One row per line keeps the complete development-case table reviewable.
    table_path = args.output / "cases.json"
    table_path.write_text(
        '{\n"columns": '
        + json.dumps(columns)
        + ',\n"rows": [\n'
        + ",\n".join(json.dumps(row, allow_nan=False) for row in table)
        + "\n]\n}\n"
    )
    result["case_table_sha256"] = sha(table_path)
    write_json(args.output / "results.json", result)
    print("Wrote complete compact readout:", args.output)


if __name__ == "__main__":
    main()
