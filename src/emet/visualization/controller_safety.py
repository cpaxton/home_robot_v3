# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline inspection of episode evaluation reports; never starts a simulator.

Run ``python -m emet.visualization.controller_safety --help`` for usage.
Each input remains a separate cohort; the viewer does not infer matched trials
or model/controller compatibility from labels or coincident seed numbers.
"""

import argparse
import hashlib
import html
import io
import json
import math
from collections import defaultdict
from pathlib import Path

FLAGS = ("unsafe_contact", "wheel_lift", "retention_lost", "invalid_physics")
MEASUREMENTS = (
    "time",
    "peak_contact_force",
    "contact_impulse",
    "peak_measured_jerk",
    "peak_measured_acceleration",
    "joint_error",
    "xy_error",
    "yaw_error",
    "ee_position_error",
    "ee_orientation_error",
    "minimum_clearance",
    "obstacle_clearance",
    "max_payload_translation",
    "max_payload_rotation",
    "rms_measured_jerk",
    "rms_measured_acceleration",
    "max_wheel_gap",
    "min_wheel_load",
    "final_contact_force",
    "post_event_peak_contact_force",
    "event_count",
    "event_elapsed",
    "obstruction_steps",
)


def wilson(hits, count):
    """95% binomial interval; zero observations yield unknown, not zero risk."""
    if not count:
        return None
    z = 1.959963984540054
    p = hits / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0, center - radius), min(1, center + radius)


def outcome(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), bool)]
    hits, count = sum(values), len(values)
    return {"hits": hits, "count": count, "unknown": len(rows) - count, "interval": wilson(hits, count)}


def summarize(rows):
    # Failed setup is an attempted task failure, not an executed safe episode.
    attempted = [dict(r, is_success=False) if r.get("initialization_failed") is True else r for r in rows]
    executed = [r for r in rows if r.get("initialization_failed") is not True]
    supported = [dict(r, wheel_lift=None) if r.get("wheel_support_evaluated") is not True else r for r in executed]
    # Record IDs identify rows, not independent initial states. In reports that
    # omit case_id, seed is scoped to this report's robot/scenario cohort.
    trial_keys = [json.dumps([r.get("case_id"), r["seed"]], sort_keys=True) for r in rows if r.get("seed") is not None]
    record_ids = [json.dumps(r["episode_id"], sort_keys=True) for r in rows if r.get("episode_id") is not None]
    repeats = len(trial_keys) - len(set(trial_keys))
    missing = len(rows) - len(trial_keys)
    repeated_records = len(record_ids) - len(set(record_ids))
    results = {
        "success": outcome(attempted, "is_success"),
        **{key: outcome(supported if key == "wheel_lift" else executed, key) for key in FLAGS},
    }
    if repeats or missing or repeated_records:
        # Keep descriptive counts, but do not imply repeated records provide
        # independent binomial evidence. Absence of repeats is only a check;
        # the displayed intervals still assume independently sampled trials.
        for result in results.values():
            result["interval"] = None
    return {
        "attempts": len(rows),
        "initialization_failures": len(rows) - len(executed),
        "unique_trial_keys": len(set(trial_keys)),
        "repeated_identities": repeats,
        "missing_identities": missing,
        "repeated_record_ids": repeated_records,
        **results,
    }


def metadata_object(report, key):
    value = report.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object or null")
    return value


def reject_nonfinite(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def load_report(label, path):
    raw = path.read_bytes()
    report = json.loads(raw, parse_constant=reject_nonfinite)
    if not isinstance(report, dict) or not isinstance(report.get("episodes"), list) or not report["episodes"]:
        raise ValueError(f"{path}: expected a nonempty episode evaluation report")
    config = metadata_object(report, "config")
    scope = metadata_object(report, "safety_scope")
    support = scope.get("wheel_lift_evaluated")
    if support is not None and not isinstance(support, bool):
        raise ValueError("wheel_lift_evaluated must be boolean or null")
    source = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "config": config,
        "safety_scope": scope,
        "scope": report.get("scope"),
        "teacher_approach": report.get("teacher_approach"),
        "measurement_definitions": metadata_object(report, "measurement_definitions"),
        "force_definition": report.get("force_definition", "definition unspecified"),
        "provenance_note": "File hash identifies this report, not the evaluated model or controller revision.",
    }
    cohorts = defaultdict(list)
    for row in report["episodes"]:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: episode must be an object")
        for key in ("is_success", "initialization_failed", "wheel_support_evaluated", *FLAGS):
            if row.get(key) is not None and not isinstance(row[key], bool):
                raise ValueError(f"{path}: {key} must be boolean or null")
        robot = row.get("robot", report.get("robot", config.get("robot", "unspecified")))
        scenario = row.get(
            "scenario", config.get("base_scenario", config.get("scenario", config.get("task", "unspecified")))
        )
        clean = {
            key: row[key]
            for key in (
                "seed",
                "episode_id",
                "case_id",
                "code",
                "is_success",
                "initialization_failed",
                "wheel_support_evaluated",
                *FLAGS,
                *MEASUREMENTS,
                "config",
            )
            if key in row
        }
        if "wheel_support_evaluated" not in clean:
            clean["wheel_support_evaluated"] = scope.get("wheel_lift_evaluated")
        cohorts[(str(robot), str(scenario))].append(clean)
    return [
        {"name": f"{label} / {robot} / {scenario}", "source": source, "rows": rows, "summary": summarize(rows)}
        for (robot, scenario), rows in cohorts.items()
    ]


def rate_text(value):
    if not value["count"]:
        return "Unknown (0 evaluated)"
    unknown = f"; {value['unknown']} unknown" if value["unknown"] else ""
    if value["interval"] is None:
        return f"{value['hits']}/{value['count']} (CI unavailable: trial independence unverified){unknown}"
    low, high = value["interval"]
    return f"{value['hits']}/{value['count']} ({low:.0%}–{high:.0%} CI){unknown}"


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def force_plot(cohort):
    # Agg performs no GPU/EGL initialization; SVG works without a web service.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    groups = defaultdict(list)
    for row in cohort["rows"]:
        if row.get("initialization_failed") is True:
            continue
        if not all(finite_number(row.get(key)) for key in ("time", "peak_contact_force")):
            continue
        status = (
            "Success"
            if row.get("is_success") is True
            else "Failure"
            if row.get("is_success") is False
            else "Unknown outcome"
        )
        groups[status].append(row)
    for status, rows in groups.items():
        marker, color = {"Success": ("o", "#18755a"), "Failure": ("x", "#b44135"), "Unknown outcome": ("^", "#666666")}[
            status
        ]
        ax.scatter(
            [r["time"] for r in rows],
            [r["peak_contact_force"] for r in rows],
            label=f"{status} (n={len(rows)})",
            marker=marker,
            color=color,
            alpha=0.75,
        )
    if groups:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No measured force/time pairs", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Episode duration (s)", ylabel="Reported peak contact-force metric (N)")
    ax.grid(alpha=0.15)
    output = io.StringIO()
    fig.savefig(output, format="svg")
    plt.close(fig)
    return "<svg" + output.getvalue().split("<svg", 1)[1]


def write_report(cohorts, destination):
    if not cohorts:
        raise ValueError("at least one cohort is required")
    template = Path(__file__).with_suffix(".html").read_text()
    rows, plots = [], []
    for index, cohort in enumerate(cohorts):
        summary = cohort["summary"]
        identity_note = (
            f"; {summary['repeated_identities']} repeated trials, {summary['missing_identities']} unidentified trials, "
            f"{summary['repeated_record_ids']} repeated record IDs"
            if summary["repeated_identities"] or summary["missing_identities"] or summary["repeated_record_ids"]
            else ""
        )
        cells = [
            html.escape(cohort["name"]),
            str(summary["attempts"]) + identity_note,
            str(summary["initialization_failures"]),
        ]
        cells.extend(html.escape(rate_text(summary[key])) for key in ("success", *FLAGS))
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        plots.append(
            f'<div class="plot" id="plot-{index}" {"hidden" if index else ""}>' + force_plot(cohort) + "</div>"
        )
    # Inline script data must not be able to terminate its own HTML element.
    data = json.dumps(cohorts, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    content = template.replace("/*REPORT_DATA*/ null", data, 1)
    content = content.replace("<!--SUMMARY-->", "\n".join(rows)).replace("<!--PLOTS-->", "\n".join(plots))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Episode JSON report; repeat for additional cohorts. No automatic pairing.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Standalone offline HTML report")
    args = parser.parse_args()
    cohorts = []
    for value in args.report:
        label, separator, path = value.partition("=")
        if not label or not separator or not path:
            parser.error("--report requires LABEL=PATH")
        try:
            cohorts.extend(load_report(label, Path(path)))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    write_report(cohorts, args.output)
    print(f"Wrote {args.output}: {len(cohorts)} cohorts, {sum(len(c['rows']) for c in cohorts)} episodes")


if __name__ == "__main__":
    main()
