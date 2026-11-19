#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Join frozen harness assignments to evidence and export paper inputs."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from emet.agent.skills.specs import CHAT_SKILL_SPECS, EQA_SKILL_SPECS

STATUSES = ("passed", "failed", "infrastructure_error", "timeout", "blocked", "pending")
TIMINGS = ("wall_s", "simulation_s", "model_s", "planning_s")


def load_manifest(path, root):
    protocol = json.loads(path.read_text())
    if protocol["schema_version"] != 1:
        raise ValueError("Unsupported assessment version")
    names = [name for group in protocol["groups"] for name in group["tools"]]
    registry = {spec.name for spec in (*CHAT_SKILL_SPECS, *EQA_SKILL_SPECS)}
    if set(names) != registry or len(names) != len(registry):
        raise ValueError(
            f"Tool inventory mismatch: missing={sorted(registry - set(names))}, "
            f"unknown={sorted(set(names) - registry)}, duplicates={len(names) - len(set(names))}"
        )
    dev_cases = json.loads((root / protocol["development_cases"]).read_text())["episodes"]
    cases = list(protocol["cases"])
    for case in dev_cases:
        for stage, adapter, tier in (
            ("geometry", "MuJoCo probe", "geometry control"),
            ("active", "Stretch", "observed simulation"),
        ):
            cases.append({"id": case["id"], "stage": stage, "adapter": adapter, "tier": tier, "status": "pending"})
        if case["family"] in ("recall", "search_act", "insufficient"):
            for arm in ("full_history", "current_only"):
                cases.append(
                    {
                        "id": f"{case['id']}-{arm}",
                        "stage": "replay",
                        "adapter": "recorded views",
                        "tier": "recorded evidence",
                        "status": "pending",
                    }
                )
    keys = [(case["stage"], case["id"]) for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate frozen case")
    for case in cases:
        if case["status"] not in ("pending", "blocked") or (case["status"] == "blocked" and not case.get("reason")):
            raise ValueError(f"Invalid initial status for {case['id']}")
        if "fixture" in case and not (root / case["fixture"]).is_file():
            raise ValueError(f"Missing fixture for {case['id']}")
    return protocol, cases


def audited_results(path, cases):
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["stage"], row["id"])
            if key not in cases or row["status"] not in STATUSES or row["status"] == "pending":
                raise ValueError(f"Invalid assessment result {key}")
            if row["status"] == "blocked":
                if not row.get("reason"):
                    raise ValueError(f"Missing blocker for {key}")
            else:
                artifact = (path.parent / row["artifact"]).resolve()
                if not artifact.is_file() or not row.get("source_sha"):
                    raise ValueError(f"Missing evidence or source for {key}")
                row["artifact"] = str(artifact)
                row["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if row["status"] in ("passed", "failed") and not row.get("independent_check"):
                raise ValueError(f"Missing independent check for {key}")
            rows.append(row)
    return rows


def export(manifest_path, report_paths, out, root):
    protocol, assignments = load_manifest(manifest_path, root)
    index = {(row["stage"], row["id"]): row for row in assignments}
    results = {}
    for path in report_paths:
        for row in audited_results(path, index):
            key = (row["stage"], row["id"])
            if key in results:
                raise ValueError(f"Multiple results for {key}; export retries as a separate cohort")
            results[key] = row
    rows = []
    for key, case in index.items():
        result = results.get(key, {})
        rows.append(
            {
                **case,
                **result,
                "adapter": case["adapter"],
                "tier": case["tier"],
                "reason": result.get("reason", "" if result else case.get("reason", "")),
            }
        )

    out.mkdir(parents=True, exist_ok=False)
    columns = (
        "stage",
        "id",
        "adapter",
        "tier",
        "status",
        "independent_check",
        *TIMINGS,
        "reason",
        "source_sha",
        "artifact",
        "artifact_sha256",
    )
    with (out / "case_outcomes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "# Agent capability matrix",
        "",
        "The evidence column is the audited starting inventory. Run results are in case_outcomes.csv.",
        "",
        "| Group | Registry entries | Adapters | Input evidence | Prior evidence | Independent check | Next check |",
        "|---|---|---|---|---|---|---|",
    ]
    for group in protocol["groups"]:
        tools = ", ".join(f"`{name}`" for name in group["tools"])
        table.append(
            "| "
            + " | ".join(
                (
                    group["id"],
                    tools,
                    ", ".join(group["adapters"]),
                    group["input"],
                    group["evidence"],
                    group["check"],
                    group["next"],
                )
            )
            + " |"
        )
    (out / "capability_matrix.md").write_text("\n".join(table) + "\n")

    counts = {}
    for tier in dict.fromkeys(row["tier"] for row in rows):
        counts[tier] = dict(Counter(row["status"] for row in rows if row["tier"] == tier))
    summary = {
        "base_source": protocol["base_source"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "assigned": len(rows),
        "counts_by_evidence_tier": counts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rows


def plot_outcomes(rows, out):
    import matplotlib.pyplot as plt

    tiers = list(dict.fromkeys(row["tier"] for row in rows))
    fig, (outcome, cost) = plt.subplots(1, 2, figsize=(12, max(4, len(tiers) * 0.54)))
    colors = ("#257e62", "#a13c35", "#784887", "#bd640c", "#52627a", "#b8bec8")
    left = [0] * len(tiers)
    for status, color in zip(STATUSES, colors, strict=True):
        values = [sum(row["tier"] == tier and row["status"] == status for row in rows) for tier in tiers]
        if any(values):
            outcome.barh(tiers, values, left=left, color=color, label=status.replace("_", " "))
        left = [old + new for old, new in zip(left, values, strict=True)]
    outcome.set_xlabel("Assigned cases")
    outcome.invert_yaxis()
    outcome.legend(fontsize=8, loc="best")
    for index, tier in enumerate(tiers):
        seconds = [row["wall_s"] for row in rows if row["tier"] == tier and row.get("wall_s") is not None]
        cost.scatter(seconds, [index] * len(seconds), s=18, color="#264867", alpha=0.7)
        if not seconds:
            cost.text(0.03, index, "Unavailable", transform=cost.get_yaxis_transform(), va="center", color="#596273")
    cost.set_yticks(range(len(tiers)), tiers)
    cost.set_ylim(len(tiers) - 0.5, -0.5)
    cost.set_xlabel("Wall seconds per case")
    cost.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"outcomes_and_cost.{suffix}", dpi=160)
    plt.close(fig)


def plot_recorded_examples(geometry, replay, out):
    """Show one history-dependent success and one retained abstention failure."""
    import matplotlib.pyplot as plt
    from PIL import Image

    manifest_path = geometry / "evidence_manifest.json"
    protocol = json.loads((replay / "protocol.json").read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != protocol["evidence_sha256"]:
        raise ValueError("Replay refers to a different recorded-evidence manifest")
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in (replay / "results.jsonl").read_text().splitlines() if line.strip()]
    paired = {}
    for row in rows:
        paired.setdefault(row["episode_id"], {})[row["arm"]] = row
    eligible = [
        (case_id, arms)
        for case_id, arms in sorted(paired.items())
        if set(arms) == {"full_history", "current_only"} and not any(r.get("error") for r in arms.values())
    ]
    pass_example = next(
        (
            (case, arms)
            for case, arms in eligible
            if arms["full_history"]["correct"] and not arms["current_only"]["correct"]
        ),
        None,
    )
    fail_example = next(
        (
            (case, arms)
            for case, arms in eligible
            if arms["full_history"]["reference_answer"] == "abstain" and not arms["full_history"]["correct"]
        ),
        None,
    )
    selected = [example for example in (pass_example, fail_example) if example is not None]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 3.5 * len(selected)), squeeze=False)
    provenance = []
    try:
        for index, (case_id, arms) in enumerate(selected):
            case = manifest["episodes"][case_id]
            frames = sorted(case["frames"], key=lambda item: item["index"])
            for axis, frame, role in zip(axes[index], frames, ("Earlier view", "Current view"), strict=True):
                path = geometry / frame["rgb"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
                    raise ValueError(f"Recorded image hash mismatch for {case_id}")
                with Image.open(path) as image:
                    axis.imshow(image)
                axis.set_title(role)
                axis.set_axis_off()
            expected = arms["full_history"]["reference_answer"]
            history = arms["full_history"]["parsed"]
            current = arms["current_only"]["parsed"]
            history_answer = "abstain" if history.get("abstain") else history["answer"]
            current_answer = "abstain" if current.get("abstain") else current["answer"]
            axes[index, 0].set_title(f"{case_id} · Earlier view")
            axes[index, 0].text(
                0,
                -0.05,
                f"Expected: {expected} · History: {history_answer} · Current only: {current_answer}",
                transform=axes[index, 0].transAxes,
                va="top",
                fontsize=9,
            )
            provenance.append({"case_id": case_id, "frames": frames, "responses": arms})
        fig.suptitle("Recorded evidence and model answers")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(out / f"recorded_examples.{suffix}", dpi=160)
        (out / "recorded_examples.json").write_text(json.dumps(provenance, indent=2) + "\n")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/eval/harness_assessment_20261119.json"))
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--recorded-geometry", type=Path)
    parser.add_argument("--recorded-replay", type=Path)
    args = parser.parse_args()
    if bool(args.recorded_geometry) != bool(args.recorded_replay):
        parser.error("--recorded-geometry and --recorded-replay must be supplied together")
    root = Path(__file__).resolve().parents[1]
    rows = export(args.manifest, args.results, args.output, root)
    if args.plot:
        plot_outcomes(rows, args.output)
    if args.recorded_geometry:
        plot_recorded_examples(args.recorded_geometry, args.recorded_replay, args.output)


if __name__ == "__main__":
    main()
