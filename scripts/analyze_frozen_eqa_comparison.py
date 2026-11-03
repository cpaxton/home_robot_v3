#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Audit frozen packets/budgets, summarize trials and export human review figures."""

import argparse
import hashlib
import json
import re
import statistics
import textwrap
from pathlib import Path

import numpy as np

ARMS = ("images_only", "reference_graph", "emet_answer_packet")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run, cases_path):
    manifest = json.loads((run / "manifest.json").read_text())
    cases = json.loads(cases_path.read_text())
    rows = json.loads((run / "scored_results.json").read_text())
    if sha256(cases_path) != manifest["cases_sha256"]:
        raise ValueError("Model-input cases changed after inference")
    expected_trials = {(case["id"], seed, arm) for case in cases for seed in manifest["arm_seeds"] for arm in ARMS}
    actual_trials = {(row["case_id"], row["seed"], row["arm"]) for row in rows}
    if actual_trials != expected_trials or len(rows) != len(expected_trials):
        raise ValueError("Missing, duplicate or unexpected comparison trials")
    for case in cases:
        hashes = [sha256(cases_path.parent / item["path"]) for item in case["images"]]
        if hashes != manifest["images_sha256"][case["id"]]:
            raise ValueError("Fixture PNGs changed after inference")
    for index, row in enumerate(rows):
        records = [
            json.loads(line) for line in (run / f"trial-{index:04d}/transport_calls.jsonl").read_text().splitlines()
        ]
        caps = [record["request"]["max_tokens"] for record in records]
        expected_caps = [64, 1984] if row["arm"] == "emet_answer_packet" else [2048]
        if caps != expected_caps or row["model_calls"] != len(records):
            raise ValueError("Hidden inference or a mismatched allowance")
        image_calls = [record for record in records if record.get("image_sha256")]
        if len(image_calls) != 1 or image_calls[0]["image_sha256"] != manifest["images_sha256"][row["case_id"]]:
            raise ValueError("Image bytes/order differ between conditions")
        if any("error" in record for record in records):
            raise ValueError("Transport/infrastructure failure needs diagnosis before comparison")
    metrics = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        usages = [usage for row in selected for usage in row["usage"]]
        tokens_known = all(usage is not None for usage in usages)
        metrics[arm] = {
            "planned": len(selected),
            "completed": sum(row["complete"] for row in selected),
            "correct_if_stopped_here": sum(row["correct_if_stopped_here"] is True for row in selected),
            "confident_incorrect": sum(
                row["complete"] and row.get("is_confident", False) and row["correct_if_stopped_here"] is False
                for row in selected
            ),
            "model_calls": sum(row["model_calls"] for row in selected),
            "mean_wall_s": statistics.mean(row["wall_s"] for row in selected),
            "median_wall_s": statistics.median(row["wall_s"] for row in selected),
            "mean_prompt_tokens": sum(usage["prompt_tokens"] for usage in usages) / len(selected)
            if tokens_known
            else None,
            "mean_generated_tokens": sum(usage["completion_tokens"] for usage in usages) / len(selected)
            if tokens_known
            else None,
        }
    result = {
        "scope": "selected development snapshots; correlated trials, not active-agent accuracy",
        "snapshot_count": len(cases),
        "question_count": len({(case["question"], tuple(case["choices"])) for case in cases}),
        "seed_count": len(manifest["arm_seeds"]),
        "analysis_driver_sha256": sha256(__file__),
        "packet_and_budget_audit": "passed",
        "source_commits": manifest["source_commits"],
        "scored_results_sha256": sha256(run / "scored_results.json"),
        "metrics": metrics,
    }
    (run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return cases, rows, metrics


def figures(run, cases_path, cases, metrics, runs_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    output = run / "figures"
    output.mkdir(exist_ok=True)
    provenance = json.loads((cases_path.parent / "manifest.json").read_text())
    sources = {case["id"]: case for case in provenance["cases"]}
    for case in cases:
        fig = plt.figure(figsize=(15, 9))
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], top=0.84, bottom=0.12)
        selected_ids = []
        for index, item in enumerate(case["images"]):
            ax = fig.add_subplot(grid[index // 2, index % 2])
            ax.imshow(Image.open(cases_path.parent / item["path"]))
            sequence = int(re.search(r"sequence=(\d+)", item["label"])[1])
            selected_ids.append(sequence)
            ax.set_title(
                f"Image {index + 1}: {'CURRENT' if index == 0 else 'HISTORY'}, frame {sequence}", color=f"C{index}"
            )
            ax.axis("off")
        ax = fig.add_subplot(grid[:, 2])
        source = sources[case["id"]]
        records = [
            json.loads(line) for line in (runs_root / source["source"] / "frames.jsonl").read_text().splitlines()
        ]
        records = [record for record in records if record["sequence"] <= source["capture"]]
        poses = {record["sequence"]: np.asarray(record["odometry"]["camera_pose"]) for record in records}
        xy = np.array([pose[:2, 3] for pose in poses.values()])
        ax.plot(xy[:, 0], xy[:, 1], color="0.6", linewidth=1)
        for index, sequence in enumerate(selected_ids):
            pose = poses[sequence]
            forward = pose[:2, 2] / np.linalg.norm(pose[:2, 2])
            ax.scatter(*pose[:2, 3], color=f"C{index}", label=f"Image {index + 1}")
            ax.arrow(*pose[:2, 3], *(forward * 0.15), color=f"C{index}", width=0.005, length_includes_head=True)
        ax.set_title("Estimated camera path\nlocal RGB-D odometry")
        span = max(np.ptp(xy, axis=0).max() + 0.5, 1.0)
        center = (xy.min(axis=0) + xy.max(axis=0)) / 2
        ax.set(
            xlim=(center[0] - span / 2, center[0] + span / 2),
            ylim=(center[1] - span / 2, center[1] + span / 2),
            xlabel="local X (m)",
            ylabel="local Y (m)",
        )
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(case["id"] + "\n" + textwrap.fill(case["question"], 100), fontsize=14)
        choices = " | ".join(f"{chr(65 + i)}: {choice}" for i, choice in enumerate(case["choices"]))
        fig.text(0.02, 0.055, textwrap.fill(choices, 140), fontsize=10)
        fig.text(
            0.02,
            0.015,
            "Human audit only. Each condition received the independent PNGs, not this panel. Collector: "
            + source["collector"],
            fontsize=9,
        )
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{case['id']}.{extension}", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["Images only", "GraphEQA", "Emet packet"]
    for ax, key, title in zip(
        axes,
        ["mean_wall_s", "mean_prompt_tokens", "mean_generated_tokens"],
        ["Answer path wall time (s)", "Prompt tokens / trial", "Generated tokens / trial"],
        strict=True,
    ):
        values = [metrics[arm][key] for arm in ARMS]
        bars = ax.bar(labels, values)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(values) * 1.2)
        ax.tick_params(axis="x", labelrotation=15)
    seed_count = len(json.loads((run / "manifest.json").read_text())["arm_seeds"])
    fig.suptitle(
        f"{len(cases)} selected snapshots × {seed_count} seeds; includes keyword calls, excludes collection/model loading",
        fontsize=10,
    )
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"answer-path-costs.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="Root of captured source runs for path figures")
    args = parser.parse_args()
    cases, rows, metrics = audit(args.run, args.cases)
    figures(args.run, args.cases, cases, metrics, args.runs)
    print(json.dumps(metrics, indent=2))
