"""Small, reproducible paper bundles from complete preflight certificates."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from .recording import load_run, write_json


def architecture_figure(destination: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    boxes = [
        (0.2, 2.9, "Robot observations\nRGB-D + pose", True),
        (3.3, 3.65, "Fast voxel map\ngeometry + retrieval", False),
        (3.3, 2.05, "Sparse semantic graph\nevidence + temporal state", False),
        (6.4, 2.9, "Existing task agent\nquery / confirm / revise", False),
        (9.4, 2.9, "Shared CHAT tools\nplan / execute", True),
        (9.4, 0.6, "Assisted simulator\nrby1 fixture adapter", True),
        (6.4, 0.6, "Private evaluator\nmeasured goal predicates", True),
        (3.3, 0.6, "Recorded evidence\nreplay / figures / tables", True),
    ]
    for x, y, label, exercised in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.35,
                0.95,
                boxstyle="round,pad=.06",
                linewidth=1.5,
                edgecolor="#246480" if exercised else "#778899",
                facecolor="#e7f2f7" if exercised else "#f3f5f7",
                linestyle="-" if exercised else "--",
            )
        )
        ax.text(x + 1.175, y + 0.475, label, ha="center", va="center", fontsize=10)
    for start, end, pending in [
        ((2.6, 3.5), (3.25, 4), True),
        ((2.6, 3.1), (3.25, 2.55), True),
        ((5.7, 4), (6.35, 3.55), True),
        ((5.7, 2.55), (6.35, 3.2), True),
        ((8.8, 3.4), (9.35, 3.4), True),
        ((10.6, 2.85), (10.6, 1.6), False),
        ((9.35, 1.1), (8.8, 1.1), False),
        ((6.35, 1.1), (5.7, 1.1), False),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="->", mutation_scale=14, color="#64748b", linestyle="--" if pending else "-"
            )
        )
    ax.text(
        0.2,
        0.25,
        "Solid: exercised by assisted preflight. Dashed: existing-agent/memory integration still to validate.",
        fontsize=10,
        color="#334155",
    )
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def export_paper(batch: Path, paper: Path) -> dict:
    certificate = json.loads((batch / "certificate.json").read_text())
    grouped = defaultdict(list)
    seen = set()
    implementation = None
    profiles = {}
    for row in certificate["rows"]:
        path = (batch / row["path"]).resolve()
        if not path.is_relative_to(batch.resolve()) or path in seen:
            raise ValueError("unsafe or repeated certificate run path")
        seen.add(path)
        manifest, events, metrics = load_run(path)
        artifacts = json.loads((path / "artifacts.json").read_text())
        if (
            manifest["control"] != "witness"
            or manifest["suite_fingerprint"] != certificate["suite_fingerprint"]
            or metrics["status"] != "completed"
            or not metrics["success"]
            or not artifacts["complete"]
            or artifacts["source_event_hash"] != metrics["event_hash"]
        ):
            raise ValueError("paper export requires complete matched assisted witnesses")
        hashes = manifest.get("implementation_sha256")
        if implementation is not None and hashes != implementation:
            raise ValueError("implementation changed between repeats")
        implementation = hashes
        eid = manifest["episode"]["id"]
        profile = {key: manifest[key] for key in ("episode_fingerprint", "robot", "seed", "assistance", "mode")}
        if eid in profiles and profiles[eid] != profile:
            raise ValueError("episode or assistance changed between repeats")
        profiles[eid] = profile
        for name in artifacts["artifacts"]:
            artifact = (path / name).resolve()
            if not artifact.is_relative_to(path):
                raise ValueError(f"unsafe artifact path {name}")
            if not artifact.is_file():
                raise ValueError(f"missing artifact {name}")
            if artifacts.get("sha256", {}).get(name) != hashlib.sha256(artifact.read_bytes()).hexdigest():
                raise ValueError(f"artifact hash mismatch: {name}")
        grouped[manifest["episode"]["id"]].append((path, manifest, metrics))
    if not grouped or any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("paper preflight export requires exactly three repeats per selected episode")
    data = paper / "data" / "agent_task_preflight"
    figures = paper / "figs"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "scope": certificate["scope"],
        "suite_fingerprint": certificate["suite_fingerprint"],
        "implementation_sha256": implementation,
        "episodes": [],
        "paid_cost_usd": 0,
    }
    table_rows = []
    for eid, rows in grouped.items():
        representative, manifest, _ = rows[0]
        dst = figures / f"agent_task_{eid}.pdf"
        shutil.copyfile(representative / "overview.pdf", dst)
        summary["episodes"].append(
            {
                "id": eid,
                "repeats": 3,
                "successful": 3,
                "goal_count": rows[0][2]["total"],
                "robot": manifest["robot"],
                "assistance": manifest["assistance"],
                "source_revisions": sorted({m["source_revision"] for _, m, _ in rows}),
                "event_hashes": [m["event_hash"] for _, _, m in rows],
                "figure_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            }
        )
        label = eid.replace("_", " ").capitalize()
        table_rows.append(f"{label} & 3/3 & {rows[0][2]['total']} \\\\")
    architecture_figure(figures / "agent_task_architecture.pdf")
    write_json(data / "summary.json", summary)
    (data / "table.tex").write_text(
        "% Generated by emet eval agent-tasks export --paper-dir; assisted controls only.\n"
        "\\begin{tabular}{lrr}\n\\toprule\nCase & Successful resets & Placement goals \\\\\n\\midrule\n"
        + "\n".join(table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return summary


def export_agent_paper(runs: list[Path], paper: Path, *, bundle: str = "agent_task_policy") -> dict:
    """Publish actual local-policy outcomes, including failures, separately from witnesses."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", bundle):
        raise ValueError("bundle must be a safe lowercase identifier")
    rows = []
    sources = []
    seen = set()
    for root in runs:
        manifest, events, metrics = load_run(root)
        if manifest.get("control") != "local_agent" or manifest["models"].get("mock"):
            raise ValueError("policy export requires an actual local-model run")
        if not any(e["kind"] == "model_output" for e in events) or not metrics.get("agent_ran"):
            raise ValueError("policy did not execute")
        artifacts = json.loads((root / "artifacts.json").read_text())
        if not artifacts["complete"] or artifacts["source_event_hash"] != metrics["event_hash"]:
            raise ValueError("policy export requires complete matched visual evidence")
        for name in ("overview.pdf", "overview.png", "report.html", "storyboard.png", "episode.rrd", "episode.mp4"):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != artifacts["sha256"][name]:
                raise ValueError(f"artifact hash mismatch: {name}")
        key = (manifest["suite"]["name"], manifest["episode"]["id"])
        if key in seen:
            raise ValueError("duplicate policy layout/task; export one declared diagnostic per pair")
        seen.add(key)
        rows.append(
            {
                "suite": key[0],
                "episode": key[1],
                "model": manifest["models"],
                "model_runtime": next((e["policy"] for e in events if e["kind"] == "model_ready"), {}),
                "suite_fingerprint": manifest["suite_fingerprint"],
                "episode_fingerprint": manifest["episode_fingerprint"],
                "elapsed_s": metrics.get("elapsed_s"),
                "response_repairs": metrics.get("response_repairs", 0),
                "temporal_order_passed": metrics.get("temporal_order_passed"),
                "model_call_elapsed_s": sum(
                    e["policy"].get("elapsed_s", 0) for e in events if e["kind"] == "model_output"
                ),
                "assistance": manifest["assistance"],
                "status": metrics["status"],
                "completed": metrics["completed"],
                "total": metrics["total"],
                "model_rounds": metrics["model_rounds"],
                "actions": metrics["actions"],
                "event_hash": metrics["event_hash"],
                "source_revision": manifest["source_revision"],
                "implementation_sha256": manifest["implementation_sha256"],
                "shared_implementation_sha256": manifest.get("shared_implementation_sha256", {}),
                "observation_revisions": len(metrics.get("observed_memory_revisions", [])),
                "agent_status": metrics["agent_status"],
                "source_run": str(root.resolve()),
            }
        )
        sources.append((root, key))
    if not rows:
        raise ValueError("no policy runs selected")
    data, figures = paper / "data" / bundle, paper / "figs"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for row, (root, key) in zip(rows, sources, strict=True):
        dst = figures / f"{bundle}_{key[1]}.pdf"
        if sum(other["episode"] == key[1] for other in rows) > 1:
            dst = figures / f"{bundle}_{hashlib.sha256(key[0].encode()).hexdigest()[:8]}_{key[1]}.pdf"
        shutil.copyfile(root / "overview.pdf", dst)
        row["figure"] = dst.name
        row["figure_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
    summary = {
        "scope": "one local-policy diagnostic per selected layout/task; assisted perception and skills",
        "paid_cost_usd": 0,
        "runs": rows,
    }
    write_json(data / "summary.json", summary)
    table_rows = [
        f"{r['episode'].replace('_', ' ')} & {r['completed']}/{r['total']} & "
        f"{r['model_rounds']} & {r['status'].replace('_', ' ')} \\\\"
        for r in rows
    ]
    (data / "table.tex").write_text(
        "% Local policy diagnostics, not assisted witnesses.\n"
        "\\begin{tabular}{lrrl}\n\\toprule\nTask & Goals & Decisions & Outcome \\\\\n\\midrule\n"
        + "\n".join(table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return summary
