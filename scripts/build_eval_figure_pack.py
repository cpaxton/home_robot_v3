#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build cross-benchmark eval figures and summary from overnight smoke artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# OVMM find-phase: localize object on start_recep (FindObj) and goal_recep (FindRec).
OVMM_FIND_PHASE_TASK = (
    "Move object from start_recep to goal_recep: FindObj scores the object on its "
    "start receptacle; FindRec scores the goal receptacle for placement."
)
OVMM_METRIC_LABELS = {
    "find_object_success": "FindObj (object on start_recep; typically easier)",
    "find_recep_success": "FindRec (goal_recep for placement; typically harder)",
    "find_partial_success": "mean(FindObj, FindRec)",
}
# Real OVMM paper reference rates (not comparable to this memory-localization harness).
OVMM_PHASE_DIFFICULTY_NOTE = (
    "FindObj is usually easier than FindRec (~70% vs ~30% on real OVMM). "
    "Object-only success is partial progress; both phases are needed for pick/place."
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_hmeqa_jsonls(run_id: str, results_root: Path) -> list[Path]:
    return sorted(results_root.glob(f"*{run_id}*hmeqa*.jsonl"))


def _summarize_hmeqa(paths: list[Path]) -> dict:
    out: dict[str, object] = {"files": [str(p) for p in paths], "methods": {}}
    for p in paths:
        rows = _load_jsonl(p)
        if not rows:
            continue
        method = rows[0].get("method", p.stem)
        n = len(rows)
        correct = sum(1 for r in rows if r.get("correct"))
        out["methods"][method] = {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0}
    return out


def _ovmm_success(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    return bool(value)


def _summarize_ovmm(run_id: str, ovmm_root: Path) -> dict:
    out: dict[str, object] = {
        "task": OVMM_FIND_PHASE_TASK,
        "difficulty_note": OVMM_PHASE_DIFFICULTY_NOTE,
        "primary_metric": "find_recep_success",
        "metric_labels": OVMM_METRIC_LABELS,
        "backends": {},
        "episodes": [],
    }
    for p in sorted(ovmm_root.glob(f"{run_id}_*/*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        backend = row.get("backend", p.stem)
        obj_ok = _ovmm_success(row.get("find_object_success"))
        recep_ok = _ovmm_success(row.get("find_recep_success"))
        partial_ok = _ovmm_success(row.get("find_partial_success"))
        object_query = row.get("object_query", "?")
        start_recep = row.get("start_recep", "?")
        goal_recep = row.get("goal_recep", "?")
        out["episodes"].append(
            {
                "episode_id": row.get("episode_id", p.stem),
                "backend": backend,
                "object_query": object_query,
                "start_recep": start_recep,
                "goal_recep": goal_recep,
                "find_object_success": obj_ok,
                "find_recep_success": recep_ok,
                "find_partial_success": partial_ok,
                "outcome": (
                    "both"
                    if obj_ok and recep_ok
                    else "object_only"
                    if obj_ok
                    else "recep_only"
                    if recep_ok
                    else "neither"
                ),
                "task": f"move {object_query} from {start_recep} to {goal_recep}",
            }
        )
        out["backends"].setdefault(
            backend,
            {
                "n": 0,
                "find_object_success": 0,
                "find_recep_success": 0,
                "partial_success": 0,
                "find_both_success": 0,
                "find_object_only": 0,
                "find_recep_only": 0,
            },
        )
        stats = out["backends"][backend]
        stats["n"] += 1
        stats["find_object_success"] += int(obj_ok)
        stats["find_recep_success"] += int(recep_ok)
        stats["partial_success"] += int(partial_ok)
        if obj_ok and recep_ok:
            stats["find_both_success"] += 1
        elif obj_ok:
            stats["find_object_only"] += 1
        elif recep_ok:
            stats["find_recep_only"] += 1
    for stats in out["backends"].values():
        if not isinstance(stats, dict):
            continue
        n = stats.get("n", 0) or 1
        stats["find_object_success_rate"] = stats.get("find_object_success", 0) / n
        stats["find_recep_success_rate"] = stats.get("find_recep_success", 0) / n
        stats["find_partial_success_rate"] = stats.get("partial_success", 0) / n
        stats["find_both_success_rate"] = stats.get("find_both_success", 0) / n
        stats["find_object_only_rate"] = stats.get("find_object_only", 0) / n
        stats["find_recep_only_rate"] = stats.get("find_recep_only", 0) / n
    return out


def _summarize_sqa3d(run_id: str, sqa3d_root: Path) -> dict:
    out: dict[str, object] = {"files": [], "methods": {}}
    for p in sorted(sqa3d_root.glob(f"{run_id}_*/*.jsonl")):
        rows = _load_jsonl(p)
        if not rows:
            continue
        out["files"].append(str(p))
        method = str(rows[0].get("method", p.parent.name))
        n = len(rows)
        correct = sum(1 for r in rows if r.get("em"))
        methods = out["methods"]
        assert isinstance(methods, dict)
        methods.setdefault(method, {"n": 0, "correct": 0})
        stats = methods[method]
        assert isinstance(stats, dict)
        stats["n"] = int(stats.get("n", 0)) + n
        stats["correct"] = int(stats.get("correct", 0)) + correct
    methods = out["methods"]
    assert isinstance(methods, dict)
    for stats in methods.values():
        if not isinstance(stats, dict):
            continue
        n = int(stats.get("n", 0) or 0)
        stats["em@1"] = (int(stats.get("correct", 0)) / n) if n else 0.0
    return out


def _track_has_data(track_summary: dict) -> bool:
    if track_summary.get("methods"):
        return True
    if track_summary.get("backends"):
        return True
    if track_summary.get("episodes"):
        return True
    return False


def _investigate_status(summary: dict) -> tuple[str, bool]:
    hmeqa_acc = [
        v.get("accuracy", 0.0) for v in summary.get("hmeqa", {}).get("methods", {}).values() if isinstance(v, dict)
    ]
    ovmm_any_success = [
        (v.get("find_object_success", 0) > 0 or v.get("find_recep_success", 0) > 0 or v.get("partial_success", 0) > 0)
        for v in summary.get("ovmm", {}).get("backends", {}).values()
        if isinstance(v, dict)
    ]
    sqa3d_em = [v.get("em@1", 0.0) for v in summary.get("sqa3d", {}).get("methods", {}).values() if isinstance(v, dict)]
    investigate = not (any(a > 0 for a in hmeqa_acc) or any(ovmm_any_success) or any(e > 0 for e in sqa3d_em))
    return ("INVESTIGATE" if investigate else "OK", investigate)


def build_summary(
    run_id: str,
    *,
    results_root: Path,
    ovmm_root: Path,
    sqa3d_root: Path,
    artifact_tag: str | None = None,
) -> dict:
    """Aggregate HM-EQA, OVMM, and SQA3D metrics for a smoke run."""
    hmeqa = _summarize_hmeqa(_find_hmeqa_jsonls(run_id, results_root))
    ovmm = _summarize_ovmm(run_id, ovmm_root)
    sqa3d = _summarize_sqa3d(run_id, sqa3d_root)

    if artifact_tag and artifact_tag != run_id:
        if not _track_has_data(hmeqa):
            fallback = _summarize_hmeqa(_find_hmeqa_jsonls(artifact_tag, results_root))
            if _track_has_data(fallback):
                print(
                    f"WARNING: no HM-EQA files for run_id={run_id!r}; using artifact_tag={artifact_tag!r}",
                    file=sys.stderr,
                )
                hmeqa = fallback
        if not _track_has_data(ovmm):
            fallback = _summarize_ovmm(artifact_tag, ovmm_root)
            if _track_has_data(fallback):
                print(
                    f"WARNING: no OVMM files for run_id={run_id!r}; using artifact_tag={artifact_tag!r}",
                    file=sys.stderr,
                )
                ovmm = fallback
        if not _track_has_data(sqa3d):
            fallback = _summarize_sqa3d(artifact_tag, sqa3d_root)
            if _track_has_data(fallback):
                print(
                    f"WARNING: no SQA3D files for run_id={run_id!r}; using artifact_tag={artifact_tag!r}",
                    file=sys.stderr,
                )
                sqa3d = fallback

    summary: dict = {
        "run_id": run_id,
        "artifact_tag": artifact_tag or run_id,
        "hmeqa": hmeqa,
        "ovmm": ovmm,
        "sqa3d": sqa3d,
    }
    status, investigate = _investigate_status(summary)
    summary["status"] = status
    summary["investigate"] = investigate
    return summary


def write_summary_csv(summary: dict, path: Path) -> None:
    csv_lines = ["track,method,n,success_metric,value"]
    for method, stats in summary.get("hmeqa", {}).get("methods", {}).items():
        if isinstance(stats, dict):
            csv_lines.append(f"hmeqa,{method},{stats.get('n', 0)},accuracy,{stats.get('accuracy', 0.0):.4f}")
    for backend, stats in summary.get("ovmm", {}).get("backends", {}).items():
        if isinstance(stats, dict):
            n = stats.get("n", 0) or 1
            for metric in (
                "find_object_success_rate",
                "find_recep_success_rate",
                "find_partial_success_rate",
                "find_both_success_rate",
                "find_object_only_rate",
            ):
                rate = stats.get(metric, stats.get(metric.replace("_rate", ""), 0) / n)
                csv_lines.append(f"ovmm,{backend},{n},{metric},{rate:.4f}")
    for method, stats in summary.get("sqa3d", {}).get("methods", {}).items():
        if isinstance(stats, dict):
            csv_lines.append(f"sqa3d,{method},{stats.get('n', 0)},em@1,{stats.get('em@1', 0.0):.4f}")
    path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def _print_ovmm_digest(ovmm_summary: dict) -> None:
    backends = ovmm_summary.get("backends", {})
    if not backends:
        print("OVMM: no runs found")
        return
    print("OVMM find-phase (FindObj easier; FindRec harder):")
    for backend, stats in sorted(backends.items()):
        if not isinstance(stats, dict):
            continue
        n = stats.get("n", 0)
        obj = stats.get("find_object_success_rate", 0.0)
        recep = stats.get("find_recep_success_rate", 0.0)
        both = stats.get("find_both_success_rate", 0.0)
        obj_only = stats.get("find_object_only_rate", 0.0)
        print(f"  {backend}: n={n} FindObj={obj:.0%} FindRec={recep:.0%} both={both:.0%} object_only={obj_only:.0%}")


def _plot_ovmm_success_bars(ovmm_summary: dict, out_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    backends: list[str] = []
    obj_rates: list[float] = []
    recep_rates: list[float] = []
    for backend, stats in ovmm_summary.get("backends", {}).items():
        if not isinstance(stats, dict):
            continue
        backends.append(str(backend))
        obj_rates.append(float(stats.get("find_object_success_rate", 0.0)))
        recep_rates.append(float(stats.get("find_recep_success_rate", 0.0)))
    if not backends:
        return None

    x = np.arange(len(backends))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(4, 2 * len(backends)), 4))
    ax.bar(x - width / 2, obj_rates, width, label="FindObj (easier)")
    ax.bar(x + width / 2, recep_rates, width, label="FindRec (harder)")
    ax.set_ylabel("success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("OVMM find-phase: FindObj vs FindRec")
    fig.text(
        0.5,
        0.01,
        "FindObj is typically easier (~70% vs ~30% on real OVMM); object-only = partial.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = out_dir / "ovmm_findobj_findrec.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_map_grid(episodes_root: Path, run_id: str, out_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return None

    maps: list[tuple[str, np.ndarray]] = []
    for bundle in sorted(episodes_root.glob(f"{run_id}*/*/topdown_map.png")):
        maps.append((bundle.parent.name, np.asarray(Image.open(bundle))))
    if not maps:
        for bundle in sorted(episodes_root.rglob("topdown_map.png")):
            if run_id in str(bundle):
                maps.append((bundle.parent.name, np.asarray(Image.open(bundle))))
    if not maps:
        return None

    n = len(maps)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (title, img) in zip(axes_flat, maps, strict=False):
        ax.imshow(img)
        ax.set_title(title[:40], fontsize=8)
        ax.axis("off")
    for ax in axes_flat[len(maps) :]:
        ax.axis("off")
    fig.tight_layout()
    out = out_dir / "topdown_map_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _artifact_glob_tag(run_id: str, artifact_tag: str | None) -> str:
    if artifact_tag and artifact_tag != run_id:
        return artifact_tag
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="Fallback tag for artifact globs when it differs from --run-id (e.g. overnight TAG)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--dynagraph-tuning-root",
        type=Path,
        default=Path.home() / "runs" / "emet" / "dynagraph_tuning",
        help="Also scan tuning matrix JSONL copies under <root>/<run-id>/",
    )
    parser.add_argument(
        "--render-retrieval-panels",
        action="store_true",
        help="Run render_graph_retrieval_panel.py on first HM-EQA JSONL",
    )
    args = parser.parse_args()

    run_id = args.run_id
    artifact_tag = args.artifact_tag
    if artifact_tag and artifact_tag != run_id:
        print(
            f"WARNING: --artifact-tag={artifact_tag!r} differs from --run-id={run_id!r}; "
            "summary run_id stays --run-id; globs fall back to artifact-tag when empty.",
            file=sys.stderr,
        )

    habitat_root = Path.home() / ".cache" / "habitat_eqa"
    results_root = habitat_root / "results"
    episodes_root = habitat_root / "episodes"
    ovmm_root = Path.home() / "runs" / "emet" / "ovmm_habitat"
    sqa3d_root = Path.home() / "runs" / "emet" / "sqa3d"
    out_dir = args.output_dir or (Path.home() / "runs" / "emet" / "eval_smoke" / run_id / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        run_id,
        results_root=results_root,
        ovmm_root=ovmm_root,
        sqa3d_root=sqa3d_root,
        artifact_tag=artifact_tag,
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    figures_manifest: dict[str, object] = {
        "run_id": run_id,
        "summary_json": str(summary_path),
        "figures": [],
    }

    write_summary_csv(summary, out_dir / "summary.csv")
    _print_ovmm_digest(summary["ovmm"])

    if args.summary_only:
        return

    map_tag = _artifact_glob_tag(run_id, artifact_tag)
    grid = _plot_map_grid(episodes_root, map_tag, out_dir)
    if grid is None and artifact_tag and artifact_tag != run_id:
        grid = _plot_map_grid(episodes_root, run_id, out_dir)
    if grid:
        print(f"wrote {grid}")
        figures_manifest["figures"].append(str(grid))

    ovmm_bars = _plot_ovmm_success_bars(summary["ovmm"], out_dir)
    if ovmm_bars:
        print(f"wrote {ovmm_bars}")
        figures_manifest["figures"].append(str(ovmm_bars))

    tuning_dir = args.dynagraph_tuning_root.expanduser() / run_id
    if tuning_dir.is_dir():
        for j in sorted(tuning_dir.glob("*.jsonl")):
            summary.setdefault("dynagraph_tuning_jsonl", []).append(str(j))
        maps_script = REPO / "scripts" / "render_paper_map_figures.py"
        if maps_script.is_file():
            import subprocess

            maps_out = out_dir / "paper_maps"
            subprocess.run(
                [
                    sys.executable,
                    str(maps_script),
                    "--run-id",
                    run_id,
                    "--output-dir",
                    str(maps_out),
                ],
                check=False,
            )
            manifest_maps = maps_out / "maps_manifest.json"
            if manifest_maps.is_file():
                figures_manifest["paper_maps_manifest"] = str(manifest_maps)

    if args.render_retrieval_panels:
        hmeqa_files = _find_hmeqa_jsonls(run_id, results_root)
        if not hmeqa_files and artifact_tag:
            hmeqa_files = _find_hmeqa_jsonls(artifact_tag, results_root)
        if hmeqa_files:
            import subprocess

            retrieval_out = out_dir / "retrieval_panels"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "render_graph_retrieval_panel.py"),
                    "--jsonl",
                    str(hmeqa_files[0]),
                    "--output-dir",
                    str(retrieval_out),
                ],
                check=False,
            )
            rm = retrieval_out / "retrieval_manifest.json"
            if rm.is_file():
                figures_manifest["retrieval_manifest"] = str(rm)

    (out_dir / "figures_manifest.json").write_text(
        json.dumps(figures_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    sqa3d_glob_tag = map_tag
    sqa3d_jsonls = list(sqa3d_root.glob(f"{sqa3d_glob_tag}_*/*.jsonl"))
    if not sqa3d_jsonls and artifact_tag and artifact_tag != run_id:
        sqa3d_jsonls = list(sqa3d_root.glob(f"{run_id}_*/*.jsonl"))
    for p in sqa3d_jsonls:
        try:
            from emet.benchmarks.sqa3d.analysis import generate_sqa3d_figure_bundle

            sub = out_dir / p.parent.name
            sub.mkdir(parents=True, exist_ok=True)
            generate_sqa3d_figure_bundle(p, sub)
            print(f"sqa3d figures: {sub}")
        except Exception as exc:
            print(f"sqa3d plot skip {p}: {exc}")


if __name__ == "__main__":
    main()
