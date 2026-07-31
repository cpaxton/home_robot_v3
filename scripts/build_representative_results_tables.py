#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build cross-benchmark representative-sample results tables (markdown + JSON + LaTeX snippet)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score_hmeqa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "correct": 0, "accuracy": 0.0, "question_ids": []}
    correct_ids = [int(r["question_id"]) for r in rows if r.get("correct")]
    wrong_ids = [int(r["question_id"]) for r in rows if not r.get("correct")]
    n = len(rows)
    ok = len(correct_ids)
    return {
        "n": n,
        "correct": ok,
        "accuracy": ok / n if n else 0.0,
        "correct_ids": sorted(correct_ids),
        "wrong_ids": sorted(wrong_ids),
    }


def _summarize_hmeqa_tuning(tuning_run_id: str, results_root: Path) -> dict[str, Any]:
    arms = ["baseline", "no_debias", "no_memory", "no_explore", "graph_eqa_like"]
    slices = ["holdout8", "canonical8"]
    out: dict[str, Any] = {"run_id": tuning_run_id, "arms": {}}
    for arm in arms:
        out["arms"][arm] = {}
        for sl in slices:
            p = results_root / f"subset_{tuning_run_id}_{arm}_{sl}_qwen3_vl.jsonl"
            out["arms"][arm][sl] = _score_hmeqa(_load_jsonl(p))
    return out


def _summarize_hmeqa_rep(run_id: str, results_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run_id": run_id, "methods": {}}
    for method in ("static_graph", "dynagraph"):
        for sl in ("holdout8", "canonical8"):
            p = results_root / f"subset_{run_id}_{sl}_{method}_qwen3_vl.jsonl"
            key = f"{method}_{sl}"
            out["methods"][key] = _score_hmeqa(_load_jsonl(p))
    # Also ingest postfix_nav reference if rep run missing
    ref_tag = "postfix_nav20260705_larger"
    for method in ("static_graph", "dynagraph"):
        for sl in ("holdout8",):
            key = f"{method}_{sl}_ref"
            p = results_root / f"subset_{ref_tag}_{sl}_{method}_qwen3_vl.jsonl"
            if p.is_file():
                out["methods"][key] = _score_hmeqa(_load_jsonl(p))
    return out


def _summarize_ovmm_sim(run_id: str, ovmm_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"tiers": {}, "episodes": []}
    for sub in sorted(ovmm_root.glob(f"{run_id}_*")):
        if not sub.is_dir():
            continue
        tier = sub.name.replace(f"{run_id}_", "")
        backends: dict[str, dict[str, Any]] = {}
        for p in sorted(sub.glob("*.json")):
            row = json.loads(p.read_text(encoding="utf-8"))
            backend = str(row.get("backend", p.stem))
            obj_ok = bool(row.get("find_object_success"))
            rec_ok = bool(row.get("find_recep_success"))
            partial = bool(row.get("find_partial_success"))
            ep = {
                "tier": tier,
                "episode_id": row.get("episode_id", p.stem),
                "backend": backend,
                "find_object_success": obj_ok,
                "find_recep_success": rec_ok,
                "find_partial_success": partial,
            }
            out["episodes"].append(ep)
            stats = backends.setdefault(
                backend,
                {"n": 0, "find_object": 0, "find_recep": 0, "partial": 0},
            )
            stats["n"] += 1
            stats["find_object"] += int(obj_ok)
            stats["find_recep"] += int(rec_ok)
            stats["partial"] += int(partial)
        for stats in backends.values():
            n = stats["n"] or 1
            stats["find_object_rate"] = stats["find_object"] / n
            stats["find_recep_rate"] = stats["find_recep"] / n
            stats["partial_rate"] = stats["partial"] / n
        out["tiers"][tier] = backends
    return out


def _summarize_habitat_ovmm(run_id: str, ovmm_hab_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"backends": {}}
    run_dir = ovmm_hab_root / run_id
    if not run_dir.is_dir():
        return out
    for p in sorted(run_dir.glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        backend = str(row.get("backend", "unknown"))
        out["backends"][backend] = {
            "episode_id": row.get("episode_id"),
            "find_object_success": bool(row.get("find_object_success")),
            "find_recep_success": bool(row.get("find_recep_success")),
            "find_partial_success": bool(row.get("find_partial_success")),
        }
    return out


def _summarize_sqa3d(run_id: str, sqa3d_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"methods": {}}
    for p in sorted(sqa3d_root.glob(f"{run_id}_val_q0_10_*/*.jsonl")):
        rows = _load_jsonl(p)
        if not rows:
            continue
        method = p.parent.name.split("_")[-1]
        n = len(rows)
        ok = sum(1 for r in rows if r.get("em"))
        out["methods"][method] = {"n": n, "correct": ok, "em@1": ok / n if n else 0.0}
    return out


def _summarize_dynamic(run_id: str, dyn_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"backends": {}}
    base = dyn_root / f"{run_id}_explore"
    if not base.is_dir():
        return out
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        jsons = list(sub.glob("*.json"))
        if not jsons:
            continue
        row = json.loads(jsons[0].read_text(encoding="utf-8"))
        out["backends"][sub.name] = {
            "episode_id": row.get("episode_id"),
            "explore_iters": row.get("explore_iters"),
            "coverage": row.get("coverage"),
            "eqa_accuracy": row.get("eqa_accuracy"),
        }
    return out


def _pct(num: int, den: int) -> str:
    if den <= 0:
        return "—"
    return f"{100.0 * num / den:.1f}%"


def _frac(num: int, den: int) -> str:
    if den <= 0:
        return "—"
    return f"{num}/{den}"


def build_tables(
    *,
    run_id: str,
    tuning_run_id: str,
    results_root: Path,
    ovmm_root: Path,
    ovmm_hab_root: Path,
    sqa3d_root: Path,
    dyn_root: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "tuning_run_id": tuning_run_id,
        "hmeqa_tuning": _summarize_hmeqa_tuning(tuning_run_id, results_root),
        "hmeqa_rep": _summarize_hmeqa_rep(run_id, results_root),
        "ovmm_sim": _summarize_ovmm_sim(run_id, ovmm_root),
        "habitat_ovmm": _summarize_habitat_ovmm(run_id, ovmm_hab_root),
        "sqa3d": _summarize_sqa3d(run_id, sqa3d_root),
        "dynamic_explore": _summarize_dynamic(run_id, dyn_root),
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Representative cross-benchmark sample",
        "",
        f"**Run ID:** `{data['run_id']}`  ",
        f"**Habitat tuning matrix:** `{data['tuning_run_id']}`  ",
        f"**VLM:** Qwen3-VL-8B-Instruct (`qwen3_vl`)  ",
        "",
        "## Habitat HM-EQA — dynagraph ablation (tuning matrix)",
        "",
        "| Arm | holdout-8 | canonical-8 | Notes |",
        "|-----|-----------|-------------|-------|",
    ]
    arm_notes = {
        "baseline": "tuned default: debias off, memory on, conservative explore",
        "no_debias": "debias on",
        "no_memory": "memory summary off",
        "no_explore": "explore off",
        "graph_eqa_like": "debias off, memory off, explore off",
    }
    for arm, slices in data["hmeqa_tuning"]["arms"].items():
        h = slices.get("holdout8", {})
        c = slices.get("canonical8", {})
        lines.append(
            f"| `{arm}` | {_frac(h.get('correct', 0), h.get('n', 0))} | "
            f"{_frac(c.get('correct', 0), c.get('n', 0))} | {arm_notes.get(arm, '')} |"
        )

    lines.extend(
        [
            "",
            "## Habitat HM-EQA — method comparison (representative run + reference)",
            "",
            "| Method | slice | n | accuracy |",
            "|--------|-------|---|----------|",
        ]
    )
    for key, stats in sorted(data["hmeqa_rep"]["methods"].items()):
        if not stats.get("n"):
            continue
        lines.append(
            f"| `{key.replace('_', ' ')}` | — | {stats['n']} | "
            f"{_frac(stats['correct'], stats['n'])} ({_pct(stats['correct'], stats['n'])}) |"
        )

    lines.extend(
        [
            "",
            "## OVMM find-phase (sim)",
            "",
            "| Tier | Backend | n | FindObj | FindRec | Partial |",
            "|------|---------|---|---------|---------|---------|",
        ]
    )
    for tier, backends in data["ovmm_sim"]["tiers"].items():
        for backend, stats in sorted(backends.items()):
            lines.append(
                f"| {tier} | `{backend}` | {stats['n']} | "
                f"{_pct(stats['find_object'], stats['n'])} | "
                f"{_pct(stats['find_recep'], stats['n'])} | "
                f"{_pct(stats['partial'], stats['n'])} |"
            )

    if data["habitat_ovmm"]["backends"]:
        lines.extend(
            [
                "",
                "## Habitat OVMM find (proxy)",
                "",
                "| Backend | FindObj | FindRec | Partial |",
                "|---------|---------|---------|---------|",
            ]
        )
        for backend, stats in data["habitat_ovmm"]["backends"].items():
            lines.append(
                f"| `{backend}` | "
                f"{'✓' if stats.get('find_object_success') else '✗'} | "
                f"{'✓' if stats.get('find_recep_success') else '✗'} | "
                f"{'✓' if stats.get('find_partial_success') else '✗'} |"
            )

    if data["sqa3d"]["methods"]:
        lines.extend(
            [
                "",
                "## SQA3D val (q0–10)",
                "",
                "| Method | n | EM@1 |",
                "|--------|---|------|",
            ]
        )
        for method, stats in sorted(data["sqa3d"]["methods"].items()):
            lines.append(
                f"| `{method}` | {stats['n']} | "
                f"{_frac(stats['correct'], stats['n'])} ({_pct(stats['correct'], stats['n'])}) |"
            )

    if data["dynamic_explore"]["backends"]:
        lines.extend(
            [
                "",
                "## Dynamic exploration (smoke)",
                "",
                "| Backend | episode | coverage | EQA acc |",
                "|---------|---------|----------|---------|",
            ]
        )
        for backend, stats in data["dynamic_explore"]["backends"].items():
            cov = stats.get("coverage")
            eqa = stats.get("eqa_accuracy")
            lines.append(
                f"| `{backend}` | {stats.get('episode_id', '?')} | "
                f"{cov if cov is not None else '—'} | {eqa if eqa is not None else '—'} |"
            )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"Artifacts: `~/runs/emet/representative_sample/{data['run_id']}/figures/`",
            "",
            "| Figure | File |",
            "|--------|------|",
            "| HM-EQA ablation (holdout-8) | `hmeqa_ablation_holdout8.png` |",
            "| Top-down maps | `paper_maps_tuning/q*/topdown_map.png` |",
            "| Graph retrieval panels | `retrieval_panels/retrieval_q*.png` |",
            "| OVMM backend bars | `ovmm_findobj_findrec.png` (after OVMM leg) |",
            "| LaTeX table snippet | `../tables/representative_sample.tex` |",
            "",
            f"Monitor run: `tail -f ~/runs/emet/representative_sample/nohup.log`",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex_snippet(data: dict[str, Any]) -> str:
    """LaTeX rows for paste into 05_results.tex."""
    lines = ["% Auto-generated representative sample snippet", "\\begin{table}[t]"]
    lines.append("  \\centering")
    lines.append("  \\caption{Representative cross-benchmark sample (Qwen3-VL-8B unless noted).}")
    lines.append("  \\label{tab:representative_sample}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{@{}llcc@{}}")
    lines.append("    \\toprule")
    lines.append("    Track & Method & $n$ & Metric \\\\")
    lines.append("    \\midrule")

    best_arm = "no_explore"
    h = data["hmeqa_tuning"]["arms"].get(best_arm, {}).get("holdout8", {})
    if h.get("n"):
        lines.append(
            f"    HM-EQA holdout-8 & dynagraph ({best_arm}) & {h['n']} & "
            f"{h['correct']}/{h['n']} ({100*h['correct']/h['n']:.0f}\\%) \\\\"
        )
    ge = data["hmeqa_rep"]["methods"].get("graph_eqa_holdout8_ref") or data["hmeqa_rep"]["methods"].get(
        "graph_eqa_holdout8", {}
    )
    if ge.get("n"):
        lines.append(
            f"    HM-EQA holdout-8 & graph\\_eqa & {ge['n']} & "
            f"{ge['correct']}/{ge['n']} ({100*ge['correct']/ge['n']:.0f}\\%) \\\\"
        )

    s0 = data["ovmm_sim"]["tiers"].get("s0", {})
    for backend in ("dynagraph", "static_graph", "dynamem", "ground_truth"):
        if backend not in s0:
            continue
        st = s0[backend]
        lines.append(
            f"    OVMM S0 & {backend} & {st['n']} & "
            f"FindRec {100*st['find_recep_rate']:.0f}\\% \\\\"
        )

    for method, st in data["sqa3d"]["methods"].items():
        lines.append(
            f"    SQA3D val & {method} & {st['n']} & "
            f"EM@1 {100*st['em@1']:.0f}\\% \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tuning-run-id", default="dynagraph_tune_20260706_110513")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_tables(
        run_id=args.run_id,
        tuning_run_id=args.tuning_run_id,
        results_root=Path.home() / ".cache" / "habitat_eqa" / "results",
        ovmm_root=Path.home() / "runs" / "emet" / "ovmm_find_phase",
        ovmm_hab_root=Path.home() / "runs" / "emet" / "ovmm_habitat",
        sqa3d_root=Path.home() / "runs" / "emet" / "sqa3d",
        dyn_root=Path.home() / "runs" / "emet" / "dynamic_exploration",
    )

    json_path = out_dir / "representative_sample.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    md_path = out_dir / "representative_sample.md"
    md = render_markdown(data)
    md_path.write_text(md, encoding="utf-8")

    tex_path = out_dir / "representative_sample.tex"
    tex_path.write_text(render_latex_snippet(data) + "\n", encoding="utf-8")

    # Also write to docs for discoverability
    docs_path = REPO / "docs" / "experiments" / "representative_sample_results.md"
    docs_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {docs_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
