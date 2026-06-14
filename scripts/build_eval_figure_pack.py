#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build cross-benchmark eval figures and summary from overnight smoke artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def _summarize_ovmm(run_id: str, ovmm_root: Path) -> dict:
    out: dict[str, object] = {"backends": {}}
    for p in sorted(ovmm_root.glob(f"{run_id}_*/*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        backend = row.get("backend", p.stem)
        partial = bool(row.get("find_partial_success"))
        out["backends"].setdefault(backend, {"n": 0, "partial_success": 0})
        out["backends"][backend]["n"] += 1
        out["backends"][backend]["partial_success"] += int(partial)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id
    habitat_root = Path.home() / ".cache" / "habitat_eqa"
    results_root = habitat_root / "results"
    episodes_root = habitat_root / "episodes"
    ovmm_root = Path.home() / "runs" / "emet" / "ovmm_habitat"
    out_dir = args.output_dir or (Path.home() / "runs" / "emet" / "eval_smoke" / run_id / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    hmeqa_paths = _find_hmeqa_jsonls(run_id, results_root)
    summary = {
        "run_id": run_id,
        "hmeqa": _summarize_hmeqa(hmeqa_paths),
        "ovmm": _summarize_ovmm(run_id, ovmm_root),
    }

    hmeqa_acc = [
        v.get("accuracy", 0.0)
        for v in summary["hmeqa"].get("methods", {}).values()
        if isinstance(v, dict)
    ]
    ovmm_partial = [
        (v.get("partial_success", 0) > 0)
        for v in summary["ovmm"].get("backends", {}).values()
        if isinstance(v, dict)
    ]
    investigate = not (any(a > 0 for a in hmeqa_acc) or any(ovmm_partial))
    summary["status"] = "INVESTIGATE" if investigate else "OK"
    summary["investigate"] = investigate

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    csv_lines = ["track,method,n,success_metric,value"]
    for method, stats in summary["hmeqa"].get("methods", {}).items():
        if isinstance(stats, dict):
            csv_lines.append(
                f"hmeqa,{method},{stats.get('n',0)},accuracy,{stats.get('accuracy',0.0):.4f}"
            )
    for backend, stats in summary["ovmm"].get("backends", {}).items():
        if isinstance(stats, dict):
            n = stats.get("n", 0) or 1
            rate = stats.get("partial_success", 0) / n
            csv_lines.append(f"ovmm,{backend},{n},find_partial_success_rate,{rate:.4f}")
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    if args.summary_only:
        return

    grid = _plot_map_grid(episodes_root, run_id, out_dir)
    if grid:
        print(f"wrote {grid}")

    sqa3d_jsonls = list((Path.home() / "runs" / "emet" / "sqa3d").glob(f"{run_id}_*/*.jsonl"))
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
