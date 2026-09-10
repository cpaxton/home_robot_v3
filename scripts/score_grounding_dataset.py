# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Score saved inference against separate simulator masks; never rerun inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from grounding_ablation import expand_box


def score(truth_path, results_path):
    truth = {r["input"]["arrays"] + "\n" + r["input"]["query"]: r for r in json.loads(truth_path.read_text())}
    rows = []
    for index, result in enumerate(json.loads(results_path.read_text())):
        expected = truth[result["input"]["arrays"] + "\n" + result["input"]["query"]]
        with np.load(expected["mask_file"]) as data:
            gt = data["mask"]
        row = {k: expected[k] for k in ("scene", "split", "target", "view", "visible_pixels")}
        row.update(
            accepted=bool(result["audit"]["valid"]),
            box_recall=0.0,
            box_iou=0.0,
            purity=None,
            surface_recall=None,
            elapsed_s=sum(r["elapsed_s"] for r in result.get("effective_requests", [])),
        )
        try:
            box = expand_box(result["selection"].get("box"), 0)
            h, w = gt.shape
            x0, y0 = np.floor(np.array(box[:2]) * [w, h] / 1000).astype(int)
            x1, y1 = np.ceil(np.array(box[2:]) * [w, h] / 1000).astype(int)
            pred = np.zeros_like(gt)
            pred[y0:y1, x0:x1] = True
            row["box_recall"] = float((gt & pred).sum() / max(1, gt.sum()))
            row["box_iou"] = float((gt & pred).sum() / max(1, (gt | pred).sum()))
        except ValueError:
            pass
        if row["accepted"]:
            with np.load(results_path.parent / f"{index}-support.npz") as data:
                mask = data["mask"]
            row["purity"] = float((mask & gt).sum() / max(1, mask.sum()))
            row["surface_recall"] = float((mask & gt).sum() / max(1, gt.sum()))
        # Diagnostic surface gate, NOT a full-object pose or grasp acceptance gate.
        row["pure_surface"] = row["accepted"] and row["purity"] >= 0.95
        row["false_accept"] = row["accepted"] and not row["pure_surface"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    rows = score(args.truth, args.results)
    (args.results.parent / "scores.json").write_text(json.dumps(rows, indent=2))
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in rows if r["split"] == split]
        print(
            json.dumps(
                {
                    "split": split,
                    "queries": len(selected),
                    "visible": sum(r["visible_pixels"] > 0 for r in selected),
                    "pure_surface": sum(r["pure_surface"] for r in selected),
                    "false_accept": sum(r["false_accept"] for r in selected),
                    "mean_latency_s": np.mean([r["elapsed_s"] for r in selected]),
                }
            )
        )
