#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Show actual initial/final camera evidence and measured motion; do not score tasks."""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cases = sorted(args.run.glob("cap-dev-*/result.json"))
    if not cases:
        p.error("No completed result artifacts")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 2, figsize=(10, 5 * len(cases)), squeeze=False)
    rows = []
    for index, path in enumerate(cases):
        root = path.parent
        record = json.loads(path.read_text())
        measured = json.loads((root / "measured.json").read_text())
        before, after = measured[0], measured[-1]
        start = np.asarray(before["base_world_xyt"])
        end = np.asarray(after["base_world_xyt"])
        yaw = float(np.degrees(np.arctan2(np.sin(end[2] - start[2]), np.cos(end[2] - start[2]))))
        response = sorted((root / "model_requests").glob("*/response.json"))[-1]
        content = json.loads(response.read_text())["choices"][0]["message"]["content"]
        try:
            answer = json.loads(content).get("message", content)
        except ValueError:
            answer = content
        for column, sample in enumerate((before, after)):
            image = root / f"{sample['tag']}.png"
            axes[index, column].imshow(Image.open(image))
            axes[index, column].axis("off")
            axes[index, column].set_title(f"{record['episode_id']} — {'initial' if column == 0 else 'final'} RGB")
        caption = (
            f"Question: {record['turns'][-1]}\nMeasured total yaw change: {yaw:.3f} deg; "
            f"travel: {np.linalg.norm(end[:2] - start[:2]):.4f} m. Qwen: {answer}"
        )
        axes[index, 0].text(
            0, -0.08, textwrap.fill(caption, 90), transform=axes[index, 0].transAxes, va="top", fontsize=9
        )
        rows.append(
            {
                "episode_id": record["episode_id"],
                "source_sha": record["source_sha"],
                "live_fixture_version": record.get("live_fixture_version"),
                "protocol_complete": record["complete"],
                "yaw_change_deg": yaw,
                "travel_m": float(np.linalg.norm(end[:2] - start[:2])),
                "final_model_message": answer,
                "response_artifact": str(response),
                "initial_rgb": str(root / f"{before['tag']}.png"),
                "final_rgb": str(root / f"{after['tag']}.png"),
                "task_success": None,
                "input_sha256": {
                    str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in [
                        path,
                        root / "measured.json",
                        response,
                        root / f"{before['tag']}.png",
                        root / f"{after['tag']}.png",
                    ]
                },
                "note": "Protocol completion and correct color alone do not qualify observe-turn-recall",
            }
        )
    fig.suptitle("Live CHAT evidence and measured motion — task acceptance requires both", fontsize=13)
    fig.tight_layout(h_pad=5, rect=(0, 0.06, 1, 0.96))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "live_evidence_and_motion.png", dpi=170, bbox_inches="tight")
    fig.savefig(args.output / "live_evidence_and_motion.pdf", bbox_inches="tight")
    plt.close(fig)
    (args.output / "panel_inputs.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
