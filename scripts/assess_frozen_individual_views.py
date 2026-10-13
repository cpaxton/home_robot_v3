#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
"""Isolate selected RGB evidence from multi-view presentation effects.

Diagnostic only: each image is assessed alone as its own current view, NOT as
proof of visibility from the robot's final physical position. Run serially via
the GPU job queue. Never count these as online task successes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = {row["view_id"]: row for row in manifest["frames"]}
    question = json.loads((Path(manifest["source"]) / "eqa_decisions/iter_001/meta.json").read_text())["question"]
    import torch

    from emet.core.parameters import get_parameters
    from emet.eval.agentic_vlm_assess import assess_view_with_vlm
    from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

    _, client = build_graph_eqa_vlm_clients(parameters=get_parameters("configs/emet/default.yaml"), device="cuda")
    with args.output.open("x") as output:
        for repeat in range(args.repeats):
            for view_id in args.views:
                rgb = np.asarray(Image.open(args.manifest.parent / rows[view_id]["image_path"]).convert("RGB"))
                torch.manual_seed(repeat)
                result = assess_view_with_vlm(client, question=question, rgb=rgb, is_mcq=True)
                if not result.raw:
                    raise RuntimeError("empty model response")
                row = {
                    "view_id": view_id,
                    "repeat": repeat,
                    "diagnostic": "isolated frame, not final-position visibility",
                    **result.to_dict(),
                }
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
