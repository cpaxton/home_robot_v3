# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Cache YOLOE or box-prompted SAM2 proposals; no semantic acceptance or GT reads."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["yoloe", "sam2"], default="yoloe")
    parser.add_argument("--boxes-from", type=Path, help="Exact saved Qwen results to prompt SAM2")
    args = parser.parse_args()
    if (args.backend == "sam2") != bool(args.boxes_from):
        parser.error("SAM2 requires --boxes-from; YOLOE must not use it")
    rows = yaml.safe_load(args.manifest.read_text())
    boxes = json.loads(args.boxes_from.read_text()) if args.boxes_from else None
    if boxes is not None and [row["input"] for row in boxes] != rows:
        raise ValueError("Saved boxes must match exact inputs and order")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.backend == "sam2":
        from emet.perception.detection.sam2 import SAM2Perception

        detector = SAM2Perception(configuration="s")
    else:
        from emet.perception.detection.yoloe import get_shared_yoloe_perception

        detector = get_shared_yoloe_perception(confidence_threshold=0.05, device="cuda", size="l")
    records = []
    for index, row in enumerate(rows):
        with np.load(row["arrays"], allow_pickle=False) as arrays:
            rgb = np.asarray(Image.open(row["rgb"]).convert("RGB")) if row.get("rgb") else arrays["rgb"]
        start = time.monotonic()
        if boxes is not None:
            from grounding_ablation import expand_box

            selection = boxes[index]["selection"]
            xyxy = []
            if selection.get("verified") is True:
                try:
                    box = expand_box(selection.get("box"), 0)
                    height, width = rgb.shape[:2]
                    xyxy = [np.asarray(box) * [width, height, width, height] / 1000]
                except ValueError:
                    pass  # malformed localization cannot create a segmentation prompt
            masks = detector.segment(rgb, np.asarray(xyxy)).astype(bool)
            metadata = {}
        else:
            _, instances, metadata = detector.predict(rgb, draw_instance_predictions=False, vocabulary=[row["query"]])
            ids = np.unique(instances[instances >= 0])
            masks = np.stack([instances == i for i in ids]) if len(ids) else np.empty((0, *rgb.shape[:2]), dtype=bool)
        np.savez_compressed(args.output_dir / f"{index}-masks.npz", masks=masks)
        records.append(
            {
                "input": row,
                "proposal_count": len(masks),
                "elapsed_s": time.monotonic() - start,
                "backend": "sam2.1_s" if boxes is not None else "yoloe_l",
                "boxes_source": str(args.boxes_from) if args.boxes_from else None,
                "confidence_threshold": None if boxes is not None else 0.05,
                "scores": np.asarray(metadata.get("instance_scores", [])).tolist(),
            }
        )
        (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2))
        print(f"{index}: {len(masks)} proposals for {row['query']}", flush=True)


if __name__ == "__main__":
    main()
