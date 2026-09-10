# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Cache query-conditioned YOLOE proposals; no semantic acceptance or GT reads."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from emet.perception.detection.yoloe import get_shared_yoloe_perception


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detector = get_shared_yoloe_perception(confidence_threshold=0.05, device="cuda", size="l")
    records = []
    for index, row in enumerate(yaml.safe_load(args.manifest.read_text())):
        with np.load(row["arrays"], allow_pickle=False) as arrays:
            rgb = np.asarray(Image.open(row["rgb"]).convert("RGB")) if row.get("rgb") else arrays["rgb"]
        start = time.monotonic()
        _, instances, metadata = detector.predict(rgb, draw_instance_predictions=False, vocabulary=[row["query"]])
        ids = np.unique(instances[instances >= 0])
        masks = np.stack([instances == i for i in ids]) if len(ids) else np.empty((0, *rgb.shape[:2]), dtype=bool)
        np.savez_compressed(args.output_dir / f"{index}-masks.npz", masks=masks)
        records.append(
            {
                "input": row,
                "proposal_count": len(ids),
                "elapsed_s": time.monotonic() - start,
                "backend": "yoloe_l",
                "confidence_threshold": 0.05,
                "scores": np.asarray(metadata.get("instance_scores", [])).tolist(),
            }
        )
        (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2))
        print(f"{index}: {len(ids)} proposals for {row['query']}", flush=True)


if __name__ == "__main__":
    main()
