#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Replay the live memory-caption prompt on a saved RGB array, without simulation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from emet.core.parameters import get_parameters
from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="Raw RGB HWC uint8 NPY saved by voxel memory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rgb = np.load(args.rgb, allow_pickle=False)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected HWC uint8 RGB")
    image = Image.fromarray(rgb)
    image.save(args.output_dir / "input.png")
    prompt = "List representative objects in the image (excluding floor and wall) Limit your answer in 10 words. E.G.: a table,chairs,doors"
    record = {"source": str(args.rgb), "shape": list(rgb.shape), "prompt": prompt, "max_new_tokens": 32}
    (args.output_dir / "input.json").write_text(json.dumps(record, indent=2))
    client, _ = build_graph_eqa_vlm_clients(parameters=get_parameters("dynav_config.yaml"))
    start = time.monotonic()
    response = client([image, prompt], system_prompt="", max_new_tokens=32)
    record.update(response=response, elapsed_s=time.monotonic() - start)
    (args.output_dir / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
