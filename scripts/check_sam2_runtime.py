# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Construct SAM2 and run box inference; import-only checks miss lazy dependencies.

Run inside the target environment and the same GPU-exclusive job as evaluation.
This is a runtime contract check on synthetic pixels, not a perception benchmark.
"""

import importlib.metadata
import json

import numpy as np
import torch

from emet.perception.detection.sam2 import SAM2Perception


def main():
    model = SAM2Perception(configuration="s")
    rgb = np.full((128, 128, 3), 200, dtype=np.uint8)
    rgb[32:96, 32:96] = [220, 20, 20]
    masks = model.segment(rgb, np.array([[32, 32, 96, 96]], dtype=np.float32))
    if masks.shape != (1, 128, 128) or masks.dtype != np.bool_:
        raise RuntimeError(f"Invalid SAM2 mask contract: {masks.shape}, {masks.dtype}")
    print(
        json.dumps(
            {
                "runtime_check": "passed",
                "torch": torch.__version__,
                "iopath": importlib.metadata.version("iopath"),
                "mask_pixels": int(masks.sum()),
                "semantic_accuracy_tested": False,
            }
        )
    )


if __name__ == "__main__":
    main()
