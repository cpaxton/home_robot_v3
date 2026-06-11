# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Smoke-load an EQA VL model and run one tiny multimodal generate.

Usage: python scripts/smoke_vl_model.py <family> <hf_model_id> [quantization]
"""

import sys

import numpy as np
from PIL import Image

from emet.llms.vllm_factory import create_dynamem_vllm


def main() -> int:
    family, hf_id = sys.argv[1], sys.argv[2]
    quant = sys.argv[3] if len(sys.argv) > 3 else "int4"
    client = create_dynamem_vllm(
        family,
        hf_model_id=hf_id,
        vl_model_size="3B",
        max_tokens=32,
        device="cuda",
        quantization=quant,
    )
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    out = client.generate_multimodal(
        ["What color is this image? Answer in one word.", img],
        system_prompt=None,
        max_new_tokens=8,
        reset_context=True,
    )
    print(f"SMOKE OK family={family} id={hf_id} quant={quant} reply={out!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
