#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
"""Minimal repro for post-'Qwen3-VL ready' hang during answer-only EQA.

Usage:
  EMET_AGENT_MODEL_DEBUG=1 timeout 600 uv run python scripts/debug_eqa_vlm_hang.py
  EMET_AGENT_MODEL_DEBUG=1 timeout 600 uv run python scripts/debug_eqa_vlm_hang.py --with-image --eqa-prompt
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-image", action="store_true", help="Include a synthetic RGB image")
    parser.add_argument("--eqa-prompt", action="store_true", help="Use full EQA system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-images", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("EMET_AGENT_MODEL_DEBUG", "1")

    import numpy as np
    from PIL import Image

    from emet.core.parameters import get_parameters
    from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients, release_shared_graph_eqa_vlm
    from emet.llms.prompts.eqa_prompt import EQA_PROMPT

    params = get_parameters("configs/emet/default.yaml")
    eqa = dict(params.get("eqa") or {})
    _log(
        f"config eqa.vl_family={eqa.get('vl_family')!r} "
        f"hf={eqa.get('vl_hf_model_id')!r} quant={eqa.get('vl_quantization')!r} "
        f"prefix_cache={eqa.get('vl_cache_system_prefix')!r}"
    )

    release_shared_graph_eqa_vlm()
    _log("building graph_eqa VLM clients (shared load)…")
    t0 = time.monotonic()
    keyword_client, eqa_client = build_graph_eqa_vlm_clients(parameters=params, device="cuda")
    _log(f"clients ready in {time.monotonic() - t0:.1f}s")

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            _log(f"VRAM after load: free={free/1024**2:.0f}MiB total={total/1024**2:.0f}MiB")
    except Exception as e:
        _log(f"VRAM snapshot failed: {e}")

    command: list = ["Question: Where is the sink?"]
    if args.with_image:
        for i in range(max(1, args.n_images)):
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb[:, :, 0] = 40 + 20 * i
            rgb[:, :, 1] = 80
            rgb[:, :, 2] = 120
            # Fake a bright rectangle as a "sink"
            rgb[200:280, 220:420] = (200, 200, 220)
            command.append(Image.fromarray(rgb, mode="RGB"))
            _log(f"appended image {i+1}/{args.n_images} shape={rgb.shape}")

    system_prompt = EQA_PROMPT if args.eqa_prompt else "Answer briefly."
    _log(
        f"calling eqa_client (max_new_tokens={args.max_new_tokens}, "
        f"eqa_prompt={args.eqa_prompt}, n_parts={len(command)})…"
    )
    t1 = time.monotonic()
    out = eqa_client(
        command,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    _log(f"eqa_client returned in {time.monotonic() - t1:.1f}s chars={len(out or '')}")
    print("--- response ---", flush=True)
    print((out or "")[:800], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
