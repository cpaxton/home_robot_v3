# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Smoke-load a VL checkpoint with mid-run VRAM sampling (AWQ / MoE / dense / factory).

Usage:
  uv run python scripts/smoke_vl_peak.py <hf_model_id> [quant] [out_dir]
  uv run python scripts/smoke_vl_peak.py --family internvl <hf_model_id> [quant] [out_dir]

  quant: none | int4 | int8   (default none for pre-quantized AWQ; int4 for dense/MoE bnb)
  --family: use emet create_dynamem_vllm (internvl, qwen3_vl, …)

Writes OUT/vram_trace.csv, OUT/peak_vram.txt, OUT/smoke.log and prints PEAK_VRAM.
Abort Habitat if peak leaves < ~8 GiB free on a 24 GB card.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _sample_loop(path: Path, stop: threading.Event, interval_s: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "used_mib", "free_mib"])
        t0 = time.time()
        while not stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info(0)
                used = (total - free) // (1024 * 1024)
                free_mib = free // (1024 * 1024)
            except Exception:
                used, free_mib = -1, -1
            w.writerow([int(time.time() - t0), int(used), int(free_mib)])
            f.flush()
            stop.wait(interval_s)


def _write_peak(trace: Path, out: Path) -> tuple[int, int]:
    rows = list(csv.DictReader(trace.open()))
    if not rows:
        raise SystemExit("empty vram_trace.csv")
    peak = max(rows, key=lambda r: int(r["used_mib"]))
    used = int(peak["used_mib"])
    free = int(peak["free_mib"])
    out.write_text(f"used_mib={used}\nfree_mib={free}\nt_sec={peak['t_sec']}\nsamples={len(rows)}\n")
    print(
        f"PEAK_VRAM used={used} MiB free={free} MiB at t={peak['t_sec']}s n={len(rows)}",
        flush=True,
    )
    return used, free


def _parse_args(argv: list[str]) -> tuple[str | None, str, str, Path]:
    family: str | None = None
    args = list(argv)
    if args and args[0] == "--family":
        if len(args) < 3:
            raise SystemExit("usage error")
        family = args[1]
        args = args[2:]
    if not args:
        raise SystemExit("usage error")
    hf_id = args[0]
    quant = (args[1] if len(args) > 1 else "none").lower()
    out = Path(args[2] if len(args) > 2 else os.environ.get("OUT", ".")).expanduser()
    return family, hf_id, quant, out


def main() -> int:
    try:
        family, hf_id, quant, out = _parse_args(sys.argv[1:])
    except SystemExit:
        print(__doc__, file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "smoke.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    stop = threading.Event()
    sampler = threading.Thread(target=_sample_loop, args=(out / "vram_trace.csv", stop), daemon=True)
    sampler.start()
    t0 = time.time()
    model = None
    try:
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        if family:
            from emet.llms.vllm_factory import create_dynamem_vllm

            q = None if quant in ("none", "", "awq") else quant
            log(f"Loading family={family} id={hf_id} quant={q!r} via create_dynamem_vllm")
            client = create_dynamem_vllm(
                family,
                hf_model_id=hf_id,
                vl_model_size="14B",
                max_tokens=32,
                device="cuda",
                quantization=q,
            )
            model = getattr(client, "model", None)
            reply = client.generate_multimodal(
                ["What color is this image? Answer in one word.", img],
                system_prompt=None,
                max_new_tokens=8,
                reset_context=True,
            )
            log(f"  load_s={time.time() - t0:.1f}")
            log(f"SMOKE OK family={family} id={hf_id} quant={quant} reply={reply!r}")
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            log(f"Loading {hf_id} quant={quant!r} via AutoModelForImageTextToText")
            processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            kw: dict = {
                "trust_remote_code": True,
                "device_map": {"": 0},
                "attn_implementation": "flash_attention_2",
            }
            if quant in ("int4", "int8"):
                from transformers import BitsAndBytesConfig

                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(quant == "int4"),
                    load_in_8bit=(quant == "int8"),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif quant in ("none", "", "awq"):
                kw["dtype"] = "auto"
            else:
                raise SystemExit(f"unknown quant={quant!r} (use none|int4|int8)")

            model = AutoModelForImageTextToText.from_pretrained(hf_id, **kw)
            log(f"  load_s={time.time() - t0:.1f}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {
                            "type": "text",
                            "text": "What color is this image? Answer in one word.",
                        },
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                out_ids = model.generate(**inputs, max_new_tokens=8)
            trim = out_ids[:, inputs["input_ids"].shape[-1] :]
            reply = processor.batch_decode(trim, skip_special_tokens=True)[0].strip()
            log(f"SMOKE OK id={hf_id} quant={quant} reply={reply!r}")
    except Exception as e:
        log(f"SMOKE FAIL id={hf_id} quant={quant}: {type(e).__name__}: {e}")
        stop.set()
        sampler.join(timeout=5)
        if (out / "vram_trace.csv").is_file():
            try:
                _write_peak(out / "vram_trace.csv", out / "peak_vram.txt")
            except SystemExit:
                pass
        raise
    finally:
        stop.set()
        sampler.join(timeout=5)
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    used, free = _write_peak(out / "vram_trace.csv", out / "peak_vram.txt")
    if free < 8000:
        log(f"HEADROOM FAIL free={free} MiB (<8000) — do not Habitat with this checkpoint")
        return 1
    log(f"HEADROOM OK free={free} MiB (peak used={used})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
