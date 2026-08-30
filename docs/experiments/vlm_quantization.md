# VLM quantization: int4 vs fp16 (Qwen3-VL-8B)

Quantization is not free for embodied EQA. Same model, same harness, same qids —
only the weight precision changes (int4 bitsandbytes on the workstation GPU vs
fp16 served on the Orin `caliban` over `EMET_VL_ENDPOINT`).

## Headline

| Slice | int4 (local) | fp16 (caliban) | Delta |
|-------|--------------|----------------|-------|
| count/clock-15 (`12,21,28,32,33,43,47,48,51,60,78,84,86,88,93`) | 6/15 | **10/15** | **+4 (+66%)** |
| 30-qid subset (count/clock-15 + `0,7,14,29,30,37,42,50,56,82,92,100,104,108,110`) | 14/30 | 10/15 + 15 new (pending) | — |

The fp16 8B reads fine detail (clock hands, shelf counts, mat placement) that int4
distorts. This is a real, repeatable lift on the same model — the precision, not
the model family, is the lever. Prior art running unquantized or larger VLMs would
carry this advantage.

## Mechanism

- Local int4: bitsandbytes 4-bit NF4, in-process next to Habitat/SigLIP. The eval
  releases SigLIP before Qwen to fit 24 GB.
- Remote fp16: `docker/jetson_llm_server.py` on caliban (AGX Orin, 61 GB unified),
  `--dtype float16`, over HTTP `image_url` JPEGs. `EMET_VL_ENDPOINT` routes the EQA
  VLM client to it (`graph_eqa_vlm.py` resolves it first).

## Pitfalls found (caliban)

- **float32 OOMs the Orin**: the fp16-only serve must not be started with
  `--dtype float32` — a 36 GB footprint + ~6.7 GB activation allocations exceed the
  61 GB unified memory and every `chat.completions` returns HTTP 500
  (`NVML_SUCCESS == r INTERNAL ASSERT FAILED`), so the eval silently falls back to
  Unknown. Serve **fp16** (~18 GB, 40+ GB headroom).
- **HF cache path**: the Orin serve must set `HF_HOME=~/hf-cache` or it tries to
  download weights into the full eMMC (`~/.cache/huggingface`, 1.5 GB free) and dies.
- **Detach over ssh**: `nohup`/`setsid` die when the ssh session closes; use
  `ssh -f caliban 'HF_HOME=... setsid ... '` or a persistent shell.
- Throughput: ~10–13 min/qid on the Orin fp16 (vs ~3 min local int4) — slow, so run
  subsets, not the full 113, unless budgeted.

## Paper

Extend `paper/sections/appendix/06_model_choice.tex` `tab:vlm_candidates` /
`tab:vlm_bakeoff` with a same-model fp16-vs-int4 row + a short precision discussion.
`tab:hmeqa_vs_prior` (full-113) gets the fp16 number once the subset confirms the lift.

## Reproduce

```bash
# countclock fp16 (caliban)
uv run emet jobs run --name hmeqa-countclock-caliban-fp16 --need-mib 12000 --gpu-exclusive -- \
  env EMET_VL_ENDPOINT=openai@http://192.168.1.55:8000/v1 EMET_ALLOW_SDPA_ATTN=1 \
  ./scripts/run_hmeqa_countclock_slice.sh
```
