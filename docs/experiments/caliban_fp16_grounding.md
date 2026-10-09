# Caliban FP16 grounding comparison

Purpose: test whether the observed recognition/localization mismatch disappears
at higher precision, without changing prompts or the proposal algorithm.
This is a component replay, not an OVMM success or pure quantization ablation.

Client source: `204526d7`. Managed job: `20260909_191610_4c0d41`.
Remote endpoint: `http://192.168.1.55:8000/v1` (the SSH alias `caliban` points
there; hostname HTTP resolution was not reliable). Existing service, PID 51480,
is configured for `Qwen/Qwen3-VL-8B-Instruct`, CUDA, `--dtype float16`, without
bitsandbytes quantization. No service restart or remote source edits were made.
Caliban is a 64 GB Orin with unified memory, not another RTX workstation.

Remote source is an existing dirty v4 checkout at `d07af78ac75d844ebcede886eada7e3b3b231828`.
Its server file SHA256 is
`a3a20dc0d560176d574284e7a3cce94df018a373164b0abfa32ce4e3966c3ad1`.
The server uses greedy generation, its processor chat template, and FP16 model
loading. Unlike local inference, it directly invokes the processor on images
rather than passing through qwen-vl-utils; hardware and runtime versions also
differ. Do not attribute score differences solely to precision.

Both hosts cache checkpoint snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
Local runtime: torch 2.10.0, transformers 5.10.2, bitsandbytes 0.49.2.
Caliban: torch 2.13.0+cu130, transformers 5.16.1. Both have qwen-vl-utils
0.0.14 installed, though the server's processor path differs as noted above.

The remote client normally sends JPEG. This replay opts into PNG through
`--remote-image-format png`, avoiding an additional compression confound while
preserving the same client-side resize policy. Tests verify exact PNG pixel
round trips. Normal remote JPEG defaults and production memory presets stay
unchanged. `eqa.vl_image_format: png` exposes the same explicit option for users.

## Inputs and artifacts

- Two queries on the saved integrated failure frame: red cylinder and blue cube.
  Manifest `/tmp/emet-fp16-live-grounding.yaml`; images and arrays originate in
  `/home/cpaxton/runs/emet/shared-query-wired-vlm-20260909/grounding`.
- The same 15-query, five-layout stress manifest from
  `/home/cpaxton/runs/emet/surface-stress-paired-20260909/manifest.yaml`.
- Outputs: `/home/cpaxton/runs/emet/caliban-fp16-live-20260909` and
  `/home/cpaxton/runs/emet/caliban-fp16-stress-20260909`.
- All requests run serially. Original masks/geometry IDs are used only after
  inference to score target purity, recall and center error.

```bash
EMET_VL_ENDPOINT=openai@http://192.168.1.55:8000/v1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/audit_vlm_regions.py --manifest MANIFEST \
  --output-dir NEW_DIRECTORY --strategy depth_candidates --remote-image-format png
```

## Initial live-frame outcome

Both FP16 queries still miss localization and reject their proposed surfaces.
The red-cylinder box is `[487,850,542,925]`, again on table rather than cylinder;
the FP16 selector correctly rejects that table patch. Thus this known failure
does not require int4 quantization. This does not establish that quantization
has no effect elsewhere or that the issue is unique to Qwen.

Qwen's official [grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)
uses relative 0–1000 coordinates, matching our requested convention. Its
[image-tool cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/think_with_images.ipynb)
also demonstrates zoom tools. Those are supported approaches, not evidence
that tools are required for these particular images. We have not yet compared
another VLM family or isolated official prompting, resolution, and tool use.

## Completed results

| Frozen diagnostic | Local int4, isolated panels | Caliban FP16, isolated panels |
| --- | --- | --- |
| Stress target gate (purity >=95%, center error <=15 cm) | 9/10 | 10/10 |
| Absent-query abstention | 5/5 | 5/5 |
| Saved live red-cylinder query | rejected, box misses target | rejected, box misses target |
| Saved live blue-cube query | not measured in matched local replay | rejected, box misses target |

The int4 comparison is the fresh-box isolated-panel replay, not the older
dimmed-panel variant or the replay with boxes fixed to an older run. All ten
FP16 selected stress masks have 100% target purity. Visible-mask recall ranges
from 12.7% to 75.9%; these are patches, not full-object masks or grasp poses.
Maximum center error is 11.34 cm. FP16 recovers the farther cylinder through a
104-pixel target patch (12.7% visible recall), despite an imperfect search box.

Conclusion: the FP16 setup is promising on this bounded stress battery, but one
additional recovery does not establish a statistically reliable precision
advantage. The recognition/box mismatch persists on the live view at FP16;
int4 is not necessary for that failure. This neither proves a Qwen-specific
limitation nor proves tools are mandatory. Next isolate official grounding
prompting and image/crop resolution, then compare independent proposals and/or
another VLM family on held-out cases. No task-level success is claimed.

The managed job completed all 17 queries in about four minutes. Remote-client
and factory/image tests: 14 passed, including exact PNG round-trip validation.
The existing Caliban service was left running, unmodified; no evaluation jobs
remain running from this comparison.
