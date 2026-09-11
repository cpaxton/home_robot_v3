# Shared grounding: bounded cross-task pilot

This freezes the September 10 candidate, not a new default. The earlier verified
red/blue table finds are integration evidence, not OVMM or manipulation success.

Launch: managed job `20260910_213038_d86e1a`, frozen source `b194395a` at
`/tmp/emet-cross-task-20260910-frozen`. Artifacts:
`/home/cpaxton/runs/emet/shared-grounding-cross-task-20260910`.
Results are pending; `process_status.tsv` records exits, not task acceptance.

## Fixed comparison

- Hybrid: `query_detector_segmented_pilot.yaml`, YOLOE boxes → SAM2.
- Reference: `query_segmented_support_pilot.yaml`, Qwen boxes → SAM2.
- Both: support-only Qwen3-VL-8B-Instruct int4 verification, lazy query memory,
  unchanged fusion and task budgets. This tests proposal strategies, not SAM2
  versus YOLOE segmentation, and not lazy memory versus DynaMem.
- Habitat OVMM: nearest-v2 scenes 00006 and 00025, seed 0, 12 rounds/8 nav steps.
- EQA: questions 15, 16, 25, 20 planning/10 movement steps, no semantic sensor
  labels or enriched GT hints. The CLI does not pin EQA RNG seeds: these are
  matched questions/settings, not a deterministic causal no-regression test.
- Hybrid-only Stretch table: one absent-object find and one learned pick/place.
  Velocity navigation; no teleport or oracle manipulation. This is a multistep
  integration diagnostic, **not the PR #160 TAMP battery**. The oracle TAMP
  control previously passed; learned TAMP acceptance remains outstanding.

All twelve processes run serially. Stop after a timeout to inspect cleanup.
Each has a wall cap, not an extended task budget. The runner records process
completion separately from benchmark correctness. Report abstentions, grounding
errors, navigation failures and manipulation failures separately; do not turn
tool `ok` or exit zero into physical success.

## Reproduction

From a frozen checkout with the SAM2 small, YOLOE-L and MobileCLIP checkpoints
available at the usual loader paths:

```bash
OUT=/absolute/new-artifact-directory \
HABITAT_BIN=/absolute/habitat-env/bin/emet-habitat \
AGENT_PY=/absolute/agent-env/bin/python \
SAM2_SOURCE=/absolute/pinned/segment-anything-2 \
emet jobs run --name shared-grounding-pilot --need-mib 16000 \
  --cpu-safe --gpu-exclusive -- bash scripts/run_shared_grounding_pilot.sh
```

`EMET_CONFIG` already feeds the unified preset to both Habitat runners through
`get_parameters`; no new benchmark-only config API is needed. Preflight verified
that the EQA and OVMM profiles retain the selected query options and model.
SAM2 source is pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4` for this run.
Using it through `PYTHONPATH` avoids changing the shared Habitat installation.
Habitat's installed executable sets its required C++ library path.

Review the saved grounding RGB/masks/poses, agentic traces, navigation receipts,
and EQA map/video bundles. Inspect final accepted support manually, especially
any positive absent-query response. A new contamination or wrong-object
acceptance blocks promotion even if aggregate correctness improves. Retain
zero-result failures and infrastructure exclusions. No full sweep is authorized
by this pilot; use its failures to select the next shared-mechanism fix.
