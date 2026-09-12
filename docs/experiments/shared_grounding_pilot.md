# Shared grounding: bounded cross-task pilot

Interpretation: [environment acceptance progression](../environments/README.md),
[Habitat search scope](../environments/habitat.md),
[simple-sim controls](../environments/simple_sim.md).

This freezes the September 10 candidate, not a new default. The earlier verified
red/blue table finds are integration evidence, not OVMM or manipulation success.

Launch: managed job `20260910_213038_d86e1a`, frozen source `b194395a` at
`/tmp/emet-cross-task-20260910-frozen`. Artifacts:
`/home/cpaxton/runs/emet/shared-grounding-cross-task-20260910`.
This first launch is invalid for comparing strategies: all four OVMM cases and
six EQA cases encountered SAM2's missing `iopath` dependency during model
construction. OVMM serialized the exception despite exit zero. The absent-object
find then reached its 360-second cap while exploring; learned pick/place never
started. Retain these artifacts as infrastructure failures and a search timeout,
not a model-quality score.

September 11 retry: `20260911_185541_56a4c4`, frozen `eed1d868`, artifacts
`/home/cpaxton/runs/emet/shared-grounding-habitat-retry-20260911`. Installed
`iopath==0.1.10` and `portalocker==3.2.0` in Habitat, without changing Torch or
model weights. The runner now constructs SAM2 and executes synthetic box
inference before episodes, and supports `PHASE=habitat|sim|all` (default all).
This retry selects Habitat only, with the original cases, budgets and presets.
`process_status.tsv` records exits, not task acceptance. All ten retry processes
completed without infrastructure exceptions:

| Strategy | EQA q15 / q16 / q25 | OVMM object + receptacle, two scenes |
| --- | --- | --- |
| Hybrid | correct / correct / wrong (2/3) | 0/4 |
| Qwen-box | wrong / correct / correct (2/3) | 0/4 |

These three questions do not establish no regression against the earlier 3/3
smoke. The strategies disagree on two questions despite equal aggregate scores.
Hybrid scene 00025 returned an object localization 3.615 m from the target;
Qwen-box returned a receptacle localization 5.184 m from its evaluator target.
Neither is a task success. Other phases returned no localization.

Manual review of hybrid scene 00025's accepted support
`grounding-965b2a585dd64549851ab3ce1af623dd-surface-0.png` shows bedding/fabric,
not an identifiable lamp. The proposal query is `nearest bed`, while the support
selector receives the full question about the lamp nearest a bed. Qwen accepts
the bedding while claiming a small object on it resembles a lamp. This is a
concrete target/anchor confusion and false semantic acceptance; metric distance
alone would not reveal it. Preserve identity versus relationship as separate
verification concerns in the next fix, without handing context pixels back the
authority to validate an incorrect mask.

The pending learned Stretch pick/place diagnostic was launched separately as
`20260911_194952_1415c5` on the same `eed1d868`, with a 600-second cap, velocity
navigation, hybrid preset and no oracle scene plans. It does not rerun the known
absent-query timeout first. The tool failed after 195.9 seconds with
`Command 101 failed: terminal command outcome is immutable`; the agent reported
the failure rather than claiming success. No successful learned pick/place is
established. The command lifecycle exception needs diagnosis; process completion
is not physical task acceptance.

Lifecycle follow-up `efba8d7f`: reproduced the exception when a goal had already
failed with stop unconfirmed, then a later cancellation confirmed stopping and
attempted to rewrite the terminal status to cancelled. The fix retains the failed
outcome/reason and records later stop confirmation separately, releasing motion
ownership only on confirmed stop. Both core/deploy copies match; 37 focused
command, trajectory and adapter tests pass. Same-preset bounded sim retry:
`20260911_200332_5b6840`, artifacts `~/runs/emet/lifecycle-manip-retry/evidence`.
The retry reached manipulation without the lifecycle exception, but failed
pregrasp with arm extension -0.180 m and subsequently lost current-frame target
support. No pick/place success. This run did not necessarily exercise stop recovery.

Manipulation handoff follow-up `40288111`: the verified find pose faced the
object with the camera, while Stretch's side-grasp expects the target along -Y.
The grasp adapter now owns a measured target-facing side rotation and fresh
reacquisition; find remains unchanged. Failed/nonfinite pregrasp IK now returns
failure instead of falling through into visual servo. 23 focused tests pass.
Bounded same-preset sim check: `20260911_202359_853d5b`, artifacts
`~/runs/emet/grasp-handoff-retry/evidence`. The retry completed orientation and
reacquisition, then reached pregrasp (IK arm -0.039 m, clamped by the legacy 5 cm
allowance). Wrist tracking rejected absent/ambiguous current-frame support;
pickup failed and place was skipped. This is not manipulation acceptance.

`3b2ce750` additionally propagates arm-motion failure rather than treating valid
IK as execution success (25 focused tests pass). `a7c45096` retains wrist RGB,
depth, calibration and grounded world points on tracking rejection under the
existing episode evidence opt-in. Serial diagnostic `20260911_202818_77c40c`
uses that frozen source; artifacts `~/runs/emet/wrist-tracking-audit/evidence`.
No tracking-tolerance relaxation or oracle fallback was introduced.

The wrist audit completed with failed pickup, not a task success. Manual review
of `wrist_tracking/grounding-054937277d1849de892afab78332be86.png` shows the red
cylinder clearly. Its stored world median projects to wrist pixel (160.3, 216.1),
with expected camera depth 0.380 m versus observed 0.370 m. This argues against
a gross camera-frame error in this observation. Replaying the current expanded
world bounding box gives two components of 4,726 and 2,005 pixels, triggering the
intended ambiguity rejection. Bounding-box-only association is insufficient in
this close view; object-specific wrist support is needed. Neither largest-mask
fallback nor looser margins is justified by this evidence.

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
