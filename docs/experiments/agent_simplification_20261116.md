# Simplified find handoff and model interface

The [complexity audit](agent_complexity_review_20261116.md) identified a concrete
contract mismatch: find rejected a retained admitted object location after a
camera turn because current visual verification was false. The first repair
separates those facts, without another recovery policy or evidence store.

## Changes and checks

- `52ec697d`: the task's selected grounded object can supply its last-observed
  location without current visual confirmation. Existing query records reject
  invalidation, fresh negative evidence, released references and restored
  checkpoints. Object-label/coordinate checks exclude camera positions and
  nonfinite coordinates. Result metadata exposes location semantics and visual
  verification separately. The coordinator's answer is preserved.
- `d108002a`: model-final advertises investigate once, retaining the old
  navigate alias handler for compatibility. Find memory uses search guidance
  without count/MCQ instructions; ordinary EQA rendering remains unchanged.
- `52800c30`: the serial room driver accepts explicit generic model endpoints;
  local int4 defaults and environment isolation remain. Remote precision and
  decoding are recorded as unverified. No model-specific navigation branch.

The controller still calls `prepare_query_target` to reacquire fresh object
geometry before manipulation. Tests verify that returning a remembered find
coordinate neither refreshes the candidate's grounding revision nor invokes
physical preparation. Missing/current-query references and mismatched object
labels cannot be replaced by unrelated graph nodes.

CPU validation: **386 pass, two simulator opt-ins skip** in
`/tmp/agent-simplification-contracts-20261116.xml`. The skipped cases require
`RUN_OVMM_FIND_TESTS=1`; their absence is not live simulation acceptance. The
suite covers EQA routing/answers/memory, OVMM handoff and metrics, query grounding
and manipulation reacquisition, compatibility dispatch and model-runner config.
Pre-commit checks pass. A larger executor rewrite is intentionally deferred.

## First live development check

Frozen source `52800c30`, managed job `20261116_014730_97090b`, existing
`molmo_rby1_ithor_s2_idx0` episode, CLI seed 1, local Qwen3-VL-8B int4, four
initial scan views, 12 rounds/eight navigation calls per phase. Physical
navigation, model-selected actions, mask admission and the 2.5 m support gate
are unchanged. Sensor depth is supplied by simulation; object poses/labels
are not supplied as agent knowledge. This remains a find-only development check.

| Result | Bowl | Microwave |
| --- | --- | --- |
| Metric find success | Pass | Fail |
| Decision rounds | 4 | 11 |
| Navigation calls charged | 1 | 3 |
| Selected grounded object | 7 | None |
| Current visual verification at submit | True | False |
| Explicit final answer | on the counter | above the stove, mounted on the wall |

The invocation completes normally; the gate returns a task failure because only
one phase passes. Query wall time is **177.30 s**, episode wall time **264.26 s**.
Compared with the earlier 0/2 development run, the bowl closes sooner, but the
whole episode is slower (previously 147.82/202.22 s). These changed, stochastic
trajectories do not establish accuracy or runtime improvement.

The bowl is admitted at rounds 0 and 2 and the model submits at round 3. Its
last-observed provenance is recorded correctly. Because current verification
also happens to be true here, this live trial does not independently exercise
the false-verification handoff branch; deterministic regressions cover that case.

For microwave, the model first inspects memory and verifies the current view,
then requests investigation of object 7 (the earlier bowl). It subsequently
selects rotations, floor observation and short forward moves. Grounding at round
8 abstains with `no unambiguous surface selected; another view is needed`.
The model eventually submits its semantic answer without geometry. The initial
microwave image visibly includes the appliance above the stove; this is a useful
task-transition/context test rather than a case where the target was never seen.
The request clearly identifies the active microwave task and labels the nearby
historical object as bowl. Preserve this failure for cross-model replay instead
of adding a model-specific action redirect or forcing a successful location.

## Context/action audit and next test

All **15 requests** remain within eight images and 64,000 working characters
(maximum 63,620); seven earlier state texts are archived by reference at most.
An audit verifies **30** recurring assistant/tool messages remain exact and all
**15** executions belong to the coordinator's selected action lists, without
rewrites. Pixel recall uses a valid image ID. The actual system prompt advertises
**11 tools instead of 12**, and is **9,445 instead of 9,839 characters**. This is
a modest interface reduction, not evidence that prompt length caused success.

Artifacts: `~/runs/emet/simplified-find-20261116-r1/`; frozen protocol, copied
runner, test report and `simplified-find-live-audit-20261116.json` are under
`~/runs/emet/simplified-find-cohort-20261116-r1/`. Original failed rows and images
remain intact. No remote model generation was run in this check.

Next follow the [bounded model evaluation plan](../plans/2026-11-16_harness_simplification_and_model_eval.md):
eight fixed-context calls first, then targeted closed-loop comparisons, then
paired EQA/room OVMM and a minimal TAMP acceptance check. Broader task acceptance,
a current-main comparison and manipulation qualification remain open.
