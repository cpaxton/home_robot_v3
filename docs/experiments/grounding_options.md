# Grounding hypotheses and option register

One harness for EQA, find/OVMM and TAMP. Keep perception assistance optional and
separate from robot-independent evidence and execution contracts. These are
working hypotheses, not paper conclusions or a reason to change production defaults.

## Model capability hypothesis

A stronger VLM (for example a GPT-family model) might identify and localize from
the original RGB plus a concise query without our Qwen-specific crop/candidate
assistance. Fewer calls and retries might offset higher per-call cost. Neither
claim has been tested. Do not assume better language reasoning implies accurate
coordinates, segmentation, calibrated uncertainty, or safe manipulation.

Compare three assistance levels on frozen inputs, with the same scoring:

1. Minimal: RGB + query -> box/point or abstention.
2. Assisted: add context crops and measured candidate support.
3. Closed-loop: bounded correction or fresh-view acquisition.

Record exact model/version, precision, input resizing, prompts, token budgets,
latency, token usage/cost and retries. Compare wrong-object acceptance, localization,
mask purity AND retained recall, not just answer rate. If exact inference stacks
differ, report those confounds. Do not send imagery to a paid/external model as
part of the current local sweep. Stronger-model experiment wiring remains TODO.

Every model must still obey valid depth/calibration/timestamps, evidence provenance,
fresh reacquisition before action, and navigation/manipulation safety checks.
Those requirements are not optional prompting support. An EQA observation can
support an answer without qualifying as an OVMM grasp target.

## Options (maintain this list as results arrive)

| Option | Evidence/status | Next decision |
| --- | --- | --- |
| Whole-visible-object box prompt | First pilot: 7 pure vs baseline 5; expanded explicit-preset run: 9/31 visible | Keep experimental; does not meet acceptance |
| Context-only final selection | Rejects known detector mug error; no score gain on expanded Qwen-box cache | Useful rejection option, not a geometry fix |
| YOLOE/other cheap proposals | Better bowl masks; misses targets and proposes distractors | Optional search/proposal source, never final authority |
| Box expansion | 25% padding worsened contamination | Keep as recorded ablation, disabled |
| Box self-check / one repair | Did not improve the first pilot | Disabled; not independent verification |
| Batch target-blind identity -> text match | Misbinds background support; 0 pure / 10 impure on RGB-D candidates | Do not promote; consider one-candidate-at-a-time binding separately |
| Promptable segmentation | SAM2: 13 pure vs 9 RGB-D on fixed boxes; wrong-surface acceptances remain | Shared opt-in provider, not final authority; see segmented grounding report |
| Multi-view/reacquisition | Not tested in this new verification battery | Test after offline acceptance; retain pose/time linkage |
| Stronger model with less assistance | Untested hypothesis above | Minimal/assisted/closed-loop paired comparison |

Findings: [box/proposal comparison](grounding_verification_ablation.md),
[context/blind comparison](candidate_context_verification.md), and
[FP16 comparison](caliban_fp16_grounding.md). The FP16 results do not isolate
quantization causality. Existing EQA smoke success is not OVMM or TAMP acceptance.
Latest: [SAM2 comparison and shared integration](segmented_shared_grounding.md).

## Separate best-local candidate

`configs/eval/grounding_best_local.yaml` fixes local Qwen3-VL-8B int4, 512-pixel
maximum input side, whole-object prompting, and context-only verification.
This combines promising components whose combination had not previously been
tested before this sweep. "Best local" names the experimental candidate, not a validated optimum.
No production agent preset inherits it. Blind matching and expansion are disabled.

`scripts/run_grounding_ablation.py --preset configs/eval/grounding_best_local.yaml`
reuses the normal scene caches and writes the resolved configuration alongside
artifacts. It first generates proposals, then runs isolated and context verification
on the same masks with matched settings. Proposal-stage VLM selection is retained
for audit but does not gate the subsequent replay; runtime therefore includes
diagnostic overhead and must not be reported as optimized online performance.

Keep previous development, previous held-out, and supplementary close-view rows
separate. The expanded 60-row cache contains repeated objects/views, not 60
independent episodes. Do not tune against held-out outcomes. Advance to a bounded
find/OVMM pilot only if the candidate preserves correct surfaces, avoids known
wrong-object acceptance, and has defensible geometry. No full task sweep is implied.

## Expanded local sweep: completed 2026-09-10

Source `b8121694`, serial CPU-safe/GPU-exclusive job `20260910_082055_b0826f`.
Output `/home/cpaxton/runs/emet/best-local-grounding-20260910` includes the preset,
resolved parameters, 60-row manifest/truth, proposal-stage responses and masks,
and paired `verification/isolated` and `verification/context` results/scores/panels.
Local Qwen3-VL-8B int4 only; no stronger-model calls were made. Runtime about four
minutes including GPU admission and diagnostic stages, not online agent latency.

| Split | Views / visible targets | Isolated pure / impure accepted | Context pure / impure accepted |
| --- | --- | --- | --- |
| Previous development | 28 / 14 | 5 / 6 | 5 / 6 |
| Previous held-out | 12 / 7 | 0 / 0 | 0 / 0 |
| Supplementary near views | 20 / 10 | 4 / 3 | 4 / 3 |
| Total | 60 / 31 | 9 / 9 | 9 / 9 |

Both reject all 29 zero-visibility rows. All accepted masks have some target
overlap; none is the zero-overlap wrong-object failure seen in the earlier
detector experiment. The nine impure acceptances are below the unchanged 95%
purity gate, not nine wholly wrong-object selections. No acceptance gate was relaxed.

Supplementary recoveries: bottle (95.1% purity, 91.9% visible recall), two bowl
views (98.5%/89.2% and 95.2%/94.6%), and a sponge patch (100%/4.2%). The tiny
sponge patch emphasizes that a pure surface is not complete geometry or grasp
acceptance. Accepted paper towel (26.3% purity), sugar cube (53.1%), and broccoli
(82.0%) still mix substantial non-target support. The visible high-angle sponge,
turmeric and pickle slice did not produce candidates; more context cannot recover
an absent proposal. Specific food identity may also require readable labels or a
better view, not simply a more confident response.

The original whole-object pilot's 7/21 score must not be treated as a guaranteed
improvement: this run uses a newly resolved explicit preset and fresh generations.
The controlled comparison here is identical candidate masks with isolated versus
context verification, which shows no aggregate score gain. Do not attribute any
cross-run difference solely to the prompt.

Decision: retain the separate local preset for reproducibility, do not promote it
to final OVMM acceptance. Next prioritize cleaner mask/point support and candidate
coverage, with context-only as an optional identity check. Keep the stronger-model
minimal-assistance hypothesis open, while measuring geometry independently. The
shared find/OVMM/TAMP harness still needs fresh-view and execution acceptance;
this offline sweep does not establish task success. Focused tests: 35 passed.
