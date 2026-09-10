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
| Whole-visible-object box prompt | 7 pure surfaces vs baseline 5 on 21 visible targets | Candidate for local sweep, not proven winner |
| Context-only final selection | Rejects known mug error, retains five pure detector selections | Pair with isolated control on expanded cache |
| YOLOE/other cheap proposals | Better bowl masks; misses targets and proposes distractors | Optional search/proposal source, never final authority |
| Box expansion | 25% padding worsened contamination | Keep as recorded ablation, disabled |
| Box self-check / one repair | Did not improve the first pilot | Disabled; not independent verification |
| Batch target-blind identity -> text match | Misbinds background support; 0 pure / 10 impure on RGB-D candidates | Do not promote; consider one-candidate-at-a-time binding separately |
| Promptable segmentation | Not tested | Compare cleaner support without granting identity authority |
| Multi-view/reacquisition | Not tested in this new verification battery | Test after offline acceptance; retain pose/time linkage |
| Stronger model with less assistance | Untested hypothesis above | Minimal/assisted/closed-loop paired comparison |

Findings: [box/proposal comparison](grounding_verification_ablation.md),
[context/blind comparison](candidate_context_verification.md), and
[FP16 comparison](caliban_fp16_grounding.md). The FP16 results do not isolate
quantization causality. Existing EQA smoke success is not OVMM or TAMP acceptance.

## Separate best-local candidate

`configs/eval/grounding_best_local.yaml` fixes local Qwen3-VL-8B int4, 512-pixel
maximum input side, whole-object prompting, and context-only verification.
This combines promising components whose combination has not previously been
tested. "Best local" names the experimental candidate, not a validated optimum.
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
