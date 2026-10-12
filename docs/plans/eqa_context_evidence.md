# Assessed-view and context handoff pilot

Branch: `fix/eqa-context-evidence`, based on #184 at `48b56aeb`.

## Hypothesis and scope

Object-facing navigation can remove room/relationship context. Preserve the
latest retained voxel frame before an investigation and pair it with the fresh
assessment image. This is an earlier pre-approach view, not a reconstruction of
the old heading at the arrival position. No extra motion or simulator truth is
used. Duplicate images are omitted.

The assessment prompt distinguishes current visibility (Image 1) from earlier
context. OVMM grounding still uses the current RGB-D frame, never the context
image or its camera pose. The final answer receives the same assessed packet,
including voxel-only evidence with no graph object. The configured final-image
budget remains enforced; the packet replaces diversified/fallback image
selection for this submission. Existing real graph identity is retained only
when its pixels match the primary evidence image.

Answer precedence, confidence rules, navigation, model, and motion budgets are
unchanged. This tests evidence handling, not a new answer-verification policy.

## Acceptance

Unit checks cover image order, context roles, deduplication, image limits,
voxel-only final-answer handoff, and no fabricated graph observations. Run the
adjacent assessment, answer, grounding, and OVMM routing regressions.

Compare q12/q14/q56/q6 seed0 against the frozen #184 run
`slice184-eqa4-retry-20261012` (1/4). That parent matches the three integration
losses; #181 scored 4/4 and #182 scored 0/4 on this diagnostic subset. Main/#181
include two fallback successes, not four evidence-based visual successes.
Keep Qwen3-VL-8B int4 and all budgets fixed. Inspect saved final-answer images
and prompt for both frames before interpreting accuracy. Expand to the small
EQA dev set and S0 find only if the evidence path works; no full sweep or merge
claim follows from four diagnostic questions.

Known unrelated test failure on the parent:
`test_voxel_sim_upgrades_full_frame_absent_to_present` expects PRESENT without
RGB while the assessment returns ABSENT. Do not weaken it to clear this pilot.
