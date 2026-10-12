# Assessed-view and context handoff pilot

Branch: `fix/eqa-context-evidence`, based on #184 at `48b56aeb`.

## Main integration — 2026-10-12

Merged main `9dd46c9c` (placement/TAMP #179, spawn-state #200,
articulation #201) into this branch. Conflict resolution preserves both
articulation access in scene-task metadata and model-derived measured head
capabilities. No evidence-selection defaults changed.

Validation: 120 focused tests passed on the merged #199 source; 173 passed on
the dependent #202 branch, including view selection, EQA evidence/answering,
OVMM routing, navigation/floor recovery, head arrival/capabilities, scene-task
extraction and articulation. These are contract tests, not new task scores.

Next gates, in order:

1. Re-run the frozen four-question EQA diagnostic on the updated source; retain
   exact prompts/images and distinguish infrastructure failures from wrong answers.
2. Integrate #202 into the shared captured-view path, preserving acquisition
   timestamps. The current store has no timestamp field; do not substitute
   retention time and present it as acquisition time.
3. Capture actual arrival-before-turn evidence at the navigation boundary.
   The current final yaw is part of the trajectory; a pre-approach image cannot
   be relabeled as an arrival/pre-turn image. Preserve clearance checks.
4. Compare selection on fixed histories, then paired EQA and small OVMM.
   Keep historical context out of current-view manipulation grounding.

Keep #199 and #202 draft until their respective evidence gates are met.

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

## Completed pilot — 2026-10-12

Frozen candidate `189fa8ef`, job `20261012_152553_12264b`, output
`/home/cpaxton/runs/emet/context-eqa4-20261012`: **1/4**, identical per-question
outcomes to parent #184 (q12/q14/q56 wrong, q6 correct). No infrastructure errors.
The 247 focused tests passed. The known missing-RGB failure was reproduced on
the exact parent separately; it is not introduced by this change.

Manually inspected the final image packets and assessment prompts:

- q12: assessment and final answering both receive bedside-table arrival RGB
  plus the earlier doorway view. Qwen reports one visible table as the total.
- q14: earlier context clearly includes the sofa/window and fan, but Qwen still
  selects "next to the TV" with contradictory reasoning. Evidence handoff works;
  this instance is not fixed by adding the context image.
- q56: earlier context is a hallway view without curtains. Qwen again infers
  white curtains from white walls. This frame cannot restore the blue curtains
  visible from the old arrival heading.

Keep draft. This supports the handoff contract, **not** an accuracy improvement
or selection of pre-approach context as the final strategy. The next targeted
diagnostic should compare actual pre-turn/post-turn arrival views, rather than
assuming the last retained frame has the missing context. No larger sweep yet.

The pilot exporter labels images without graph IDs `nav_fallback` even when
the explicit packet supplied them. Prompt text and saved pixels identify the
actual inputs. A subsequent metadata-only repair adds explicit `assessed_view`
and `pre_approach_context` roles; original pilot artifacts are not rewritten.
