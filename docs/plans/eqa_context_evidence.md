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

## Consolidated validation and review — 2026-10-12

Frozen `0fb37763`, job `20261012_182026_28c8b8`, artifact root
`/home/cpaxton/runs/emet/context-main-eqa4-20261012`: completed without episode
errors. Predictions match the pre-main-update pilot exactly:

| Question | Prediction | Gold | Correct |
| --- | --- | --- | --- |
| 12 | B | D | No |
| 14 | C | D | No |
| 56 | A | C | No |
| 6 | B | B | Yes |

This is **1/4 vs 1/4**, not restored EQA acceptance. q14's saved context visibly
contains the fan and sofa. Its final Qwen response says "In the living room",
but the confirmed earlier assessment says "Next to the TV" and wins the existing
answer precedence. Record this as model-call disagreement plus a policy choice,
not exclusively failure to see the object. Do not flip precedence to fit this
one gold answer: evaluate disagreements on a frozen set separately from view
selection. q56's context remains a hallway without the needed curtains; its
final response infers curtain color from wall color. q12 still equates one
visible table with the room total. Exported image roles now correctly identify
assessed-view and pre-approach context.

Main subsequently advanced to `b7a1c22d` (#195/#197). Both stack branches include
that update. The combined contract suite passed **241 tests** before the review
repairs below. The live result above belongs only to `0fb37763`, not these later
changes.

### Review disposition (review of the whole #202 ancestry)

The selector-only delta is #202 against #199; the broad diff against main also
contains the provider/navigation/inspection ancestors. Repairs are isolated
commits on #199 and inherited by #202, not new selector behavior.

| Finding | Disposition |
| --- | --- |
| 1: PIL input crashes SAM2 box segmentation | Fixed in `ef62fdc8`; NumPy/PIL empty, invalid and valid box cases tested. |
| 2: First sweep rejection loses diagnostics | Fixed in `4eb29acc`; removed duplicate validation, retained first failure, reset stale evidence. |
| 3: Invalid depth treated as obstruction | Fixed in `b99b05dc`; only measured near rays count, with mixed/all-invalid regression tests. Unknown depth is still not evidence of navigable space. |
| 4: Fallback clearance reported unknown | Intentional: the 10 m planning sentinel is not measured clearance. Existing obstacle-free test plus a new solver-failure test enforce this distinction. |
| 5: Exploration policy on precision waypoint | Open: audit callers and test final tolerances/stall behavior before changing policy. |
| 6: Anchor-less physical investigate | Open: separate optional gaze geometry from valid current-view verification; needs missing-anchor regression and hardware/sim contract test. |
| 7–8: Route rejection / no corner cutting | Conservative safety behavior; existing corner tests cover blocked diagonals and alternate routes. Partial-route caller compatibility remains to be audited. Do not loosen to improve scores. |
| 9–10: Wheel contact / head capability | Exact-model drive smoke and actuator-capability check remain required; unit tests do not establish physical acceptance. |
| 11: Legacy clamped cell conversion | Open: test edge/out-of-map camera rays in both map modes before accepting coverage scores. |
| 12 and minor cleanup | Nonblocking follow-up: exception text, diff reflow, response retention bounds, dead controller branches, nonfinite telemetry. |

The four focused review files passed **58 tests**, including SAM2 (no skips).
No live run after these repairs is claimed. Next: clear missing-anchor and
precision-policy contracts, then a serialized small OVMM smoke; wire timestamped
view selection separately. Keep all historical results tied to their source SHA.

Combined-suite follow-up: collecting the SAM2 tests alongside navigation tests
exposed eight controller-initialization failures (291 passed): SAM2 initializes
Hydra globally, while the control loader tried to initialize it again. Repair
`e6d3c032` loads the existing standalone control YAML with OmegaConf without
touching global Hydra state; unsupported Hydra defaults composition fails
explicitly. The targeted SAM2/controller/config suite passes 12 tests. This was
an actual shared-process integration problem, not a reason to isolate the tests
or clear another component's global state.

Final combined rerun at #202 `7611a900`: **300 passed**, no skips, two SWIG
deprecation warnings (23.79 seconds, one CPU core). This covers the same 241
contracts plus the 58 review checks and config-isolation regression, in one
process. It does not replace the outstanding live motion/OVMM gates.

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
