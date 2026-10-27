# Integration check checkpoint, 2026-10-27

The integration gate is not passing. Do not merge the remaining stack from
these results or treat an incomplete run as an incorrect model answer.

## Completed focused runs

- `integration-task-view-molmo-20261026`: bowl localized; microwave not
  localized. Floor recovery reaches zero unknown footprint cells and restores
  a measured, fresh task view. Physical movement resumes.
- `integration-range-feedback-molmo-20261026`, revision `a09d04a1`: same FIND
  outcome. Diagnostics confirm that the microwave proposal's median measured
  depth is about 3.36 m, with zero supported pixels under the 2.5 m grounding
  limit. The new feedback alone does not repair approach behavior.
- `integration-robocasa-v2-pair-20261026`: main-agent control completes with
  sugar-cube FIND failing and cabinet FIND passing. Candidate object search
  fails after six rounds; the episode reaches the 1500 s process cutoff during
  cabinet search and produces no final metrics. Candidate is incomplete, not
  a scored 0/2. Saved current-view RGB shows a brick wall, consistent with its
  visual assessments. Both arms use the shared corrected task protocol.

## Full serial battery

Managed job `20261027_000658_f9fd9e`, name `nonrl-stack-checks-20261027`.
Artifacts: `~/runs/emet/nonrl-harness-integration-r2-20261027/`.
Frozen candidate `a09d04a1`, main-agent plus evaluation-protocol control
`a2be9d40`. This control is not unmodified main.

The battery independently runs contract tests, paired EQA-12, S0/RoboCasa/Molmo
FIND, floor-enabled S0, scripted physical TAMP, and live public-tool TAMP.
No automatic merge. Metadata adapter tests are not EXPRESS/AbstainEQA runtime
qualification; privileged stationary TAMP is not observed household manipulation.

Initial contract run: 975 passed, one failure, zero errors/skips (976 total).
The failure is an obsolete CHAT dispatch test expecting a 360-degree clamp,
while the reviewed shared motion API clamps to 180 degrees. The integration
branch updates that assertion; all 36 dispatch/motion contract checks pass.
The active frozen evaluation checkout is unchanged. Its original contract
result remains red; a fresh full contract run on the repaired revision is
required before claiming a green contract gate.

The fresh contract run at `835028dc` passes all 976 tests, with zero failures,
errors, or skips. JUnit is retained in
`.worktrees/validation-835028dc/contracts.xml`. This clears the unit/contract
slice on that revision; it does not replace the live room gates or change the
original frozen battery's red contract artifact.

At this checkpoint EQA is running; later phases remain pending. Read the job's
`progress.json`, retained logs, and final `gate.json` before reporting results.

## Completed battery and follow-up

The frozen battery finished: EQA main 8/12, candidate 9/12; ordinary S0 passes
both arms; scripted and live privileged stationary TAMP pass both arms.
RoboCasa candidate is incomplete, main fails object FIND and passes cabinet
FIND. Both Molmo arms fail runtime discovery (missing wrapper), so they are
infrastructure failures, not a fresh performance comparison. Floor-enabled S0
fails receptacle FIND. The original contract artifact retains its one obsolete
assertion failure; the separately repaired 976-test contract run above is green.
The aggregate gate is red.

Context-only follow-up at `3916b708`: RoboCasa completes, fails sugar-cube FIND,
and passes cabinet FIND. S0 object FIND passes but receptacle FIND fails. These
results do not establish a robust improvement or non-regression. At user
request, the 2.8 cm sugar-cube episode is now optional stress coverage; historical
scores remain unchanged and RoboCasa basic acceptance needs a replacement.

The full-battery driver now requires and preflights an explicit Molmo interpreter,
records it, and supplies it to both arms before any long run. Next use paired S0
and Molmo checks on the current frozen integration source to separate persistent
viewpoint/approach problems from runtime setup. No full sweep or merge is justified
yet. Observed manipulation, nonprivileged multi-step TAMP, real-robot floor safety,
Habitat OVMM, and EXPRESS/AbstainEQA runtime evaluation remain unqualified.
