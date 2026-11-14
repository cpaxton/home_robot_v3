# Whole-body validation: implementation passes, the new pilot does not track

The fresh implementation gate, teacher preflight, fixed 8,192-decision pilot,
and corrected paired evaluation are complete at immutable `a29da15d`. Reject
the pilot and retain source `e7b42569…`. The independently completed jaw-unit
curriculum at `0fba6121` is also rejected. Neither candidate is hardware ready.

This report records the completed first pilot, not the latest running job.
The later [normalization comparison and its current blocker](../plans/mars_whole_body_advantage_pair.md#recorded-attempt-2026-11-14)
are tracked separately. Correct experimental code may merge with documented
performance limitations; the rejection here applies to checkpoint promotion.

## Fresh implementation and teacher checks

All **1,170** source-bound regressions pass with zero failures, errors or skips.
The receipt verifies against the development worktree as well as the immutable
runtime. These are local checks, not GitHub CI claims.

The teacher passes **164/164** general tracking cases. Actual public-tool TAMP
pregrasp passes **5/6**, with confirmed measured stops in all six episodes.
The declared `object_changed_replan` challenge remains a failed episode; both
independent audit cases pass. Experimental training admission is distinct from
all-success qualification. The goals, reference limits, frozen acceptance sets
and hard guards are unchanged.

The first full regression attempt had one setup failure: the extracted snapshot
lacked Git metadata required by the artifact/index test. Restore the actual
frozen commit/index, verify gzip ignoring and absence of tracked traces, then
rerun the entire gate. No test is skipped or weakened. Extracted compute
snapshots need this Git context before running repository-index checks.

## Fixed new-policy pilot

This is a cold-start experiment with a new full-history GRU, separate critics,
privileged critic inputs, explicit reference velocity, and a new reward. It is
not an isolated architecture ablation or a continuation of the retained actor.
Its small budget does not establish that a GRU or multiple critics are worse.
It does establish that this checkpoint cannot replace the working source.

The pilot completes **8,192 decisions / 8,795 physical control ticks**, with
4,350 general ticks and 4,445 TAMP ticks: **49.46% / 50.54%**. All updates are
finite and offline W&B records the losses, advantage correlations and counts.
The ledger contains 166 completed episodes and one additional reset at the
budget boundary. There are **0/166** completed training primitives and six
wheel-unloading guard events during stochastic exploration. Preserve every
event; the final deterministic evaluation is a separate measurement.

| Paired general cases | Retained source tracking/full | New final tracking/full |
|---|---:|---:|
| Base |49/50, 42/50|6/50, 2/50|
| Earlier arm |87/88, 80/88|0/88, 0/88|
| Coupled |2/8, 2/8|0/8, 0/8|
| Supported open-jaw arm |12/18, 12/18|0/18, 0/18|
| Total |150/164, 136/164|6/164, 2/164|

All original initial-state and reference identities match the teacher. The
retained source reproduces its previous category counts. No unsafe-force,
wheel-lift, invalid-physics or retention flag trips in either deterministic
general evaluation. Early tracking stops limit exposure; this does not prove
that continuing a failed trajectory would be safe.

The failure is already visible on a simple reachable pitch-up case: the final
exits the 10 mm tracking tube after 1.45 s. The reference hand rises about
6.2 mm while the measured hand moves down about 2.7 mm. Base error is only
0.086 mm at that stop. This is an arm-command/tracking failure, not evidence
that a larger base region or a longer deadline would solve the case.

[Recorded pitch-up comparison](../../paper/data/whole_body_control/mars_whole_body_validation_v2/pitch-up-failure.svg)
uses the unchanged source and final rollouts; it introduces no new simulation.

Actual TAMP pregrasp is **0/6 for source and final**, including **0/2 audit
cases** for each. All twelve failed runs confirm a stop; no force, wheel,
invalid-physics or retention flag trips. Preserve the exact timeout, tracking
envelope and generic operation-failure codes rather than inferring hidden
causes. These primitives end before closure/lift/carry/release and cannot be
reported as full pick/place results.

The first paired TAMP attempt was invalid instrumentation: an evaluation
script lacked a multiprocessing `__main__` guard, so spawned children re-ran
startup and exited before execution. Preserve those twelve invalid rows and
logs separately. After guarding startup, rerun all twelve TAMP measurements
with the same frozen weights and cases; verify and reuse the valid complete
general rollouts. The numbers above use only that corrected execution. No
retraining or physics change is hidden in this repair.

## Independently completed jaw-unit curriculum

The fixed 262,144-transition trial and all its planned post-tests now complete
at `0fba6121`. Its 1,150-test receipt, teacher164/164, fresh teacher12/12,
source admission and exact self/parent critic-only warmup checks remain bound
to that implementation, not the later GRU implementation.

| General full success | Retained source | Prior supported final | Jaw-schedule final |
|---|---:|---:|---:|
| Base |42/50|43/50|40/50|
| Earlier arm |80/88|63/88|53/88|
| Coupled |2/8|3/8|2/8|
| Supported arm |12/18|16/18|4/18|

Earlier arm tracking stays87/88, but34 cases fail base jerk. Supported tracking
is16/18 while full success is4/18: eleven cases exceed the teacher-relative
force budget, with additional jerk failures. Peak supported force is4.77 N,
below the independent unsafe-force threshold; relative smoothness/force
failure is not relabeled a damaging collision. All164 reset/reference pairs
match, and no expanded/fresh tracking hard-safety flag trips.

Fresh full scores are old5/12, near-straight5/12, translation8/12, supported4/12,
versus source12/12,12/12,11/12,3/12. Teacher reports pass all 48 cases; 24 are
cached as declared in the protocol. All old free-space audits retain12/12
tracking; supported final tracks10/12 versus source5/12.

Nine full public-tool comparisons complete: joint/legacy3/3 (a different mode),
matched analytic Cartesian0/3 due retention loss, learned0/3. Learned calls
are230/102/230, ending in timeout/envelope/timeout. All six failed
Cartesian-reference/learned runs confirm stop. Static or scheduled jaw
rescaling is not adopted, and no intermediate checkpoint is selected.

## Consequences for the next training stage

Keep the existing source and every independent gate. Before scaling this new
policy, inspect cold-start exploration, completion/progress credit and
per-critic advantage normalization. Initial action Gaussian std0.1 differs
substantially from the retained source's masked coefficient exploration;
zero/rare contact returns can also leave normalized critic-error advantages
with substantial actor weight. These are follow-up hypotheses, not isolated
causal findings. Smaller, physically scaled exploration and an easier
completion-producing distribution deserve controlled comparisons.

The current dataset also lacks empirical hand-only/free-base and partial-mask
coverage, actor-active terminal holds, physical grasp→lift/loaded placement,
and admitted disturbance/fault strata. Privileged critics do not provide
missing actor geometry, command-age or validity observations. Future loaded
rewards must distinguish necessary grip/support forces from harmful contact;
the current L1 force bound and all-wheel contact loads are not support or
damage certificates. Keep those extensions explicitly versioned.

[Compact results and checksums](../../paper/data/whole_body_control/mars_whole_body_validation_v2/results.json)
preserve both rejected trials, all denominators, original setup errors and the
evaluation repair. Raw traces, weights, training state and offline W&B remain
ignored. No further training is launched by this report. Hardware motion and
automatic promotion remain disabled.
