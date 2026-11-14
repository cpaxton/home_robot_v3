# Supported-contact learning improves, but whole-body nonregression still fails

The fixed 262,144-transition trial and all declared post-tests complete at
immutable `6a110adc`. Training adds supported open-jaw contact at 12.5% of
control steps; the LR1 optimizer, rewards, normalization, physics, safety limits
and earlier cases remain fixed. The final is rejected and the original source
checkpoint remains retained. No intermediate checkpoint is selected.

## Supported-case contract (reviewed 2026-11-14)

These cases are open-jaw arm tracking in the deterministic supported pregrasp
fixture. Their joint deltas start at that fixture's measured arm pose; FK
constructs the hand reference, and joint-margin, collision and wheel-support
admission still apply. The fixture configuration is scoped to this reset and
does not change the next ordinary reset.

The `pregrasp` phase matches the runtime phase token. Grip correction is
executor-disabled in this phase, with retention and release both unauthorized;
these cases do **not** train grasp closure or establish object retention.
The base target remains active in all three SE(2) coordinates at a stationary
pose. It is not a masked-out or physically immobilized base: the tight arm-case
base contract and base jerk measurements still apply. Thus good hand tracking
can coexist with a failed combined contract due to base motion.

The existing [supported-fixture tests](../../src/test/rl/manipulation/test_supported_mastery.py)
cover case preservation, invalid reset configurations, open-jaw behavior during
steps and exact restoration of the following ordinary reset. This review found
no supported-contact implementation defect; the behavioral results below
remain unchanged.

| Earlier 146-case subset | Retained source tracking/full | Prior LR1 final | Supported-contact final |
|---|---:|---:|---:|
| Base |49/50, 42/50|50/50, 42/50|50/50, 43/50|
| Arm |87/88, 80/88|87/88, 68/88|87/88, 63/88|
| Coupled |2/8, 2/8|3/8, 3/8|3/8, 3/8|

The added 18 supported cases improve from source 12/18 tracking/full to final
16/18 tracking/full. On the twelve fixed new-seed supported cases, tracking
improves 5/12 to 9/12 and full success 3/12 to 9/12; the teacher passes 12/12.
This is evidence that contact-containing tracking can be learned. It does not
establish transfer to the complete public-tool task or preserve previous skills.

The main nonregression failure remains base motion during arm tracking. On the
earlier 88 arm cases, 24 final cases fail base linear jerk and one crosses the
tracking envelope. Original 16-arm full success falls from source 15/16 to 8/16.
The three earlier fresh audits achieve final full 5/12, 8/12 and 10/12, versus
source 12/12, 12/12 and 11/12. All retain 12/12 tracking. Across the complete
164-case evaluation, final arm tracking is 103/106 and full success 79/106,
versus source 99/106 and 92/106. Tracking improvement cannot hide the full-contract
regression. No unsafe-contact, wheel-lift, invalid-physics or retention flag trips
in the expanded or fresh tracking evaluations; relative force/smoothness gates
still fail in some cases.

## Actual public-tool execution

All nine comparisons complete. Joint/legacy succeeds 3/3, using a different
execution mode. Learned Cartesian execution remains 0/3: the two 20 g cases
cross the 10 mm pregrasp path tube after 55 calls, and the 50 g case times out
after 250 calls. These failure codes and call counts match the previous LR1
final, although this alone does not establish identical physical trajectories.

The matched analytic Cartesian reference also completes 0/3: two retention
failures and one wall-clock cycle deadline. That deadline run is inconclusive
for later physical behavior; do not relabel it as a retention failure. All six
failed Cartesian-reference/learned episodes confirm measured stop. No actor
fallback, task threshold change, model tuning or input transform is introduced.

## Verified launch and complete evidence

All **1,138** fresh source-bound regressions pass. Teacher admission passes
**164/164** and the **12/12** frozen contact audit. All earlier 146 teacher
physical cases reproduce exactly. Source-fidelity admission and self/parent
critic-only warmup comparisons pass before actor updates. The receipt also
verifies against the current code; documentation/data publication does not change
the tested implementation. These are local checks, not GitHub CI claims.

[Complete cases, checkpoints, older subsets, four fresh audits and nine tool episodes](../../paper/data/whole_body_control/mars_supported_contact_v1/results.json)
and [canonical learning curves](../../paper/data/whole_body_control/mars_supported_contact_v1/learning-curves.svg)
retain every outcome, artifact checksum and source identity. Raw traces, models
and offline W&B remain ignored. This research run is finished; no additional
training run or hardware test is active from this supervisor.

The next iteration needs to prevent wheel-command/smoothness regression during
arm learning and explain why the supported-contact gains do not transfer to the
actual pregrasp schedule. More training of the unchanged recipe is not supported
by this result. Loaded retention and full observed-agent/Herman acceptance remain
open.
