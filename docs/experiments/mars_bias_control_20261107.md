# Lower bias learning rate limits regression but does not qualify the controller

The matched 262,144-transition control changes the gain-bias learning-rate
multiplier from 300 to 1. Source, seed, data, reward schedule, exploration,
critic learning rate, normalization, physics and acceptance thresholds stay fixed.
Both training and every declared post-test complete at immutable `2227f58a`.
No intermediate checkpoint is selected.

| Same 146 cases | Retained source tracking/full | Previous LR300 final | LR1 final |
|---|---:|---:|---:|
| Base |49/50, 42/50|50/50, 42/50|50/50, 42/50|
| Arm |87/88, 80/88|84/88, 14/88|87/88, 68/88|
| Coupled |2/8, 2/8|4/8, 4/8|3/8, 3/8|

The smaller bias rate substantially reduces arm regression. It does not
preserve the source's full arm success: 19 arm cases fail base linear jerk and
one times out. Original 16-arm full success falls 15/16 to 10/16; the original
four coupled cases remain 0/4. Keep the source.

All three separate twelve-case audits retain 12/12 tracking. Full passes are
8/12, 7/12 and 11/12, versus source 12/12, 12/12 and 11/12. These failures are
base jerk. No recorded force, wheel, retention or invalid-physics flag trips in
the expanded free-space evaluation or these fresh audits; smoothness still fails.

All nine public-tool comparisons finish. Joint/legacy completes 3/3, matched
analytic Cartesian execution loses retention in 0/3, and the learned controller
completes 0/3, stopping at pregrasp after 55/250/55 calls. The two 20 g cases
cross the 10 mm path-error bound; the 50 g case times out. All six failed learned
or Cartesian-reference stops confirm. The joint baseline uses a different mode.

[Every checkpoint, case, original subset and post-test](../../paper/data/whole_body_control/mars_tamp_translation_bias1_v1/results.json)
and [matched learning curves](../../paper/data/whole_body_control/mars_tamp_translation_bias1_v1/learning-curves.svg)
preserve the rejected result. This trial narrows one optimization problem; it is
not a successful general whole-body controller or hardware qualification.

## Contact-distribution diagnosis and next bounded trial

Two witness-derived motions pass in free space for teacher, source and LR1 final
(2/2 each). With the existing supported/open-jaw reset, the teacher still passes
2/2, while both learned actors fail 0/2. These are controlled development probes,
not exact replays: reset poses are settled independently, paths are recompiled,
and the full TAMP timing, terminal correction and scene are not reproduced.

On 229 saved pregrasp inputs, changing only the phase label has a modest effect.
Measured jaw deflection is a stronger sensitivity: training's almost constant
aperture gives roughly 0.000115 rad empirical standard deviation. Small physical
deflection can therefore produce about 19 normalized units. An in-memory
physical-jaw-unit intervention approximately doubles median source arm velocity
toward the goal (1.24 to 2.60 mm/s), but does not complete any public transfer.
All 12 paired diagnostic episodes remain failures; one transformed final run
hits its inference deadline and is inconclusive for physical tracking. Every
failed stop confirms. No transform is exported or adopted. These sensitivities
do not establish a complete causal explanation or justify hiding jaw feedback.

[Complete contact/input diagnostics](../../paper/data/whole_body_control/mars_pregrasp_contact_audit_v1/results.json)
retain every sample, intervention and failure. They motivate adding real
supported open-jaw contact to the training distribution before another broad
extension. The new suite preserves all 146 earlier cases and adds six parents
at three amplitudes, giving 164 admission cases. It reuses existing fixture
initialization; no grasp/close/lift task manager runs during pose tracking.
The contact families receive 25% of arm sampling, or 12.5% of total control-step
budgets. Base/arm/coupled budgets stay 30/50/20 and normal tracking remains the
majority. No normalization, model, action filter or guard change accompanies it.

Before optimization: require fresh source-bound full regressions, teacher
164/164, exact old 146-case teacher replay, twelve fixed new-seed contact teacher
cases, source fidelity, and exact critic-only warmup behavior. The next fixed
final remains 262,144 transitions, with the LR1 recipe otherwise unchanged.
Evaluate all prior subsets, four fresh audits, all contact families and the
same nine public-tool comparisons. Retain the source unless accuracy, full
contract, safety and public-tool nonregression gates pass. Hardware remains off.
