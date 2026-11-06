# Expand hand translation coverage before another refinement

Keep the retained controller. The prior refinement does not establish reliable
whole-body or loaded TAMP execution. A reference-distribution audit finds a
specific gap worth testing before increasing training time again.

Of 60 arm training references, 18 have orientation excursion at most 0.005 rad.
Those move the hand at most 11.94 mm, with sampled linear speed at most
7.46 mm/s. The three actual retimed TAMP pregrasps move about 24.4–24.6 mm,
reaching 11.86–11.93 mm/s. Their elbow reference reaches 0.125–0.135 rad/s;
the old training maximum is 0.075 rad/s. These are reference comparisons,
not evidence that this gap alone causes the policy failures.

Twenty training references stay within 3 mm of their start; orientation-only
motion is legitimate in some of these. With the actual endpoint tolerances
(3 mm, 0.02 rad), six stay strictly inside both bands throughout and four more
lie on the angular boundary. Do not mistake a larger 0.05 rad reporting band
for the training threshold.

![Reference excursions](../../paper/data/whole_body_control/mars_tamp_translation_audit_v1/coverage.svg)

The new development suite preserves all 122 prior cases, physics, thresholds
and validation seeds. Eight extra arm families combine two near-straight
postures with signed elbow/shoulder translations while preserving hand
orientation. Existing amplitude scaling yields 24 extra cases: 146 total,
50 base / 88 arm / eight coupled. New families receive half of arm sampling
mass; prior families retain the other half. Overall control-step allocation
remains 30% base / 50% arm / 20% coupled.

Only the extra families use a 2.5 s minimum duration. The original minimum
remains 3 s. Retiming always respects the existing physical speed, acceleration
and jerk bounds; the option can only slow a compiled path. Caliban's isolated
geometry preview admits 24/24, with excursions 4.89–25.32 mm and joint velocity
at most 0.15 rad/s. This uses a mixed development snapshot for reset/FK geometry
only and is explicitly ineligible for qualification. All1,123 fresh source-bound local regressions now pass, including the nine
static tests and native derivative-bound test. The physical teacher passes
146/146, reproducing every previous122-case metric, reward, code and
reset/reference identity exactly. The separate frozen translation teacher
passes12/12. [Preflight evidence](../../paper/data/whole_body_control/mars_tamp_translation_audit_v1/preflight.json)
keeps every case and source/report digest. This establishes achievable teacher
references under the admitted simulator, not learned-policy success.

Twelve additional continuous-posture translation inputs were frozen before
training, with disjoint seeds 34400000–34400011. Preserve their failed admission
and outcomes; never select seeds based on success. The
[compact audit and input cases](../../paper/data/whole_body_control/mars_tamp_translation_audit_v1/distribution.json)
retain source digests, per-case excursions and speeds, and geometry-preview
provenance. Raw traces remain ignored.

The declared next trial starts from the retained actor, uses trainable arm,
base and motion context, unchanged grip output, the prior objective schedule,
seed/rates/noise, and a fixed 262,144-transition final. The current stop latch
is included; this is not an isolated distribution-only experiment relative to
the pre-fix context trial. First establish source-fidelity admission and exact
critic-warmup physical behavior. Evaluate every original and added case, both
older twelve-case audits, the new twelve-case audit, and actual public-tool
first transfers. Keep every intermediate without selecting one on validation.
No automatic promotion and no hardware motion. Loaded-grasp planning remains
a separate diagnostic; free-space training is not a retention qualification.

The first training entrypoint stops before source admission or any PPO update:
offline W&B cannot open its local IPC socket inside the sandbox. Preserve its
output and error separately; retry the unchanged recipe with local socket
access and caches under `/tmp`. The receipt and both teacher reports are
reverified against the same immutable source before retry. Offline W&B now
starts successfully; source-fidelity initialization precedes optimization.
The final-test supervisor is restarted with the same inputs after archiving its
upstream-failure state. There is no checkpoint selection or recipe change.

The complete target plot is generated at
`logs/eval/mars-tamp-translation-v1-teacher/targets.svg`. To inspect one case
without running a simulator, use the existing report plotter:

```bash
PYTHONPATH=src .venv/bin/python -m emet.rl.manipulation.mastery plot \
  --preflight logs/eval/mars-tamp-translation-v1-teacher/report.json.gz \
  --case-id train:arm:tamp_translation_A_elbow_translate_positive \
  --output logs/eval/mars-translation-A-preview
```

Use a fresh output directory. Visualization sub-selection never changes an
admission, training or final evaluation gate.
