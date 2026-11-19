# Native closure-position pilot — November 14

Source `13d22945` compares the existing default close preset with Stretch's
existing loose-close position preset. Both use the same sampled retention
checks, model, 120-second planning budget, geometry and physical thresholds.
All cases retain their original poses, instructions and final regions. This
is privileged stationary simulated Stretch development, not training, hardware,
observed manipulation, a force-control experiment or held-out qualification.

| Preset | Original case | Full task | Transfers | Receipt |
| --- | --- | --- | --- | --- |
| Default (same-source ablation) | swap-right25 | fail | 1/3 | post_lift_hold / payload_not_retained |
| Loose | swap-right25 | pass | 3/3 | complete verified goal receipt |
| Loose | familiar control | fail | 1/2 | post_lift_hold / payload_not_retained |

The fixed loose preset is **not a general solution**: it passes the prior swap
failure but fails the familiar control. Candidate outcomes are 1/2 tasks and
4/5 declared transfers. The default ablation remains separate. The one observed
invalid-handle check rejects without an additional command. There are no
forbidden commands or infrastructure failures. Original 4/8 native and 3/4
Caliban development records are unchanged.

The new failure witness distinguishes motion, post-lift hold, loaded handoff
and post-release support. It passes through the primitive and task APIs as
finite-vocabulary JSON without private body IDs. The original 20 mm / 0.1 rad
retention bound is now checked at every sampled hold update, so a late violation
cannot be hidden by a later successful handoff sample. This strengthens sampled
coverage; it does not establish detection between samples or change the limits.

All **324 affected tests pass**, with no failures/errors/skips. These unit and
contract results do not turn the failed physical task into a pass. The parent
full-configuration failure/skip remain outside this change.

A private reconstructed-control probe motivated the preset comparison, but its
reconstructed baseline also passed: it is neither exact replay nor live task
acceptance, and it cannot establish the physical cause. The next hypothesis is
a position target relative to measured contact closure, rather than one fixed
preset. It must preserve the model/thresholds and pass actual replay before
adoption. No force or passive-pad sensor is assumed on hardware.

[Protocol](../../configs/eval/tamp_native_retention_pilot_20261114.json) and
[compact source/artifact manifest](tamp_native_closure_pilot_20261114.json).
Raw attempts are retained in ignored `.tamp-runs/retention-20261114/`.
