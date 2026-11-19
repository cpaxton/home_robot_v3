# Native contact-relative position-hold pilot — November 14

Frozen source `ba7430ea` closes with the ordinary preset, waits for motor
settling, then targets the midpoint between the measured encoder position and
the original close target. A second settling interval shares the original
ten-second closing budget. No force measurement, passive-pad feedback, contact
parameter change or acceptance-threshold change is introduced. The native
path remains privileged, stationary, simulated Stretch development.

| Condition | Original case | Full task | Transfers | Failure |
| --- | --- | --- | --- | --- |
| Fixed loose (same-source ablation) | familiar control | fail | 1/2 | payload retention during loaded placement |
| Contact-relative | familiar control | pass | 2/2 | — |
| Contact-relative | swap-right25 | fail | 1/3 | lift_motion / payload_not_retained |
| Contact-relative | shift15 | pass | 2/2 | — |

The candidate passes **2/3 tasks, 5/7 declared transfers**, with both observed
no-command handle checks passing. The candidate produces 1,477 audited commands,
zero forbidden commands and zero infrastructure failures. All **336 affected
tests pass**, with zero failures/errors/skips. The familiar control is preserved,
but the swap remains a failed task. This is not a general retention solution.
Historical native 4/8 and Caliban 3/4 populations are unchanged.

The failed cube lift reaches 0.100110 rad rotation from its original grasp
reference against the unchanged 0.1 rad bound. Its public task receipt preserves
the typed lift-motion witness. The loose ablation instead passes lift/handoff
and fails during loaded placement; that receipt has the public retention code
but no finer verification witness. These distinct failure locations are retained.

Private FK/collision reconstruction finds initially one edge contact per finger
in the failed rotated pinch, versus multiple opposing-face contact points in
the successful control. The selected swap jaw direction is rotated from the
box faces; the current search can accept that lower-height candidate before
trying a higher, face-aligned candidate. These sparse geometry reconstructions
are not force measurements, exact dynamics replay or causal proof. They motivate
testing geometric candidate ordering rather than further treating one motor
target as a sufficient grasp-quality fix.

A reconstructed control-only component sweep also retains its failures:
the default, 50% and 75% position-error conditions pass that component; 25%
fails retention. It lacks original control/warm-start state and cannot substitute
for the failed live task. The midpoint was frozen before the live pilot.

All four live episodes use CPU 0–7 serially, the original case-file hash and the
unchanged 120-second planning gate. Resets are independent, without an exact
paired-state certificate. No held-out, hardware, observed manipulation, local
GPU/VLM, training or qualification claim is made.

[Frozen protocol](../../configs/eval/tamp_native_contact_relative_pilot_20261114.json)
and [compact source/artifact manifest](tamp_native_contact_relative_pilot_20261114.json).
Raw attempts, the diagnostic plot and private reconstructions remain under
ignored `.tamp-runs/contact-relative-20261114/`; prior fixed-preset failures remain
under `.tamp-runs/retention-20261114/`.
