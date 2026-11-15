# Native geometric grasp-order pilot — November 14

Frozen source `2e9b824c` ranks the existing single-box grasp candidates by jaw
alignment with measured collision-geometry face normals. Nearly aligned poses
retain their prior pitch/height order. Other/compound shapes keep the original
order. The candidate population, shared 256-call IK cap, 120-second planning
gate, model, retention limits and whole-footprint goals are unchanged. The
candidate also retains the prior encoder-relative position-hold policy.

| Condition | Original case | Full task | Transfers | Failure |
| --- | --- | --- | --- | --- |
| Aligned order + contact-relative hold | swap-right25 | fail | 1/3 | post_lift_hold / payload_not_retained |
| Aligned order + contact-relative hold | familiar control | pass | 2/2 | — |
| Aligned order + contact-relative hold | shift15 | pass | 2/2 | — |
| Aligned order + default close (ablation) | swap-right25 | fail | 1/3 | lift_motion / payload_not_retained |

The candidate remains **2/3 tasks, 5/7 transfers**. Both observed no-command
handle checks pass. There are 1,459 candidate audited commands, zero forbidden
commands and zero infrastructure failures. All **345 affected tests pass** with
zero failures/errors/skips; Ruff is clean. This preserves both familiar layouts
but does not solve the swap or establish an aggregate improvement over the
prior contact-relative pilot. Historical native 4/8 and Caliban 3/4 scores remain
unchanged. No new VLM baseline is claimed.

The swap completes cube lift at 0.096147 rad relative rotation, then reaches
0.100040 rad during the sampled hold against the original 0.1 rad bound. It
remains a failure. The ordinary-close ablation fails during lift at 0.100431 rad.
Typed receipts expose these substages; the implementation additionally exposes
`loaded_motion` for retention loss during placement. A near miss is not rounded
into a pass, and a later good sample cannot renew a violated reference.

## Fixed-base sensitivity

The failed task's planner log rejects better-aligned pregrasp poses by IK before
choosing a rotated pinch. A separate private IK probe removes the selected
0.175 rad yaw offset, retaining the captured arm start, joint margins and
3 mm / 0.1 rad IK tolerances. Across grasp heights 0/12/24 mm:

| Private world-X base offset | Aligned pregrasp orientation residual | IK |
| --- | --- | --- |
| 0 mm | 0.1529–0.1532 rad | fail |
| +10 mm | 0.1127–0.1130 rad | fail |
| +25 mm | 0.0563–0.0566 rad | pass |
| +40 mm | 0.0043–0.0044 rad | pass |

Grasp-pose IK has the same fail/pass split. Negative offsets and all raw
results remain in the artifact. This is evidence of base-pose sensitivity in
the numerical solver, not proof of global infeasibility, collision clearance,
an executable base path or task success. Only private qpos is shifted; no live
base is teleported and no manipulation score comes from the probe.

The next coherent feature is a checked unloaded base approach before grasp:
pure combined base/arm planning, swept-geometry validation, normal base
controllers, measured arrival, then fresh grasp refinement. The
[implementation/validation plan](../plans/2026-11-14_native_tamp_base_approach.md)
keeps unknown outcomes partial and requires original full-task replay before
any readiness promotion. The current follow-up remains draft.

[Protocol](../../configs/eval/tamp_native_face_alignment_pilot_20261114.json) and
[compact source/artifact manifest](tamp_native_face_alignment_pilot_20261114.json).
All four live trials are serial on CPU 0–7 with independent resets. Raw traces,
the comparison figure, planner failures and the private IK probe remain under
ignored `.tamp-runs/face-alignment-20261114/`. Prior attempts are preserved.
This is privileged stationary simulated Stretch development, with no training,
hardware, local GPU/VLM, observed manipulation or held-out qualification claim.
