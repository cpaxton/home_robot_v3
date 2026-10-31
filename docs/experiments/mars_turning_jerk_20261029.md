# Mars turning and jerk correction

The completed 262k-step policy learned accurate translations but failed full
mastery (0/10). A new opt-in final motor-command filter addresses the fore/aft
correction chatter. Replaying the **same actor weights without retraining** now
passes **6/10 full mastery** and **10/10 smoothness** checks on the ten frozen base
cases. Both splits pass hold, forward and reverse. Turning remains the deficit.
This is a declared execution-dynamics intervention, not a newly qualified policy.
The first 16k-step refinement checkpoint subsequently improves full mastery to
**8/10**, retaining **10/10 smoothness**. The bounded run has now completed at **9/10 full mastery**, with all ten smoothness
checks passing. The training left turn remains at 0.02725 rad heading error.

## Control change and measured effect

The old acceleration slew limited command magnitude changes but allowed abrupt
acceleration reversals after the extra velocity-feedback correction. Add an opt-in
critically damped second-order final velocity filter with experimental linear/yaw
jerk settings 0.24 m/s^3 and 1.2 rad/s^3. It runs at the physics/actuator cadence,
not only the 20 Hz policy cadence. Native speed/acceleration caps remain; hard
speed boundaries and fault stops take priority over smoothing. Stop clears the
filter velocity and acceleration states. Repeated calls at the same physics
instant cannot advance the filter. No physical state is reset by enabling it.

These are experimental simulator settings, not manufacturer jerk or damage
ratings. The filter adds command lag and needs separate disturbance/loaded/hardware
validation. Legacy configuration defaults retain the previous behavior.

| Training case | Old final peak linear jerk | Same actor + filter | New terminal quiet | Full mastery |
| --- | ---: | ---: | ---: | --- |
| Hold | 1.31 m/s^3 | 0.014 m/s^3 | 9.0 s | Pass |
| Forward | 1.97 m/s^3 | 0.026 m/s^3 | 9.0 s | Pass |
| Reverse | 2.19 m/s^3 | 0.017 m/s^3 | 9.0 s | Pass |

Validation hold/forward/reverse also pass. Training left/right yaw errors remain
0.0453/0.0147 rad, versus the unchanged 0.01 rad tolerance. Validation left/right
errors are 0.0075/0.0258 rad; the left case has only 1.8 seconds of final quiet.
Every case now passes the paired teacher-relative force/jerk checks. The teacher
passes **26/26** across all four stages under the new execution settings.

## Reward timing and yaw correction

Rewards already arrive **every 50 ms (20 Hz)**. Completion is a separate event;
learning never waits two seconds for all feedback. The opt-in objective adds:

- Coarse capture credit plus contract-scale fine base/yaw credit. The original
  0.2 rad yaw density alone was too broad relative to 0.01 rad acceptance.
- Base translation/yaw progress measured before/after the same newly active goal.
  Advancing the reference clock cannot manufacture progress reward. Position
  progress retains the allowed base XY region rather than encouraging submillimeter
  corrections inside it.
- Smooth terminal quiet credit every tick using measured arm velocity and all six
  floating-base velocity DOFs. It declines continuously with motion rather than
  dropping to zero at the stop threshold.
- A robust normalized jerk cost with explicit per-axis experimental scales.
  Joint/angular and linear units are normalized separately. This supplements the
  original physical force/retention costs and independent stop guards.
- The one-time completion bonus waits for **two continuous quiet seconds**, matching
  the mastery gate. Dense pose/progress/quiet rewards continue before and after it.

`mars_tracking_mastery_v2.json` preserves v1 case identities, reference-generation
settings, accuracy/holding/force/jerk gates and horizon exactly. Its loader rejects
changes to those frozen fields. Only objective and declared execution settings
change. These reused development cases are not new held-out qualification.

Exact optimizer resume rejects changed objective/filter settings. Weight transfer
records the dynamics change and starts a fresh optimizer. TAMP checkpoint attachment
also applies the manifest's physical slew/jerk settings, so execution cannot silently
omit the filter used in simulation. No teacher residual, action imitation, task-stage
actor input or hardware wiring is added.

## Verification and bounded next run

The focused control/objective/mastery/chassis set passes **38 tests**, including
continuous reward shape near yaw tolerance, all-DOF quiet scoring, dimensional jerk
normalization, command derivative bounds, signed step response, repeated-time
invariance, hard stop and actual two-second bonus timing. Current-source full
controller regressions passed **873 tests** for that runtime. The single refinement
below subsequently completed;
its initial checkpoint retains 6/10 full mastery. The first post-update 16k
checkpoint reaches 8/10; these are interim results, not a final training outcome.

After the gate passes, run **one** bounded 131,072-step base-only refinement from
the completed seed-90 actor/critic with a fresh optimizer. Use learning rate 5e-5,
base exploration standard deviation 0.01 and arm/gripper standard deviation 0.03.
Evaluate both complete splits every 16,384 steps against the paired v2 teacher.
Keep one CPU14 environment, CUDA off, BLAS/OMP/Torch one thread and offline scalar
W&B. No repeated seeds, automatic extension, stage advancement or promotion.

The starting intervention and later learning outcomes must remain separate: reward
changes cannot explain action improvements in a frozen-actor replay. Initial and
final checkpoints will reveal whether training learns turns while preserving the
six newly stable cases and all smoothness gates.

Runtime revision: `fd0c864c`. Teacher traces, frozen-actor intervention, suite,
regression receipt/XML and the actual refinement launch are archived under
`paper/data/whole_body_control/turning_jerk_v2_setup/`. Live metrics/checkpoints are
under `logs/manipulation/mars-mastery-objective-v2-seed90/`. Weights remain local.

The revised suite is explicit in commands, so legacy comparisons remain available:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 taskset -c 14 .venv/bin/python -m emet.rl.manipulation.mastery preflight \
 --suite configs/benchmarks/mars_tracking_mastery_v2.json \
 --output logs/eval/my-v2-teacher
```

Generate a fresh full regression receipt with the existing mastery workflow before
training. The checked-in launch driver records the complete bounded warm-start
command; use fresh output paths rather than overwriting a checkpoint or receipt.

## First refinement checkpoint (16,384 steps)

All six hold/translation cases remain mastered; one training and one validation
turn now also pass, giving **4/5 train, 4/5 validation, 8/10 total**. All ten cases
retain their paired smoothness passes. This is the first scheduled checkpoint,
not validation-based checkpoint selection. The bounded run subsequently completed
as recorded below.

| Turn | Initial yaw error | At 16k | Final quiet at 16k | Result |
| --- | ---: | ---: | ---: | --- |
| Train left | 0.0453 rad | 0.0316 rad | 0 s | Timeout |
| Train right | 0.0147 rad | 0.0048 rad | 3.90 s | Pass |
| Validation left | 0.0075 rad | 0.0045 rad | 3.75 s | Pass |
| Validation right | 0.0258 rad | 0.0097 rad | 0.15 s | Timeout |

The last case reaches heading tolerance too late to satisfy continuous terminal
holding; the training left case still misses heading accuracy. Both remain
failures under the unchanged 12-second horizon and 0.01 rad / two-second gates.
Full first-checkpoint traces and bundle identity are archived with the setup.

## Completed refinement and filter follow-up

The 131,072-step refinement finished with **4/5 training, 5/5 validation, 9/10 full
mastery**, retaining all ten smoothness passes. From 98k onward, each scheduled
checkpoint and final `last` has that same count. No validation-selected checkpoint
or promotion is claimed. The remaining training left turn ends at 0.02725 rad;
its raw command remains too weak under both filtered and unfiltered execution.

A subsequent actuator audit finds that the added filter has 0.65 s small-step rise
time and that removing it from this actor destroys smoothness. The existing 9/10
controller is preserved; a separate bounded no-added-filter learner is now running.
See the [filter audit and direct-command learning comparison](mars_filter_audit_20261029.md).
Prior interim checkpoint results above remain historical.
