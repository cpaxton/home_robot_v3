# Tracking-majority retention training, 2026-10-24

This implements the opt-in [retention training plan](../plans/retention_training.md).
The policy retains its existing control reward. A separate advisory head predicts
contact occupancy from encoder/command history; it cannot authorize release or
override stop guards. This report is a training integration diagnostic, not
qualification of a stronger controller.

## First smoke experiment

The first completed experiment added 2048 PPO steps to a compatible legacy
transformer checkpoint using zero-influence initialization for the new retention
inputs. The PPO optimizer was fresh; its deterministic actor outputs were
preserved by the input expansion. The existing checkpoint was not overwritten.
The local source bundle was
`logs/manipulation/carry-ppo-v1/seed-31/step-200704`, not a qualified public-task
controller. Robot/action/model compatibility checks remained enabled.

Training used one CPU worker, seed 44, 8 s horizons, target scale 0.25,
randomization scale 0.05 and offline scalar W&B tracking. The preset reset weights
were 70/10/12/5/3 percent reach/pregrasp/carry/place/disturbance. Actual rollout
step shares were 64.31/7.62/21.92/6.15/0 percent. Disturbance was not sampled in
this short run; expected reset probabilities are not a coverage guarantee.

The occupancy head saw 1062 negative and 544 positive training samples. Its
episode-disjoint training audit had 31 positive and 370 negative frames: 100%
held recall, 17.03% empty-hand false positives and Brier score 0.0983. These frames
are temporally correlated, not 401 independent grasp trials.

An additional 12-attempt diagnostic exposed a failure hidden by easy empty-hand
examples: all 117 labeled closed-empty frames were predicted held. Loaded frames
also had 100% recall. Six reset/reference admissions failed, leaving three admitted
episodes in each category; every rejection remains in the raw report. This
development diagnostic is not an independent calibration result.

The raw encoders do distinguish these examples. In a selected empty hand,
gripper position and command error approach zero. A selected held payload blocks
closure around 0.344 rad with approximately -0.344 rad command error. The
estimator therefore needed stronger counterexample sampling rather than an
invented object-present observation. It remains advisory and uncalibrated.

The final implementation balances auxiliary held/empty minibatches and reserves
a quarter for retention-command/empty-hand examples when available. This changes
neither policy reset weights nor PPO reward. A continuation uses the extended
bundle with the PPO optimizer restored; the advisory head restores weights and
starts a fresh optimizer/replay. A separate report records that continuation.

## Balanced-head continuation

Seed 45 added 4096 PPO steps, reaching 6144 since the warm-start. Its training
audit had 318 held and 659 empty frames with 100% recall and no false positives.
However, a new teacher-driven diagnostic at seeds 510000–510005 admitted all
twelve attempts and still produced **67.52% false positives** on 234 closed-empty
frames, with 100% recall on 234 held frames. Zero training-audit errors did not
establish generalization to a different controller's motion.

The final estimator removes arm velocities, arm tracking errors and previous
whole-body actions from its inputs. It uses gripper-local measurements, commands
and retention intent only. A new feature-contract version makes this change
explicit and reinitializes the advisory head while retaining the PPO optimizer.
The failed development heads remain loadable and their reports are preserved.

## Gripper-local head and paired new-seed diagnostic

Seed 46 added another 4096 PPO steps, reaching **10,240** since the warm-start.
The PPO optimizer was restored and the changed advisory feature contract
explicitly initialized a new head. Actual rollout step shares were **60.64%
tracking**, 3.61% pregrasp, 25.51% carry, 7.01% placement and 3.22% disturbance.
All five task categories were sampled; tracking remains the majority of steps.

The new training audit had 318 positive and 608 negative frames, with 100% held
recall, zero empty-hand false positives and Brier score 0.00946. These repeated
audit frames do not provide independent calibration.

Both the previous arm/action head and the new gripper-only head were then tested
on the same **new seeds 610000–610005**, with six closed-empty tracking attempts
and six loaded holds. All twelve admitted. Teacher actions drive both tests;
the saved physical metrics, timestamps, gripper positions/errors/velocities and
contact labels match exactly. Different policy actions cannot explain the
comparison because neither actor drives these diagnostic rollouts.

| Head | Held recall, 234 frames | Empty false positives, 234 frames | Empty Brier score |
| --- | ---: | ---: | ---: |
| Development arm/action head | 100% | 56.84% | 0.3064 |
| Gripper-local head | 100% | 0% | 0.01087 |

The 234 frames in each column come from six episodes, not 234 independent
objects or grasps. The threshold is an uncalibrated 0.5; no execution decision
uses it. This is an encouraging initial occupancy diagnostic, not evidence of
force closure, stable retention or real-world object detection.

The final full-difficulty validation still gives **0/1 successes in each of the
five categories**, just as at initialization. These short, 8 s-horizon smoke
runs establish that the mixed training, warm-start/resume, separate head,
normalization, export and scalar logging execute; they do not demonstrate a
better controller. Performance experiments need adequate horizons, paired
denominators and longer training. No checkpoint was selected for promotion.

Complete run metadata, source/model identities, metrics, checkpoint manifests,
validation failures and diagnostic samples are archived alongside summaries:

- [Initial smoke](../../paper/data/whole_body_control/retention_training_smoke_v1.json)
- [Balanced-head continuation](../../paper/data/whole_body_control/retention_training_balanced_v2.json)
- [Gripper-local continuation and paired diagnostic](../../paper/data/whole_body_control/retention_training_gripper_v3.json)

The adjacent `.json.gz` files contain the full reports. Learned controller
weights remain local, identified by checksums; they were not uploaded as W&B
artifacts or deployed. All three completed runs use offline W&B scalar logs.
An earlier attempt failed to start W&B inside the process sandbox before any
PPO updates; the successful runs allowed its local loopback service and kept
its cache directories under `/tmp`.

The final source passed **603 regression tests** with no failures, errors or
skips. The archived source-bound
[receipt](../../paper/data/whole_body_control/mars-retention-regressions-v3.receipt.json)
and adjacent JUnit XML cover the existing RL/core/TAMP/agent/guard suites plus
the new mixture, label, reward-equivalence, estimator and compatibility tests.
Relevant lint, runtime package parity and the paper build pass. This software
regression evidence does not replace the task-performance gates above.

One full-difficulty validation episode per category gave no policy successes
before or after the first smoke run. This is a small integration run, not
evidence of retention or tracking improvement. The original public teacher
16/30 result remains separate and bound to its earlier source snapshot.

## Required follow-up

Evaluate closed-empty, loaded, supported, slipping and released transitions with
new seeds and held-out grasp/object distributions. Report episode-level
uncertainty as well as frame-level recall/false positives, and calibrate on an
independent split. Compare adequate paired tracking and loaded-task samples with
the original checkpoint and teacher. Preserve all reset failures and physical
guard events. Broad object geometry and calibrated actuator/force randomization
remain future work. No hardware command, deployment or controller promotion
was performed by these experiments.
