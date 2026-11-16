# Mars continuation regression and general tracking audit

The two 16,384-step settling-reward pilots completed, including automatic replay.
Neither improves the original 65,536-step checkpoint on the selected moving
poses. Goal-gated settling improves relative to ordinary continuation, but both
remain 0/3 public programs and 0/6 transfers. All six failed public executions
stop on the existing tracking envelope during pregrasp with confirmed stops.

| Six-second development result | Original | Continued objective | Goal-gated objective |
| --- | --- | --- | --- |
| Nearby position (mm) | 0.40–0.42 | 1.94–1.98 | 1.13–1.18 |
| Nearby orientation (rad) | 0.023–0.025 | 0.038–0.040 | 0.023–0.024 |
| Wrist orientation (rad) | 0.054–0.059 | 0.104–0.108 | 0.076–0.080 |

Each range spans two cases, not a statistical confidence interval. Both pilots
retain more than five seconds of continuous strict quiet in their two stationary
probes. Starting validation matches the source checkpoint exactly in all five
categories, locating the regression after updates rather than initial loading.

The first continued-objective public replay presents an 11.86 mm downward
reference displacement while the measured hand descends 2.00 mm. Goal gating
improves this to 3.67 mm descent for a 13.59 mm reference displacement, still
crossing the unchanged 10 mm error bound. Its maximum base XY error is 0.76 mm
and yaw error 0.0082 rad. The failure occurs before grasping. Measured joint
velocities respond to commands; this is insufficient learned task motion, not
evidence of a completely stalled actuator.

## Offline action counterfactuals

`normalization_action_audit_v2.json` swaps old/new network weights and old/new
normalization on identical recorded observations from all twelve pose probes.
Each recorded policy/normalization pair reproduces its recorded actions within
1e-5 before counterfactual comparisons. Changed weights have approximately 4–9 times the
arm-action RMS effect of changed normalization alone in moving-goal probes.
This identifies where command differences arise on these observations; it does
not prove counterfactual rollout success or isolate training's causal factors.

The source run ends at learning rate 1e-5. Both continuations restart a fresh
optimizer at 5e-5, and the first reported approximate KL is about 0.056 against
target 0.01. Observation-statistics prior count is reduced to 1,024 before it
adapts again. The four-way diagnostic in the linked plan separates learning-rate
restart and normalization. Architecture and training task distribution remain
fixed during this experiment.

## Review repairs and main integration

Latest `main` (`61da55cb`) merges cleanly into the isolated controller branch,
including agent/harness changes. The primary workspace remains on its separate
review branch.

- Running retention cost now uses current translation/rotation error. Peak
  evidence and the original sticky retention-loss thresholds remain unchanged;
  executor-authorized release clears the running term without erasing history.
- Risk estimates retain per-head predictions when supported. The overall result
  remains unknown when required safety labels lack independent calibration;
  `unavailable_heads` now explains each missing head. A safe policy with only
  negative outcomes cannot establish calibrated rare-event risk. Future
  simulator challenge data must be separate from promotion evidence and match
  the same controller/model; absent positives remain unsupported.
- The Stretch post-training oracle runner is packaged with the Python module.
  Its existing script is a compatibility CLI, and post-training no longer
  depends on repository-relative `parents[4]` script discovery. A package-layout
  test covers CLI startup and argument parsing without a repository/scripts
  tree; a full external Stretch oracle run was not repeated for this repair.
- An explicit training-only fixed-normalization option freezes copied running
  statistics while PPO still updates weights. Runtime preprocessing and bundle
  admission checks are unchanged.

Completed pilot data, original training sources, all numbered manifests, offline
scalar W&B records and full failed replay traces are archived in
`paper/data/whole_body_control/continuation_reward_ablation_v1.{json,json.gz}`.
The offline action audit includes its exact driver. No weights are uploaded and
no checkpoint is promoted. Fresh validation and launch status are appended only
after they actually complete.

The [general tracking plan](../plans/general_whole_body_tracking.md) defines the
broader validation matrix and subsequent curriculum. TAMP remains a bounded
acceptance subset, not the entire intended control task.

The merged implementation passes **756 fresh source-bound regression tests**.
The JUnit XML and verified receipt are archived under
`mars-general-continuation-v1-regressions-20261026`. Additional agent/skill/tool
and evaluation-harness integration checks pass after the main merge. Runtime
package parity, focused tests and relevant lint/whitespace checks pass. The
retention-cost repair affects loaded rewards; it does not explain a pregrasp
failure before retention is active.

The updated paper builds successfully. A general hand-only smoke case is admitted
geometrically but times out for both teacher and baseline; the baseline ends at
23.5 mm / 0.087 rad without wheel/force faults. This is negative diagnostic
evidence, not a successful generic-control baseline or proof of infeasibility.
The upcoming paired matrix retains teacher failures and actual retarget delivery.

The four matched 8,192-step pilots **started at 2026-10-26T15:05:39.001538+00:00**
from committed source `244dc1a2`, after verifying the
756-test receipt. They run sequentially on CPU 14 with CUDA
disabled and offline scalar W&B in `logs/manipulation/mars-general-continuation-v11/`.
The immutable `general_continuation_v1_launch.json` records all commands, sources,
configurations and the automatic evaluation driver. After all four complete,
`logs/eval/general-continuation-v11-20261026/` receives the original baseline plus
all candidates on ten general reach/retarget/push cases, six pose probes and
three public programs each. Teacher and failed-admission outcomes remain visible.
Frozen variants assert identical normalization across every numbered checkpoint.
Launch is not a completion or improvement claim; status/exits remain local.
No automatic qualification or hardware promotion is enabled.

A later audit correction fixes base/arm RMS grouping; original per-action
statistics and replay outcomes are unchanged. The corrected v2 diagnostic and
completed four-way results are in the [recovery report](mars_recovery_translation_20261026.md).
