# Mars wrapper reward training pilot

The frozen runtime is `5f0eb3c2`, with 3,533 source/test/asset inputs. Source archive
and checksums are retained in `logs/recovery/wrapper-reward-5f0eb3c2-source-record.json`.
This is simulator learning; no hardware, default promotion or loaded-carry claim.

The declared comparison starts both reward arms from the retained scoped-feedback
v4 actor (weights SHA-256 `46291c71ea4dccd93feaf94d4b0e88f3deb1ee5945667964300f3c9f1c165813`).
Both use fresh critics and identical seed 107, distribution and optimizer settings.
The contract-reward arm runs first; the legacy-reward control follows. Each has
16,384 decisions, including 1,024 critic-only decisions. Actor learning rate
falls from 5e-5 to 1e-5; action standard deviation starts at 0.02; rollouts are
256 decisions, with three PPO epochs, 64-sample minibatches, clip 0.05 and
KL limit 0.01. Checkpoints are saved every 2,048 decisions and training state
after every rollout. Final checkpoints are evaluated without audit selection.

The outer distribution remains 50/50 general/TAMP by physical control ticks.
Within general, 80% is coupled hand tracking and 20% base-only control. Twelve
new training cases cover forward/reverse travel, both yaw signs, full/XY-only/free
base masks and nearby FK-witness hand trajectories. Twelve fresh audit cases
have disjoint seeds and perturbed motion parameters. Existing ten-case coupled
and six-case TAMP readouts remain separate regression cohorts. TAMP targets the
unloaded, open-jaw grasp-pose prefix; this does not train loaded grasp retention.
Smoothness, force and physical guards remain active; randomization starts at zero.

The same sensor-only actor is trained on separately fingerprinted general and
TAMP simulator plants. A general-plant export is not qualified checkpoint
admission for the other plant, an agent, or hardware. Teacher labels are used
only for admission/comparison and never enter PPO loss. Neither existing held-out
cases nor new audit cases enter training.

Exact cases, protocol, source-bound supervisor and experiment runner are preserved
under ignored `logs/recovery/wrapper-reward-training-v1/`; all weights, updates,
initial/final metrics and offline W&B runs go under ignored
`logs/manipulation/wrapper-reward-training-v1/`. Do not commit raw compressed traces
or weights. Compare physical outcomes, not raw returns across different reward units.

At launch, full regression job `20261116_224138_a8c555` uses CPU15; teacher
admission/baseline job `20261116_224526_d8b8b3` uses CPU14. The latter must pass
all declared training references and six unchanged TAMP teacher cases with
confirmed stops and no scored safety faults. Audit failures remain counted.
Learning waits for both actual jobs to finish successfully and verifies the
fresh complete source receipt. Queued prerequisites are not reported as training.
The simulator viewer on port 8766 is stopped to free CPU14 during this pilot.

The v1 admission completes at **22/24 new teacher cases**, **11/12 training**,
and **6/6 TAMP prefixes**. Both reverse free-base hand cases (training and audit)
trip unsafe-contact stops; learning never starts. These failed cases and the
rejected manifest are retained. The v2 protocol changes only the affected
training case to an XY-constrained, yaw-free base mask and records the complete
delta; the free-base audit failure remains in the denominator. V2 uses the same
runtime and training budget. Revised admission job `20261116_224836_3e01ed`
precedes training job `20261116_224923_96115d`; the latter is explicitly waiting
for revised admission and the full regression receipt. Exact v2 producers and
protocol are under `logs/recovery/wrapper-reward-training-v2/` and results under
`logs/manipulation/wrapper-reward-training-v2/`. Scored accuracy is captured
before stop; v2 safety flags also include the completed stop interval.

The revised teacher passes all **12/12 training references**. Its reverse
free-base audit remains an unsafe-contact failure and is not removed. Final
admission additionally waits for all six TAMP prefixes and baseline measurements.
The paired pilot is deliberately small: one seed, fixed budgets, no confidence
or hardware-reliability claim. General exports retain their explicit scoped
wheel-feedback execution contract.

Fresh full regressions finish at **1,563 passing tests**, zero failures/errors/
skips, in **1,110.89 s**. The receipt verifies against the frozen runtime and
unchanged canonical control/test sources. Revised admission/baseline completes:

| Controller / cohort | Training | Fresh audit | Existing coupled | TAMP prefix |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 12/12 | 11/12 | separate earlier readout | 6/6 |
| Retained actor | 11/12 | 9/12 | 6/10 | 6/6 |

All 40 retained-actor baseline episodes confirm a stop. Its three new-case
tracking timeouts and one free-base audit contact stop remain failures.
The revised teacher's free-base audit contact stop also remains counted.

Actual new-reward learning starts in managed job `20261116_224923_96115d` after
both prerequisite jobs complete. Offline W&B run: **`mqbn78v9`**. At the first
recorded actor-update checkpoint, training has completed **1,536 decisions /
1,781 physical ticks**, including 1,024 critic-only and 512 actor-update
decisions. A noisy training TAMP rollout reports `object_changed_replan`;
training episodes are not deterministic qualification results. No learned
accuracy improvement is claimed before the fixed final evaluation.

The legacy control follows automatically. Managed finalizer
`20261116_225632_6bdce8` waits for both fixed runs, checks matched initial states,
source/model/reference identities and first rollout actions, then writes
`physical-comparison.json` and a separate offline W&B outcome table. Final
metrics include accuracy, settling, drift, jerk, forces and safety flags;
compare physical outcomes rather than returns from different reward units.

The first exported checkpoint at **2,048 decisions** reloads successfully,
contains finite actor weights, and changes **10 actor tensors** after its first
1,024 actor-update decisions. Its weights SHA-256 is
`6527f10793fbe73b058b16ba9ff0db351842d2baaba79ea13a3be2d7ac3f5442`.
Base rate-feedback mode and the navigation-feedback configuration are preserved.
This validates actual learning/export plumbing, not checkpoint performance.
The run continues toward its fixed 16,384-decision budget; consult the live
`contract/run.json` for current counts rather than treating this launch report
as final results.
